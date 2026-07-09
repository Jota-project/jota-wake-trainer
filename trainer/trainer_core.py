# trainer/trainer_core.py
"""
Entrenamiento real con openWakeWord (API 0.6.0+).

Historial: la versión anterior de este módulo llamaba a
`openwakeword.train.train_custom_model(...)`, una función que nunca ha
existido en la librería (ni en 0.6.0 ni en versiones anteriores). El diseño
original asumía que openWakeWord generaba negativos y aplicaba augmentación
"internamente" a partir solo de clips positivos — pero openWakeWord es un
clasificador binario: sin datos negativos no hay nada que aprender. Esa
combinación de dos fallos (función inexistente + falta total de datos
negativos) es la razón de que el entrenamiento se quedara siempre en
"pending".

Este módulo usa la API real: `openwakeword.train.Model` + `auto_train`,
alimentado con:
  - features positivas calculadas a partir de los clips grabados/sintetizados
  - negativos "duros" sintetizados con las mismas fuentes TTS del proyecto
    (variaciones fonéticas cercanas a la wake word + frases genéricas)
  - negativos generales precalculados (ver trainer/negative_data.py)

Y exporta a ONNX + TFLite (ver trainer/tflite_export.py) — el .tflite es lo
que carga el addon openWakeWord de Wyoming/Home Assistant en CPU.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, TYPE_CHECKING
import logging
import random
import warnings
import numpy as np
from rich.console import Console
from rich.markup import escape
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

if TYPE_CHECKING:
    from trainer.state import Project

from trainer.audio_utils import repair_clips

console = Console()

_IMPORT_ERROR: Optional[BaseException] = None
HAS_TRAIN_DEPS = False


def _silence_known_noisy_warnings() -> None:
    """
    Silencia warnings internos de librerías de terceros que ya hemos
    investigado uno por uno y confirmado que son puramente informativos —
    no indican ningún problema con los datos ni con el modelo entrenado.
    No se silencia nada de forma genérica (por categoría global): cada
    filtro apunta a un mensaje y módulo concretos, así que cualquier otro
    warning nuevo (que sí podría ser relevante) se sigue mostrando.

    1) openwakeword.data.augment_clips construye
       torch_audiomentations.Compose(...) sin pasar `output_type`, así que
       cada llamada dispara un FutureWarning avisando de que el valor por
       defecto cambiará de 'tensor' a 'dict' en v0.12/v0.13. Es un warning
       interno de la librería (no de nuestro código, no lo podemos arreglar
       pasando el argumento nosotros) y es puramente informativo mientras
       sigamos pineados en la serie 0.12.x (ver pyproject.toml — pin
       explícito con comentario de por qué).

    2) torchmetrics todavía importa `pkg_resources` en tiempo de import
       (torchmetrics/utilities/imports.py) para comprobar versiones de
       paquetes, y pkg_resources avisa de que esa API está deprecada. No es
       una acción nuestra ni de openwakeword — es torchmetrics comprobando
       sus propias dependencias al cargar. Mientras sigamos con
       `setuptools<82` (ver pyproject.toml), pkg_resources sigue existiendo
       y funcionando con normalidad; el aviso es solo un recordatorio de que
       algún día torchmetrics tendrá que dejar de usarlo.

    3) openwakeword.data.create_fixed_size_clip construye el array de
       relleno con `np.zeros(n_samples)`, que por defecto es float64 (numpy
       no infiere el dtype del clip original). audiomentations recibe ese
       array float64 y avisa de que lo convierte a float32 antes de
       procesar — la conversión ocurre de todos modos y es correcta, así
       que el aviso no señala ninguna pérdida de datos real.
    """
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module=r"torch_audiomentations\..*",
        message=r"Transforms now expect an `output_type` argument",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"torchmetrics\..*",
        message=r"pkg_resources is deprecated as an API",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"audiomentations\..*",
        message=r"Warning: input samples dtype is np\.float64",
    )


try:
    import torch

    import torchaudio  # noqa: F401
    if not hasattr(torchaudio, "set_audio_backend"):
        # torchaudio >=2.1 eliminó esta función; algunas versiones de
        # speechbrain / torch-audiomentations todavía intentan llamarla.
        # Con este stub se convierte en un no-op en vez de un crash.
        torchaudio.set_audio_backend = lambda *a, **kw: None  # type: ignore[attr-defined]

    _silence_known_noisy_warnings()

    import openwakeword
    import openwakeword.utils as oww_utils
    from openwakeword.data import augment_clips, mmap_batch_generator
    from openwakeword.utils import compute_features_from_generator
    from openwakeword.train import Model as TrainModel

    HAS_TRAIN_DEPS = True
except ImportError as exc:  # pragma: no cover - depende del entorno
    _IMPORT_ERROR = exc

from trainer.tflite_export import convert_onnx_to_tflite_in_subprocess, TFLiteExportError
from trainer.negative_data import ensure_negative_features

INSTALL_HINT = (
    "openWakeWord no está instalado con soporte de entrenamiento.\n"
    "Instálalo con:\n"
    "  pip install -e '.[train]'\n"
    "(y, si quieres exportar a .tflite además de .onnx: pip install -e '.[tflite]')"
)


@dataclass
class TrainingConfig:
    model_name: str
    output_dir: Path
    negative_mode: Literal["quick", "full"] = "quick"
    # steps=50000 y layer_size=128 son los valores por defecto de
    # openwakeword.train.Model/Model.auto_train (ver openwakeword/train.py:
    # `def __init__(self, ..., layer_dim=128, ...)` y
    # `def auto_train(self, ..., steps=50000, ...)`, este último descrito
    # explícitamente como lo que "produce modelos relativamente fuertes
    # automáticamente"). Este proyecto usaba steps=5000 (10x menos) y
    # layer_size=32 (4x menos, ~1/16 de los parámetros de la capa de
    # entrada) sin ninguna razón documentada — probablemente para que las
    # pruebas del wizard fueran rápidas durante el desarrollo, nunca
    # corregido para entrenamientos reales. Resultado real observado:
    # ok_jota.tflite pesaba 207 KB (un modelo bien entrenado ronda 1-4 MB,
    # p.ej. hey_jarvis.tflite ~1.2 MB) y no detectaba ni con voz clara — la
    # red no tenía ni capacidad (layer_size) ni pasos suficientes para
    # aprender la wake word, con independencia de la cantidad/calidad de los
    # datos de entrenamiento (que sí se habían mejorado por separado con
    # ruido/RIR reales). A steps=50000 y ~145 it/s (medido en el
    # entrenamiento real anterior) son ~6 min extra, no una diferencia
    # prohibitiva.
    steps: int = 50000
    max_negative_weight: int = 1000
    target_fp_per_hour: float = 0.2
    val_fraction: float = 0.15
    # Por qué 40 y no 8: confirmado con evidencia externa (no solo teoría)
    # que 8 rondas era muchísimo menos de lo que este recall=0.0 necesitaba.
    # Tras arreglar el reparto de clases por batch (_compute_n_per_class) Y
    # activar el logging real (_configure_training_logging), un
    # reentrenamiento con datos/pesos por lo demás correctos SIGUIÓ dando
    # "Final Model Recall: 0.0" de principio a fin. Investigando en las
    # discusiones de GitHub de openWakeWord (dscripka/openWakeWord#110,
    # discussion #62) encontramos que el propio mantenedor indica que
    # "usually between 20,000 and 50,000 [positive examples] is sufficient"
    # para que auto_train aprenda algo — y un caso reportado ahí con pocos
    # positivos describe un modelo de "206 KB" que "no funciona en absoluto",
    # prácticamente idéntico a nuestro síntoma original (207 KB, sin
    # detección). Nuestro dataset real (166 positivos de entrenamiento) está
    # 100-300x por debajo de esa escala — ningún ajuste de layer_size, steps,
    # reparto de batch o pesos lo compensa si la red literalmente nunca ve
    # suficiente variedad de positivos. No podemos generar 20-50k grabaciones
    # reales, pero sí multiplicar mucho más la augmentación synthetic
    # (pitch/EQ/ruido/RIR) por clip como paliativo barato e inmediato — de
    # 8 a 40 rondas (166*40 ≈ 6.640 instancias antes de ventaneo, todavía por
    # debajo del ideal pero un salto real desde 1.328). Si esto no basta,
    # el siguiente paso real es generar más variantes sintéticas de la
    # frase (más voces/velocidades), no seguir subiendo esto indefinidamente
    # (la augmentación sobre las mismas ~195 grabaciones base tiene
    # rendimientos decrecientes).
    augmentation_rounds: int = 40
    layer_size: int = 128
    model_type: str = "dnn"
    clip_seconds: float = 2.0
    window_steps: int = 16
    negative_phrase_count: int = 120


def collect_positive_clips(positivos_path: Path) -> list[Path]:
    """Devuelve todos los WAVs bajo positivos_path (recursivo)."""
    if not positivos_path.exists():
        return []
    return sorted(positivos_path.rglob("*.wav"))


def _split_train_val(clips: list[Path], val_fraction: float, seed: int = 42) -> tuple[list[Path], list[Path]]:
    rng = random.Random(seed)
    shuffled = clips.copy()
    rng.shuffle(shuffled)
    if len(shuffled) <= 4:
        return shuffled, []
    n_val = max(1, int(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]


def _reshape_windows(x: np.ndarray, n: int = 16) -> np.ndarray:
    """
    Reordena features de cualquier longitud temporal en ventanas fijas de
    `n` pasos (1.28s por defecto). Replica la transformación que
    openwakeword/train.py aplica en su bloque __main__ oficial, necesaria
    para que los positivos (calculados con la duración de clip que sea)
    sean compatibles con los negativos precalculados de HuggingFace, que
    siempre vienen en ventanas fijas de 16 pasos.
    """
    if x.shape[1] == n:
        return x
    flat = np.vstack(x)
    rows = [flat[i:i + n, :] for i in range(0, flat.shape[0] - n, n)]
    if not rows:
        raise ValueError(
            f"No hay suficientes datos para formar ni una ventana de {n} pasos "
            f"(disponibles: {flat.shape[0]}). Añade más clips."
        )
    return np.array(rows)


MIN_NEGATIVE_CLIPS = 5

DEFAULT_TRAIN_BATCH_SIZE = 128


def _compute_n_per_class(feature_data_files: dict, batch_size: int = DEFAULT_TRAIN_BATCH_SIZE) -> dict:
    """
    Reparto fijo e igualitario de muestras por clase en cada batch de
    entrenamiento (ver el comentario en `run_training` junto a donde se usa
    para la causa raíz completa). Deliberadamente NO depende de cuántas filas
    tenga cada fichero de features en disco — ese es justo el bug que esto
    corrige: dejar que `mmap_batch_generator` calcule el reparto solo a
    partir del tamaño en disco da a "positive" una representación
    insignificante en cuanto el dataset de negativos es mucho más grande
    (siempre, con negative_mode='full'), y el modelo nunca aprende a
    reconocer la wake word por más pasos o capacidad que se le den.
    """
    n_keys = max(1, len(feature_data_files))
    return {key: max(1, batch_size // n_keys) for key in feature_data_files}


def _validate_feature_file(path: Path, label: str) -> None:
    """
    openwakeword.data.mmap_batch_generator asume que todos los ficheros de
    features son arrays 3D (N, pasos, 96) con N>0, y si no lo son falla con
    un IndexError genérico ("tuple index out of range") sin decir en cuál.
    Lo comprobamos aquí antes para señalar exactamente qué fichero está mal
    y por qué (típicamente: 0 filas porque la síntesis de negativos falló).
    """
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 3 or arr.shape[0] == 0:
        raise ValueError(
            f"El fichero de features '{label}' ({path}) tiene shape {arr.shape}, "
            "se esperaba (N, pasos, 96) con N>0. Probablemente la síntesis de "
            "ese conjunto de clips falló o no produjo suficientes datos "
            "(revisa el log anterior — p.ej. límite de API en el provider TTS)."
        )


def _extract_features(
    clips: list[Path],
    total_length: int,
    work_dir: Path,
    name: str,
    augmentation_rounds: int = 1,
    batch_size: int = 64,
    background_clip_paths: Optional[list[str]] = None,
    rir_paths: Optional[list[str]] = None,
) -> Path:
    """
    Aplica augmentación y calcula features openWakeWord para una lista de
    clips. Si se pasan `background_clip_paths`/`rir_paths` (ver
    trainer/noise_data.py), `augment_clips` mezcla ruido de fondo real y
    convoluciona con reverberación real además de la augmentación sintética
    de siempre (EQ, distorsión, pitch-shift, ruido coloreado) — si se dejan
    vacíos, el comportamiento es idéntico al de antes de este cambio.
    """
    paths = [str(c) for c in clips] * max(1, augmentation_rounds)
    if not paths:
        raise ValueError(f"No hay clips para calcular features ({name}).")

    # openwakeword.data.trim_mmap (llamado internamente por
    # compute_features_from_generator) calcula el nombre del fichero temporal
    # con `mmap_path.strip(".npy")` — un uso erróneo de str.strip(), que no
    # quita un sufijo sino que recorta cualquier carácter de los extremos que
    # esté en el conjunto {'.', 'n', 'p', 'y'}. Con una ruta relativa que
    # empiece por "p" (como "projects/...") o un nombre que termine en "n"
    # (como "..._train.npy") esto corrompe la ruta silenciosamente
    # ("projects" -> "rojects", "train.npy" -> "trai2.npy") y revienta con
    # FileNotFoundError. Usamos ruta absoluta (evita el recorte por delante)
    # y forzamos un sufijo que nunca cae en ese conjunto de caracteres (evita
    # el recorte por detrás), sin depender de adivinar qué nombres son "seguros".
    out_path = (work_dir / f"{name}_data.npy").resolve()
    gen = augment_clips(
        paths, total_length=total_length, batch_size=min(batch_size, len(paths)),
        background_clip_paths=background_clip_paths or [],
        RIR_paths=rir_paths or [],
    )
    compute_features_from_generator(
        gen, n_total=len(paths), clip_duration=total_length,
        output_file=str(out_path), device="cpu", ncpu=1,
    )
    return out_path


NEGATIVE_CATALOG_FILE = ".negative_catalog.json"


def _load_negative_catalog(out_dir: Path) -> dict[str, str]:
    import json
    p = out_dir / NEGATIVE_CATALOG_FILE
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _save_negative_catalog(out_dir: Path, catalog: dict[str, str]) -> None:
    import json
    (out_dir / NEGATIVE_CATALOG_FILE).write_text(json.dumps(catalog, indent=2, ensure_ascii=False))


def _synthesize_negative_clips(project: "Project", out_dir: Path, n_phrases: int = 40) -> list[Path]:
    """
    Sintetiza clips negativos "duros" (variaciones fonéticas cercanas a la
    wake word + frases genéricas) usando las mismas fuentes TTS ya
    configuradas para el proyecto. Si no hay ninguna fuente configurada,
    devuelve solo los clips ya existentes (se seguirá entrenando solo con
    los negativos generales descargados).

    Usa un catálogo (frase|fuente|voz -> fichero, igual que el de síntesis
    de positivos en trainer/synthesizer.py) en vez de comparar `n_phrases`
    contra el recuento total de la carpeta: esa carpeta también contiene
    negativos importados a mano (grabaciones propias, descargas...) que no
    tienen nada que ver con las frases sintetizadas aquí, así que un simple
    "ya hay bastantes ficheros" bloqueaba la síntesis aunque las frases en
    sí nunca se hubieran generado. Con catálogo, cada frase se sintetiza
    como mucho una vez pase lo que pase con el resto de negativos de la
    carpeta, y ampliar `n_phrases` (o la lista de frases en
    negative_phrases.py) siempre añade contenido nuevo en vez de no hacer
    nada.
    """
    from trainer.negative_phrases import build_negative_phrases
    from trainer.synthesizer import (
        synthesize_piper, synthesize_openai, synthesize_elevenlabs, synthesize_google, _resolve_token,
    )
    import asyncio

    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("*.wav"))
    clips: list[Path] = list(existing)

    sources = [s for s in project.synthesis.sources if s.selected_voices]
    if not sources:
        return clips

    catalog = _load_negative_catalog(out_dir)
    phrases = build_negative_phrases(project.wake_word, n=n_phrases)
    clip_idx = len(existing) + 1

    # Un token por fuente resuelto una sola vez (no uno por frase) — mismo
    # motivo que en synthesizer.run_synthesis: evita rehacer la resolución
    # (y un posible prompt interactivo) en cada iteración.
    tokens = {id(s): _resolve_token(s) for s in sources if s.type in ("openai", "google")}

    # Barra de progreso: sin esto, 100+ llamadas de red/subproceso en serie
    # no imprimen nada hasta el final y dan la impresión de estar colgado
    # (el mismo problema de percepción que ya tuvimos con run_synthesis).
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                  TextColumn("{task.completed}/{task.total}")) as progress:
        task = progress.add_task("[cyan]negativos sintéticos[/cyan]", total=len(phrases))

        for i, phrase in enumerate(phrases):
            source = sources[i % len(sources)]
            voice = source.selected_voices[i % len(source.selected_voices)]
            key = f"{phrase}|{source.url or source.type}|{voice}"
            if key in catalog and (out_dir / catalog[key]).exists():
                progress.advance(task)
                continue

            out_path = out_dir / f"neg_{clip_idx:03d}.wav"
            try:
                if source.type == "piper":
                    synthesize_piper(phrase, voice, out_path, speed=1.0, piper_binary=source.binary)
                elif source.type == "google":
                    token = tokens.get(id(source), "")
                    asyncio.run(synthesize_google(phrase, voice, token, out_path, 1.0))
                else:
                    token = tokens.get(id(source), "")
                    if source.token_header == "xi-api-key":
                        asyncio.run(synthesize_elevenlabs(phrase, voice, source.url, token, out_path, 1.0))
                    else:
                        asyncio.run(synthesize_openai(phrase, voice, source.url, token, out_path, 1.0,
                                                       token_header=source.token_header))
                catalog[key] = out_path.name
                _save_negative_catalog(out_dir, catalog)
                clips.append(out_path)
                clip_idx += 1
            except Exception as exc:
                progress.console.print(
                    f"  [yellow]No se pudo sintetizar negativo '{escape(phrase)}': {escape(str(exc))}[/yellow]"
                )
            finally:
                progress.advance(task)

    return clips


def _prepare_noise_augmentation(full: bool) -> tuple[list[str], list[str]]:
    """
    Descarga (si hace falta) RIR + ruido de fondo reales para augmentación
    (ver trainer/noise_data.py) y devuelve (rir_paths, background_paths).

    RIR y ruido de fondo se intentan por separado (no en un único try/except
    conjunto): si uno de los dos falla (sin dependencia 'datasets', sin red,
    HuggingFace caído, un dataset con problemas de streaming...) el otro no
    se pierde. Si algo falla del todo, se avisa y esa parte se queda en
    lista vacía — el entrenamiento sigue igual que antes de que existiera
    esta función, solo sin ese beneficio concreto.
    """
    rir_paths: list[str] = []
    background_paths: list[str] = []

    try:
        from trainer.noise_data import ensure_rir_clips
        rir_paths = ensure_rir_clips()
    except Exception as exc:
        console.print(
            f"  [yellow]No se pudo preparar RIR real ({escape(str(exc))}) — sin reverberación real.[/yellow]"
        )

    try:
        from trainer.noise_data import ensure_background_noise_clips
        background_paths = ensure_background_noise_clips(full=full)
    except Exception as exc:
        console.print(
            f"  [yellow]No se pudo preparar ruido de fondo real ({escape(str(exc))}) "
            "— sin ruido de fondo real.[/yellow]"
        )

    return rir_paths, background_paths


def _configure_training_logging() -> None:
    """
    openwakeword.train.Model.train_model/auto_train reportan sus métricas
    reales de entrenamiento (recall/accuracy de validación por cada
    checkpoint, y sobre todo la línea final "Final Model Accuracy: X") por
    `logging.info(...)`, no por print(). El logging raíz de Python viene a
    nivel WARNING por defecto y este repo nunca lo configuraba — así que
    esas líneas se descartaban en silencio y jamás llegaban a la terminal,
    con independencia de lo bien o mal que fuera el entrenamiento por
    dentro. Resultado práctico: no había forma de saber, mirando el log de
    un entrenamiento, si el modelo había aprendido algo de verdad o no — el
    único dato disponible era el resultado final de 'evaluate', que no
    distingue "el modelo nunca aprendió" de "el modelo aprendió pero algo en
    la evaluación/inferencia no encaja" (formato de audio, sample rate,
    threshold...). Con esto activado, un entrenamiento con "Final Model
    Accuracy" alta seguido de una evaluación en 0% apunta claramente a un
    bug de evaluación/inferencia; una "Final Model Accuracy" ya baja de por
    sí apunta a un bug de datos/entrenamiento.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger().setLevel(logging.INFO)


def run_training(project: "Project", config: TrainingConfig) -> Path:
    """
    Lanza el entrenamiento completo con openWakeWord y devuelve la ruta al
    modelo resultante (.tflite si la conversión funcionó, si no .onnx).
    """
    _configure_training_logging()

    # Comprobamos primero lo más barato de verificar (¿hay algo que entrenar?)
    # antes de exigir las dependencias pesadas de ML o descargar nada.
    positive_clips = collect_positive_clips(project.positivos_path) + collect_positive_clips(project.sintetizados_path)
    if not positive_clips:
        raise ValueError(
            f"No se encontraron clips positivos en {project.positivos_path} ni {project.sintetizados_path}"
        )

    # El propio mantenedor de openWakeWord indica que auto_train necesita
    # típicamente 20.000-50.000 clips positivos para aprender de forma
    # fiable (github.com/dscripka/openWakeWord discussions/62 e issues/110
    # — este último describe un modelo de 206 KB que "no funciona en
    # absoluto" con pocos positivos, el mismo síntoma exacto que tuvimos
    # aquí). Avisamos siempre que estemos muy por debajo, no solo cuando ya
    # ha fallado — la augmentación (ver TrainingConfig.augmentation_rounds)
    # ayuda pero no sustituye tener grabaciones/síntesis base variadas.
    if len(positive_clips) < 2000:
        console.print(
            f"  [yellow]⚠️  Solo {len(positive_clips)} clips positivos base (antes de augmentar). "
            "openWakeWord suele necesitar 20.000-50.000 para aprender de forma fiable — con muchos "
            "menos, el modelo puede acabar con recall 0 pase lo que pase con pasos/capacidad/pesos. "
            "Considera generar más variantes sintéticas (más voces/velocidades) si el recall sigue "
            "saliendo muy bajo.[/yellow]"
        )

    if not HAS_TRAIN_DEPS:
        raise RuntimeError(f"{INSTALL_HINT}\n(error original: {_IMPORT_ERROR})")

    work_dir = config.output_dir / "_work" / config.model_name
    work_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    console.print("  [dim]Comprobando modelos base de openWakeWord...[/dim]")
    oww_utils.download_models()

    train_clips, val_clips = _split_train_val(positive_clips, config.val_fraction)
    if not val_clips:
        cut = max(1, len(train_clips) // 5)
        val_clips, train_clips = train_clips[:cut], train_clips[cut:]

    # Normaliza a 16kHz mono en el sitio antes de calcular features. Necesario
    # porque las voces Piper no comparten sample rate nativo (algunas están a
    # 16000 Hz, otras a 22050 Hz) y openwakeword.data.augment_clips exige
    # 16000 Hz exacto o falla con "Clip does not have the correct sample
    # rate!" sin decir cuál. Es un no-op barato (solo lee la cabecera) para
    # los clips que ya estén bien, así que es seguro llamarlo siempre.
    train_clips, n_repaired_train = repair_clips(train_clips)
    val_clips, n_repaired_val = repair_clips(val_clips)
    n_repaired = n_repaired_train + n_repaired_val
    if n_repaired:
        console.print(
            f"  [dim]{n_repaired} clip(s) positivos no estaban en 16kHz mono "
            "(voces Piper con distinto sample rate nativo) — corregidos en el sitio.[/dim]"
        )

    console.print(f"  [dim]Clips positivos:[/dim] {len(train_clips)} entrenamiento / {len(val_clips)} validación")

    console.print("  [dim]Preparando ruido de fondo y RIR reales para la augmentación...[/dim]")
    rir_paths, background_paths = _prepare_noise_augmentation(full=(config.negative_mode == "full"))
    if rir_paths or background_paths:
        console.print(
            f"  [dim]  {len(rir_paths)} RIR + {len(background_paths)} clips de ruido de fondo real disponibles.[/dim]"
        )

    total_length = int(config.clip_seconds * 16000)

    pos_train_feat = _extract_features(
        train_clips, total_length, work_dir, "positive_features_train",
        augmentation_rounds=config.augmentation_rounds,
        background_clip_paths=background_paths, rir_paths=rir_paths,
    )
    pos_val_feat = _extract_features(
        val_clips, total_length, work_dir, "positive_features_val",
        augmentation_rounds=max(1, config.augmentation_rounds // 4),
        background_clip_paths=background_paths, rir_paths=rir_paths,
    )

    console.print("  [dim]Generando negativos sintéticos (variaciones cercanas + frases genéricas)...[/dim]")
    negative_clips = _synthesize_negative_clips(project, project.negativos_path, n_phrases=config.negative_phrase_count)
    negative_clips, n_repaired_neg = repair_clips(negative_clips)
    if n_repaired_neg:
        console.print(
            f"  [dim]{n_repaired_neg} clip(s) negativos no estaban en 16kHz mono — corregidos en el sitio.[/dim]"
        )

    adv_neg_feat: Optional[Path] = None
    if len(negative_clips) >= MIN_NEGATIVE_CLIPS:
        adv_neg_feat = _extract_features(
            negative_clips, total_length, work_dir, "adversarial_negative_features",
            augmentation_rounds=config.augmentation_rounds,
            background_clip_paths=background_paths, rir_paths=rir_paths,
        )
        _validate_feature_file(adv_neg_feat, "adversarial_negative")
    elif negative_clips:
        console.print(
            f"  [yellow]Solo se sintetizaron {len(negative_clips)} negativos "
            f"(mínimo {MIN_NEGATIVE_CLIPS}) — probablemente por un límite de API. "
            "Se entrena solo con negativos generales.[/yellow]"
        )
    else:
        console.print(
            "  [yellow]Sin negativos sintéticos disponibles (¿sin providers TTS "
            "configurados, o todos fallaron?). Se entrena solo con negativos generales.[/yellow]"
        )

    console.print(f"  [dim]Preparando negativos generales (modo '{config.negative_mode}')...[/dim]")
    neg = ensure_negative_features(work_dir, mode=config.negative_mode, window_steps=config.window_steps)

    _validate_feature_file(pos_train_feat, "positive (train)")
    _validate_feature_file(pos_val_feat, "positive (val)")
    _validate_feature_file(neg["train"], "general_negative (train)")
    _validate_feature_file(neg["val"], "general_negative (val)")
    _validate_feature_file(neg["false_positive_val"], "false_positive_val")

    feature_data_files = {
        "positive": str(pos_train_feat),
        "general_negative": str(neg["train"]),
    }
    if adv_neg_feat is not None:
        feature_data_files["adversarial_negative"] = str(adv_neg_feat)

    def _make_label_fn(is_positive: bool):
        return (lambda x: [1] * len(x)) if is_positive else (lambda x: [0] * len(x))

    label_transforms = {key: _make_label_fn(key == "positive") for key in feature_data_files}
    window_steps = config.window_steps
    data_transforms = {key: (lambda x, n=window_steps: _reshape_windows(x, n)) for key in feature_data_files}

    # CRÍTICO: sin `n_per_class` explícito, `mmap_batch_generator` calcula
    # cuántas muestras de cada clase entran en cada batch de forma
    # PROPORCIONAL al número de filas que tiene cada fichero de features en
    # disco (ver openwakeword/data.py::mmap_batch_generator.__init__,
    # `ratio = shapes[lbl][0] / sum(...)`) — no a ningún criterio de balance
    # razonable. Con negative_mode='full' (ACAV100M, ~2000h) el fichero de
    # "general_negative" tiene órdenes de magnitud más filas que "positive"
    # (166 clips reales, unos pocos miles de ventanas tras augmentar), así
    # que ese ratio da prácticamente 0 y `max(1, ...)` deja a "positive" con
    # literalmente 1 muestra por batch de 128 — el resto, negativos — durante
    # los 50.000+ pasos enteros. Combinado con el peso extra sobre negativos
    # (`negative_weight_schedule`, hasta max_negative_weight=1000x), la señal
    # positiva queda completamente ahogada: esto es la causa confirmada de
    # que "Final Model Recall" saliera en 0.0 de principio a fin del
    # entrenamiento (no un problema de threshold, ni de TFLite/ONNX, ni de
    # falta de pasos/capacidad — eso ya se corrigió aparte). Fijamos aquí un
    # reparto fijo e igualitario entre clases, independiente del tamaño en
    # disco de cada dataset, para que "positive" tenga representación real
    # en cada batch pase lo que pase con el tamaño del set de negativos.
    n_per_class = _compute_n_per_class(feature_data_files)

    batch_gen = mmap_batch_generator(
        feature_data_files,
        data_transform_funcs=data_transforms,
        label_transform_funcs=label_transforms,
        n_per_class=n_per_class,
    )

    class _IterDataset(torch.utils.data.IterableDataset):
        def __init__(self, generator):
            self.generator = generator

        def __iter__(self):
            return self.generator

    X_train = torch.utils.data.DataLoader(_IterDataset(batch_gen), batch_size=None)

    X_val_pos = _reshape_windows(np.load(pos_val_feat), window_steps)
    X_val_neg = _reshape_windows(np.load(neg["val"]), window_steps)
    val_labels = np.hstack((np.ones(X_val_pos.shape[0]), np.zeros(X_val_neg.shape[0]))).astype(np.float32)
    X_val = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(np.vstack((X_val_pos, X_val_neg)).astype(np.float32)),
            torch.from_numpy(val_labels),
        ),
        batch_size=len(val_labels),
    )

    fp_data = _reshape_windows(np.array(np.load(neg["false_positive_val"], mmap_mode="r")), window_steps)
    fp_labels = np.zeros(fp_data.shape[0]).astype(np.float32)
    X_fp = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(fp_data.astype(np.float32)), torch.from_numpy(fp_labels)),
        batch_size=len(fp_labels),
    )

    input_shape = (window_steps, X_val_pos.shape[-1])
    oww = TrainModel(
        n_classes=1,
        input_shape=input_shape,
        model_type=config.model_type,
        layer_dim=config.layer_size,
        seconds_per_example=1280 * input_shape[0] / 16000,
    )

    console.print(f"  [dim]Entrenando ({config.steps} pasos, modelo '{config.model_type}')...[/dim]")
    best_model = oww.auto_train(
        X_train=X_train,
        X_val=X_val,
        false_positive_val_data=X_fp,
        steps=config.steps,
        max_negative_weight=config.max_negative_weight,
        target_fp_per_hour=config.target_fp_per_hour,
    )

    # openwakeword.train.Model.auto_train calcula un "Final Model Accuracy/
    # Recall/FP per hour" para el modelo combinado final, pero SOLO lo
    # imprime por logging.info (visible ahora gracias a
    # _configure_training_logging, arriba) — no lo devuelve ni lo deja en
    # ningún atributo. Lo que sí queda como atributo del propio objeto `oww`
    # es el MEJOR checkpoint visto durante el entrenamiento
    # (best_val_accuracy/best_val_recall), que imprimimos aquí de forma
    # explícita (no depende de que el usuario mire/entienda el log crudo de
    # la librería). Si esto ya sale con recall ~0 aquí, el problema es del
    # entrenamiento en sí (datos/pesos/negativos), no de la conversión a
    # TFLite ni de 'evaluate' — antes de este cambio no había forma de
    # distinguir ambos casos sin adivinar.
    #
    # OJO: `oww.best_val_fp` NO se imprime aquí a propósito — revisando el
    # código fuente de openwakeword.train.Model.train_model confirmamos que
    # ese atributo se inicializa a 1000 en el constructor y el propio código
    # de la librería nunca lo reasigna en ningún sitio (solo actualiza
    # best_val_recall/best_val_accuracy al guardar un "mejor" checkpoint).
    # Es un atributo muerto en esta versión de la librería — mostrarlo daba
    # la falsa impresión de ser una medida real ("FP val: 1000.000" en todos
    # los entrenamientos, aprendan algo o no), cuando en realidad nunca se
    # mueve de su valor por defecto.
    console.print(
        f"  [dim]Mejor checkpoint durante el entrenamiento — "
        f"accuracy val: {oww.best_val_accuracy:.3f}, "
        f"recall val: {oww.best_val_recall:.3f}[/dim]"
    )
    if oww.best_val_recall < 0.5:
        console.print(
            "  [yellow]⚠️  El recall de validación durante el entrenamiento ya es muy bajo — "
            "el modelo no está aprendiendo a reconocer la wake word (no es un problema de "
            "threshold ni de conversión a TFLite/ONNX). Revisa la cantidad/calidad de los "
            "clips positivos antes de reentrenar.[/yellow]"
        )

    oww.export_model(model=best_model, model_name=config.model_name, output_dir=str(config.output_dir))
    onnx_path = config.output_dir / f"{config.model_name}.onnx"
    tflite_path = config.output_dir / f"{config.model_name}.tflite"

    console.print("  [dim]Convirtiendo a TFLite en un proceso aparte (evita el bloqueo con torch)...[/dim]")
    try:
        convert_onnx_to_tflite_in_subprocess(onnx_path, tflite_path)
        return tflite_path
    except TFLiteExportError as exc:
        console.print(f"  [yellow]{escape(str(exc))}[/yellow]")
        return onnx_path
