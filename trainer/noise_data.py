# trainer/noise_data.py
"""
Ruido de fondo y respuestas de impulso de sala (RIR) reales, para que la
augmentación de `openwakeword.data.augment_clips` (invocada desde
trainer/trainer_core.py) sea realista en vez de puramente sintética.

Por qué existe este módulo: `augment_clips` aplica siempre EQ, distorsión,
pitch-shift y un ruido "coloreado" generado matemáticamente — pero solo
mezcla ruido de fondo real y solo convoluciona con reverberación real si se
le pasan `background_clip_paths`/`RIR_paths` explícitos, y trainer_core.py
no se los pasaba (los clips positivos y los negativos "duros" solo se
entrenaban limpios + augmentación sintética). Sin esto, el modelo nunca ve
un ejemplo con eco de habitación real ni con ruido de fondo grabado de
verdad durante el entrenamiento, así que su comportamiento en condiciones
reales (cocina con radio, salón con eco, micro lejos) puede ser peor de lo
que sugieren las métricas de validación limpias.

Sigue el mismo patrón que usa el notebook oficial de entrenamiento de
openWakeWord (`automatic_model_training.ipynb`):

  - RIR: `davidscripka/MIT_environmental_impulse_responses` (HuggingFace) —
    270 respuestas de impulso reales recopiladas por el MIT, dataset
    pequeño (pocos MB), siempre se descarga entera.
  - Ruido de fondo, modo "quick" (por defecto): un fragmento pequeño
    (~1h por defecto) de música libre de `rudraml/fma` en streaming, sin
    guardar el dataset completo — pensado para no añadir una descarga
    pesada al camino por defecto.
  - Ruido de fondo, modo "full": añade además un fragmento del split
    "balanced"/"train" de AudioSet (`agkphysics/AudioSet`, vía streaming +
    `.take(n)` — el repo se reorganizó a formato parquet y el split
    completo son ~26 GB, así que NO se descarga entero) con sonidos
    ambientales y de voz reales — mucho más diverso que solo música.
    Reutiliza el mismo modo "full"/"quick" que ya existía para los
    negativos generales (ver trainer/negative_data.py) en vez de añadir
    otro flag más.

Todo se cachea en disco bajo `data/rir/` y `data/background_noise/`
(compartido entre proyectos, igual que `data/negative_features/`) — no se
vuelve a descargar en el siguiente entrenamiento.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

from trainer.audio_utils import write_wav_16k_mono

RIR_ROOT = Path("data/rir")
NOISE_ROOT = Path("data/background_noise")


def ensure_rir_clips(dest_dir: Path = RIR_ROOT) -> list[str]:
    """
    Descarga (si hace falta) las respuestas de impulso de sala del MIT y
    devuelve las rutas locales en 16kHz mono. No-op si ya están en disco.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(dest_dir.glob("*.wav"))
    if existing:
        return [str(p) for p in existing]

    import datasets

    rir_dataset = datasets.load_dataset(
        "davidscripka/MIT_environmental_impulse_responses", split="train", streaming=True
    )
    paths: list[str] = []
    for row in rir_dataset:
        audio = row["audio"]
        name = Path(audio["path"]).with_suffix(".wav").name
        out_path = dest_dir / name
        write_wav_16k_mono(np.asarray(audio["array"]), audio["sampling_rate"], out_path)
        paths.append(str(out_path))
    return paths


def ensure_fma_noise_clips(dest_dir: Path = NOISE_ROOT, hours: float = 1.0) -> list[str]:
    """
    Descarga (si hace falta) un fragmento de música libre de FMA como ruido
    de fondo. `hours` controla cuánto: los clips de FMA son de ~30s, así
    que se piden `hours*3600/30` de ellos en streaming (sin bajar el
    dataset completo).
    """
    fma_dir = dest_dir / "fma"
    fma_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(fma_dir.glob("*.wav"))
    if existing:
        return [str(p) for p in existing]

    import datasets

    # OJO: sin `.cast_column("audio", datasets.Audio(sampling_rate=16000))`
    # a propósito. Pedirle a `datasets` que resamplee durante el decode de
    # un dataset en streaming dispara en algunos entornos
    # "Cannot seek streaming HTTP file" (el decoder necesita acceso
    # aleatorio sobre una respuesta HTTP en streaming, que no lo soporta).
    # `ensure_rir_clips` NO pide resampleo al decodificar y por eso no le
    # pasa esto — aquí seguimos el mismo patrón: se decodifica al sample
    # rate nativo del clip y se resamplea después, ya en local, con
    # `write_wav_16k_mono` (igual que hace RIR).
    fma_dataset = datasets.load_dataset("rudraml/fma", name="small", split="train", streaming=True)

    n_clips = max(1, int(hours * 3600 // 30))
    paths: list[str] = []
    it = iter(fma_dataset)
    for _ in range(n_clips):
        try:
            row = next(it)
        except StopIteration:
            break
        except Exception:
            # Un clip individual que falle (red, decode puntual...) no debe
            # tirar todo lo ya descargado — nos quedamos con lo que haya.
            break
        audio = row["audio"]
        name = Path(audio["path"]).with_suffix(".wav").name
        out_path = fma_dir / name
        write_wav_16k_mono(np.asarray(audio["array"]), audio["sampling_rate"], out_path)
        paths.append(str(out_path))
    return paths


def ensure_audioset_noise_clips(dest_dir: Path = NOISE_ROOT, hours: float = 2.0) -> list[str]:
    """
    Modo full: fragmento del split "balanced"/"train" de AudioSet (sonidos
    ambientales y de voz reales) vía streaming + `.take(n)`, sin descargar
    el split completo (~26 GB repartidos en 38 ficheros parquet).

    OJO histórico: la primera versión de esta función descargaba un
    `.tar` fijo por URL (`bal_train09.tar`), copiado del notebook oficial
    de openWakeWord de 2023. Ese repo de HuggingFace se reorganizó a
    formato parquet (confirmado en huggingface.co/datasets/agkphysics/
    AudioSet, "Convert to Parquet format") y esa ruta ya no existe —
    cualquier URL de `.tar` a día de hoy da 404. Se usa `datasets` con
    streaming en su lugar, igual que RIR y FMA, evitando además la
    augmentación con "seek" que ya nos mordió en FMA (ver
    `ensure_fma_noise_clips`): sin `.cast_column` forzando resampleo,
    decode a sample rate nativo y resampleo después en local.
    """
    audioset_dir = dest_dir / "audioset"
    audioset_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(audioset_dir.glob("*.wav"))
    if existing:
        return [str(p) for p in existing]

    import datasets

    audioset_dataset = datasets.load_dataset(
        "agkphysics/AudioSet", "balanced", split="train", streaming=True
    )

    n_clips = max(1, int(hours * 3600 // 10))  # clips de AudioSet son de ~10s
    paths: list[str] = []
    it = iter(audioset_dataset)
    for _ in range(n_clips):
        try:
            row = next(it)
        except StopIteration:
            break
        except Exception:
            break
        audio = row["audio"]
        name = Path(audio["path"]).with_suffix(".wav").name
        out_path = audioset_dir / name
        write_wav_16k_mono(np.asarray(audio["array"]), audio["sampling_rate"], out_path)
        paths.append(str(out_path))
    return paths


def ensure_background_noise_clips(
    dest_dir: Path = NOISE_ROOT, full: bool = False, quick_hours: float = 0.5
) -> list[str]:
    """
    Punto de entrada único de ruido de fondo real.

    AudioSet es la fuente principal, en modo quick (fragmento pequeño,
    `quick_hours`) y full (fragmento mayor, 2h) — es un dataset parquet
    estándar de HuggingFace, sin script de carga "custom code", así que es
    la fuente fiable de las dos.

    FMA se intenta ADEMÁS como extra, pero de forma no bloqueante: en la
    práctica su repo de HuggingFace exige ejecutar "custom code" al cargarlo
    y ha fallado con "Cannot seek streaming HTTP file" incluso decodificando
    al sample rate nativo (descartado como causa: no es el resampleo, es
    algo del propio script de carga). Si vuelve a fallar, se descarta esa
    fuente sin más — el bug real que se dio en producción era que este
    fallo también se llevaba por delante lo que ya hubiera funcionado de
    AudioSet, en vez de quedarse solo sin el extra de FMA.
    """
    hours = 2.0 if full else quick_hours
    paths = list(ensure_audioset_noise_clips(dest_dir, hours=hours))

    try:
        paths += ensure_fma_noise_clips(dest_dir, hours=quick_hours)
    except Exception:
        pass

    return paths
