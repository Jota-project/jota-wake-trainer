# trainer/tflite_export.py
"""
Conversión ONNX -> TFLite.

openWakeWord trae su propia `convert_onnx_to_tflite` (openwakeword.train),
pero depende de onnx==1.14.0 + onnx-tf==1.10.0 + tensorflow-cpu==2.8.1, un
combo abandonado desde 2024 que no instala en macOS Apple Silicon ni en
Python 3.11+ (no hay wheels). Es la causa de fondo de los "ModuleNotFoundError:
onnx_tf" que se ven por todo el rastro de issues de openWakeWord.

En su lugar usamos onnx2tf (mantenido activamente, sin la dependencia de
onnx-tf/tensorflow-probability), que es el reemplazo que la comunidad viene
usando desde 2025 para este mismo problema.

El .tflite es el formato que realmente carga el addon openWakeWord de
Wyoming/Home Assistant en CPU (el .onnx solo se usa si el addon corre con
CUDA). Por eso, aunque el entrenamiento en sí no necesita tensorflow para
nada, esta conversión final sigue siendo necesaria para desplegar en HA.

Por qué existe también `convert_onnx_to_tflite_in_subprocess`: en producción
(macOS) se observó que llamar a `convert_onnx_to_tflite` justo después de
entrenar, EN EL MISMO proceso que ya había cargado y usado `torch` durante
miles de pasos, se queda completamente colgado nada más imprimir el resumen
de `tf.saved_model.save` (los bloques "Saved artifact... Output Type...
Captures...") — sin avanzar, sin errores, e ignorando Ctrl+C por completo
(el propio intérprete de Python no llega a comprobar la señal pendiente
porque el bloqueo ocurre dentro de código nativo de TensorFlow, no en
bytecode Python). Confirmado con openwakeword.train.Model: el entrenamiento
usa `torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')`, así
que en Mac (sin CUDA) siempre es CPU — descarta que sea una pelea por la
GPU/Metal entre torch y tensorflow. Lo que sí coincide con el patrón conocido
de "torch y tensorflow cargados en el mismo proceso" es un conflicto entre
sus runtimes nativos de hilos (cada uno trae su propio OpenMP/BLAS
empaquetado) al inicializarse uno justo después de que el otro ya haya
creado y usado sus propios pools de hilos — de ahí que, confirmado también en
producción, ejecutar la conversión en un proceso nuevo (sin torch cargado
nunca) la resuelve en un segundo. La solución de raíz no es "esperar más" ni
"reintentar en el mismo proceso": es no compartir proceso con torch nunca.
"""
from __future__ import annotations
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

INSTALL_HINT = (
    "Faltan dependencias para exportar a .tflite. Instálalas con:\n"
    "  pip install -e '.[tflite]'\n"
    "El modelo .onnx ya generado sigue siendo válido; solo no se puede "
    "convertir a .tflite todavía. Si tu Home Assistant/Wyoming corre con "
    "GPU (CUDA) puedes usar directamente el .onnx."
)


class TFLiteExportError(RuntimeError):
    pass


def convert_onnx_to_tflite(onnx_path: Path, tflite_path: Path) -> Path:
    try:
        import onnx2tf  # noqa: F401
        import tensorflow  # noqa: F401
    except ImportError as exc:
        raise TFLiteExportError(INSTALL_HINT) from exc

    onnx_path = Path(onnx_path)
    tflite_path = Path(tflite_path)
    if not onnx_path.exists():
        raise TFLiteExportError(f"No existe el modelo ONNX de origen: {onnx_path}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=tmp_dir,
            non_verbose=True,
            output_signaturedefs=True,
        )

        candidates = sorted(Path(tmp_dir).glob("*float32.tflite")) or sorted(Path(tmp_dir).glob("*.tflite"))
        if not candidates:
            raise TFLiteExportError(
                "onnx2tf no generó ningún .tflite. Revisa la salida anterior para más detalle."
            )

        tflite_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(candidates[0], tflite_path)

    return tflite_path


def convert_onnx_to_tflite_in_subprocess(
    onnx_path: Path, tflite_path: Path, timeout: float = 600.0
) -> Path:
    """
    Igual que `convert_onnx_to_tflite`, pero ejecutado en un proceso `python`
    nuevo en vez de en el proceso actual. Ver el docstring del módulo: hacerlo
    en el mismo proceso que ya usó `torch` para entrenar se cuelga de forma
    irrecuperable (ni siquiera Ctrl+C funciona) en macOS. Llamar a esto desde
    el flujo de entrenamiento (trainer_core.py); `wake-trainer convert` sigue
    usando la función in-process porque ahí no hay torch cargado antes.
    """
    onnx_path = Path(onnx_path)
    tflite_path = Path(tflite_path)
    if not onnx_path.exists():
        raise TFLiteExportError(f"No existe el modelo ONNX de origen: {onnx_path}")

    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from trainer.tflite_export import convert_onnx_to_tflite, TFLiteExportError\n"
        "try:\n"
        f"    convert_onnx_to_tflite(Path({str(onnx_path)!r}), Path({str(tflite_path)!r}))\n"
        "except TFLiteExportError as exc:\n"
        "    print(f'TFLITE_EXPORT_ERROR: {exc}', file=sys.stderr)\n"
        "    sys.exit(2)\n"
    )

    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise TFLiteExportError(
            f"La conversión a .tflite no terminó en {timeout:.0f}s incluso en un "
            "proceso aislado — no es el conflicto torch/tensorflow habitual. "
            f"El .onnx sigue siendo válido; reintenta luego con 'wake-trainer "
            f"convert' o revisa manualmente."
        ) from exc

    if result.returncode != 0:
        stderr_tail = "\n".join(result.stderr.strip().splitlines()[-15:])
        if "TFLITE_EXPORT_ERROR:" in result.stderr:
            msg = result.stderr.strip().splitlines()[-1].split("TFLITE_EXPORT_ERROR: ", 1)[-1]
            raise TFLiteExportError(msg)
        raise TFLiteExportError(
            f"La conversión a .tflite falló en el subproceso (código {result.returncode}). "
            f"Últimas líneas de stderr:\n{stderr_tail}"
        )

    if not tflite_path.exists():
        raise TFLiteExportError(
            "El subproceso de conversión terminó sin error pero no generó "
            f"{tflite_path}. Revisa manualmente con 'wake-trainer convert'."
        )

    return tflite_path
