#!/usr/bin/env python3
# scripts/clone_voice_xtts.py
"""
Genera clips positivos sintéticos clonando una voz con Coqui XTTS v2 — 100%
gratis y local, sin ninguna API de pago. Pensado para el caso de Magarma:
no necesita grabar nada nuevo, basta con audios que ya tengas de él (notas
de voz, vídeos...) usados como referencia de clonado.

Por qué XTTS v2 y no otra cosa:
  - Clona una voz a partir de un puñado de segundos de audio de referencia
    (más/mejores clips de referencia = clon más fiel).
  - Soporta español de forma nativa (`language="es"`).
  - Corre enteramente en local (CPU u opcionalmente GPU) — no manda audio a
    ningún servicio de terceros, no consume cuota ni cuesta dinero.
  - Licencia: el código de la librería es MPL 2.0, pero los pesos del
    modelo (que se descargan la primera vez, ~2GB) van bajo la Coqui Public
    Model License (CPML) — uso no comercial. Para un proyecto personal
    como Jota esto no supone ningún problema.

Uso:
    pip install coqui-tts
    python3 scripts/clone_voice_xtts.py \\
        --speaker-wav ref_magarma_1.wav ref_magarma_2.wav \\
        --out-dir projects/ok_jota/data/sintetizados \\
        --prefix clone_magarma \\
        --n 40

Los WAV resultantes se escriben directamente en `--out-dir` ya normalizados
a 16kHz mono (misma lógica que trainer/audio_utils.py, pero duplicada aquí
a propósito — ver nota IMPORTANTE más abajo) — si apuntas a
`data/sintetizados` de un proyecto, `wake-trainer train` los recoge
automáticamente la próxima vez, sin más pasos ni wiring adicional.

La primera ejecución descarga el checkpoint de XTTS v2 (~2GB) desde
HuggingFace — puede tardar varios minutos. Las siguientes ejecuciones son
mucho más rápidas (el modelo se queda cacheado).

IMPORTANTE — por qué este script NO importa `trainer.audio_utils` aunque
haría el código más corto: el paquete de PyPI `coqui-tts-trainer` (una
dependencia de `coqui-tts`) se instala bajo el nombre importable `trainer`
— exactamente el mismo nombre que el paquete de este propio repo
(`trainer/`, con `cli.py`, `state.py`, etc.). En el mismo proceso solo puede
existir un módulo llamado `trainer` en `sys.modules`: en cuanto se hace
`from TTS.api import TTS`, coqui-tts importa su propio `trainer` (el de
PyPI) primero, y cualquier `from trainer.audio_utils import ...` posterior
usaría ESE `trainer` (el de coqui-tts-trainer, que no tiene `audio_utils`) y
rompería con `ImportError: cannot import name 'TrainerConfig' from
'trainer'` o similar. Por eso aquí se reimplementa el resampleo/mono a mano
con soundfile+scipy en vez de importar nada del paquete `trainer` del repo.
"""
from __future__ import annotations
import argparse
import sys
from math import gcd
from pathlib import Path

import numpy as np


def _write_wav_16k_mono(data: np.ndarray, sr: int, output_path: Path) -> None:
    """Copia local (sin depender del paquete `trainer` del repo — ver nota arriba)
    de trainer.audio_utils.write_wav_16k_mono: mezcla a mono, resamplea a 16kHz
    y escribe PCM_16."""
    import soundfile as sf
    from scipy.signal import resample_poly

    TARGET_SR = 16000
    data = np.asarray(data, dtype=np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != TARGET_SR:
        g = gcd(int(sr), int(TARGET_SR))
        data = resample_poly(data, TARGET_SR // g, sr // g)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), data, TARGET_SR, subtype="PCM_16")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--speaker-wav", required=True, nargs="+",
        help="Uno o más ficheros WAV de referencia de la voz a clonar (más clips = clon más fiel).",
    )
    parser.add_argument("--text", default="ok jota", help="Frase a sintetizar (por defecto, la wake word).")
    parser.add_argument("--out-dir", required=True, help="Carpeta de destino (p.ej. data/sintetizados del proyecto).")
    parser.add_argument("--prefix", default="clone", help="Prefijo de los ficheros generados.")
    parser.add_argument("--n", type=int, default=40, help="Cuántas variantes generar.")
    parser.add_argument("--language", default="es", help="Código de idioma XTTS (es, en, fr...).")
    args = parser.parse_args()

    for ref in args.speaker_wav:
        if not Path(ref).exists():
            print(f"No existe el fichero de referencia: {ref}", file=sys.stderr)
            sys.exit(1)

    try:
        from TTS.api import TTS
    except ImportError as exc:
        if "torch" in str(exc).lower() or "torch" in repr(getattr(exc, "name", "")).lower():
            print(
                "Falta PyTorch (coqui-tts no lo instala solo, hay que ponerlo aparte):\n"
                "  pip install torch torchaudio\n"
                f"\nError original: {exc}",
                file=sys.stderr,
            )
        elif exc.name == "TTS" or str(exc).startswith("No module named 'TTS'"):
            print(
                "Falta 'coqui-tts'. Instálalo con:\n"
                "  pip install coqui-tts\n"
                "(usa el fork de la comunidad 'coqui-tts' en vez del 'TTS' original "
                "si tienes problemas de compatibilidad con tu versión de Python)",
                file=sys.stderr,
            )
        else:
            print(
                f"Fallo al importar TTS (coqui-tts) por una dependencia suya, no por coqui-tts en sí:\n"
                f"  {exc}\n"
                "Instala el paquete que falte (el nombre suele aparecer arriba, en 'No module named ...') "
                "y vuelve a intentarlo.",
                file=sys.stderr,
            )
        sys.exit(1)

    import soundfile as sf

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Cargando XTTS v2 (la primera vez descarga ~2GB desde HuggingFace)...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

    existing = len(list(out_dir.glob(f"{args.prefix}_*.wav")))
    generated = 0
    for i in range(args.n):
        idx = existing + i + 1
        tmp_path = out_dir / f"_tmp_{args.prefix}_{idx}.wav"
        out_path = out_dir / f"{args.prefix}_{idx:03d}.wav"
        try:
            tts.tts_to_file(
                text=args.text,
                speaker_wav=args.speaker_wav,
                language=args.language,
                file_path=str(tmp_path),
            )
            data, sr = sf.read(str(tmp_path), dtype="float32")
            _write_wav_16k_mono(data, sr, out_path)
            generated += 1
            print(f"  ✓ {out_path.name}")
        except Exception as exc:
            print(f"  ✗ Fallo en la variante {idx}: {exc}", file=sys.stderr)
        finally:
            tmp_path.unlink(missing_ok=True)

    print(f"\n{generated}/{args.n} clips generados en {out_dir}")
    if generated == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
