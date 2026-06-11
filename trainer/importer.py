# trainer/importer.py
from __future__ import annotations
import shutil
import soundfile as sf
from pathlib import Path

SAMPLE_RATE = 16000
SUPPORTED_SUFFIXES = {".wav"}


def scan_directory(source_dir: Path) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Escanea source_dir y clasifica WAVs en válidos e inválidos."""
    valid: list[Path] = []
    invalid: list[tuple[Path, str]] = []

    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            invalid.append((path, f"Formato no soportado: {path.suffix}"))
            continue
        try:
            info = sf.info(str(path))
            if info.samplerate != SAMPLE_RATE:
                invalid.append((path, f"Sample rate: {info.samplerate} Hz (esperado {SAMPLE_RATE})"))
            elif info.channels != 1:
                invalid.append((path, f"Canales: {info.channels} (esperado mono)"))
            else:
                valid.append(path)
        except Exception as exc:
            invalid.append((path, f"No se puede leer: {exc}"))

    return valid, invalid


def import_clips(source_dir: Path, target_dir: Path) -> tuple[int, list[tuple[Path, str]]]:
    """Copia WAVs válidos de source_dir a target_dir con numeración secuencial."""
    target_dir.mkdir(parents=True, exist_ok=True)
    valid, invalid = scan_directory(source_dir)

    existing_count = len(list(target_dir.glob("*.wav")))
    for i, wav_path in enumerate(valid, start=existing_count + 1):
        dest = target_dir / f"{i:03d}.wav"
        shutil.copy2(str(wav_path), str(dest))

    return len(valid), invalid
