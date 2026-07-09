# trainer/audio_utils.py
"""
Utilidades para garantizar que todo el audio que entra en el pipeline de
openWakeWord (grabado, importado o sintetizado) esté en el formato que
`openwakeword.data.augment_clips` exige de forma estricta: 16 kHz, mono.

Por qué existe este módulo: los distintos modelos de voz de Piper NO
comparten sample rate. Por ejemplo, en un set típico de voces en español:

    es_ES-carlfm-x_low     -> 16000 Hz
    es_ES-mls_10246-low    -> 16000 Hz
    es_ES-mls_9972-low     -> 16000 Hz
    es_ES-davefx-medium    -> 22050 Hz
    es_ES-sharvard-medium  -> 22050 Hz
    es_AR-daniela-high     -> 22050 Hz
    es_MX-ald-medium       -> 22050 Hz
    es_MX-claude-high      -> 22050 Hz

Piper no resamplea su salida al sample rate del modelo — escribe el WAV al
sample rate nativo de cada voz. `augment_clips` (en openwakeword/data.py)
comprueba `clip_sr != sr` (con `sr=16000` fijo) y lanza
`ValueError("Error! Clip does not have the correct sample rate!")` sin decir
qué fichero es, así que un dataset mezclando varias voces Piper revienta el
entrenamiento en cuanto le toca un clip a 22050 Hz.
"""
from __future__ import annotations
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SAMPLE_RATE = 16000


def resample_to_target(data: np.ndarray, orig_sr: int, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Resamplea un array 1D de audio de orig_sr a target_sr."""
    if orig_sr == target_sr:
        return data
    from scipy.signal import resample_poly
    g = gcd(int(orig_sr), int(target_sr))
    return resample_poly(data, target_sr // g, orig_sr // g)


def to_mono(data: np.ndarray) -> np.ndarray:
    """Reduce un array (samples, channels) a mono promediando canales."""
    if data.ndim > 1:
        return data.mean(axis=1)
    return data


def write_wav_16k_mono(data: np.ndarray, sr: int, output_path: Path) -> None:
    """Escribe `data` como WAV mono PCM_16 a TARGET_SAMPLE_RATE, resampleando si hace falta."""
    data = to_mono(np.asarray(data, dtype=np.float32))
    data = resample_to_target(data, sr)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), data, TARGET_SAMPLE_RATE, subtype="PCM_16")


def needs_repair(path: Path) -> bool:
    """True si el WAV en `path` no está ya en 16kHz mono."""
    info = sf.info(str(path))
    return info.samplerate != TARGET_SAMPLE_RATE or info.channels != 1


def ensure_wav_16k_mono(path: Path) -> bool:
    """
    Si el WAV en `path` no está en 16kHz mono, lo resamplea/mezcla a mono y
    lo reescribe en el mismo sitio. Devuelve True si tuvo que corregirlo,
    False si ya estaba bien (no-op barato vía `sf.info`, sin cargar el audio).
    """
    if not needs_repair(path):
        return False
    data, sr = sf.read(str(path), dtype="float32")
    write_wav_16k_mono(data, sr, path)
    return True


def repair_clips(paths: list[Path]) -> tuple[list[Path], int]:
    """
    Aplica `ensure_wav_16k_mono` a una lista de clips, en el sitio. Los
    ficheros que no se puedan leer (corruptos, formato no soportado) se
    descartan de la lista devuelta en vez de abortar todo el proceso — el
    resto del dataset sigue siendo utilizable.

    Devuelve (clips_utilizables, cuántos tuvieron que corregirse). Todos los
    clips devueltos están ya garantizados 16kHz mono.
    """
    usable: list[Path] = []
    repaired_count = 0
    for path in paths:
        try:
            if ensure_wav_16k_mono(path):
                repaired_count += 1
            usable.append(path)
        except Exception:
            continue
    return usable, repaired_count
