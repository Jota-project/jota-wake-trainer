# trainer/negative_data.py
"""
Adquisición de datos negativos para entrenar openWakeWord.

openWakeWord entrena un clasificador binario: necesita, además de clips
positivos, una cantidad grande y diversa de audio que NO contiene la wake
word. Sin esto no hay forma de que el modelo aprenda a distinguir nada —
es la pieza que faltaba por completo en la versión anterior de este repo.

En vez de recopilar y procesar audio en bruto (que exigiría bajar decenas
de GB y calcular features nosotros mismos), usamos el dataset de features
precalculadas que publica el propio autor de openWakeWord en HuggingFace:

  https://huggingface.co/datasets/davidscripka/openwakeword_features

Dos ficheros:

  - validation_set_features.npy   (~11.3 h · ~190 MB)
    Set de validación de falsos positivos "oficial" (DiPCo + Santa Bárbara
    Corpus + MUSDB). Se usa siempre, tal cual, para medir falsos positivos
    por hora — es el mismo criterio que usa el notebook oficial de HA.

  - openwakeword_features_ACAV100M_2000_hrs_16bit.npy   (~2000 h · ~17 GB)
    Dataset general de negativos (voz, ruido, música de ACAV100M). Solo se
    descarga en modo "full".

Modo "quick" (por defecto): no descarga los 17 GB. Reutiliza el set de
validación también como negativo de entrenamiento. El modelo resultante es
razonable para uso personal, pero menos robusto frente a falsos positivos
que un modelo "full". Se puede volver a entrenar en modo "full" más tarde
sin perder nada de lo ya grabado/sintetizado.
"""
from __future__ import annotations
from pathlib import Path
from typing import Literal
import numpy as np
from numpy.lib.format import open_memmap
import httpx
from rich.progress import (
    Progress, BarColumn, DownloadColumn, TransferSpeedColumn,
    TimeRemainingColumn, TextColumn,
)

HF_BASE = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main"
VALIDATION_URL = f"{HF_BASE}/validation_set_features.npy"
ACAV_URL = f"{HF_BASE}/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"

# Compartido entre proyectos: son ficheros grandes y reutilizables tal cual.
NEGATIVE_FEATURES_ROOT = Path("data/negative_features")

VAL_NEG_ROWS = 5000  # filas reservadas del set de validación para X_val (aparte del gating de FP/hora)


def _download_stream(url: str, dest: Path, description: str) -> Path:
    """Descarga con barra de progreso y soporte de reanudación (Range)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    resume_from = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}

    with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=None) as resp:
        if resp.status_code == 416:
            # Ya está completo (el servidor no tiene más bytes que dar)
            tmp.rename(dest)
            return dest
        if resp.status_code not in (200, 206):
            resp.raise_for_status()

        content_length = int(resp.headers.get("content-length", 0))
        total = content_length + resume_from if resp.status_code == 206 else content_length
        mode = "ab" if resp.status_code == 206 else "wb"

        with open(tmp, mode) as f, Progress(
            TextColumn("[cyan]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(description, total=total or None,
                                     completed=resume_from if resp.status_code == 206 else 0)
            for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                f.write(chunk)
                progress.update(task, advance=len(chunk))

    tmp.rename(dest)
    return dest


def ensure_validation_features(root: Path = NEGATIVE_FEATURES_ROOT) -> Path:
    """Descarga (si hace falta) el set de validación de falsos positivos (~190 MB)."""
    dest = root / "validation_set_features.npy"
    if not dest.exists():
        _download_stream(
            VALIDATION_URL, dest,
            "Set de validación FP (~11.3 h, ~190 MB)",
        )
    return dest


def ensure_acav_features(root: Path = NEGATIVE_FEATURES_ROOT) -> Path:
    """Descarga (si hace falta) el dataset ACAV100M completo (~17 GB). Solo modo 'full'."""
    dest = root / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    if not dest.exists():
        _download_stream(
            ACAV_URL, dest,
            "ACAV100M — negativos generales (~2000 h, ~17 GB, puede tardar horas)",
        )
    return dest


def _ensure_val_slice(validation_path: Path, work_dir: Path) -> Path:
    """Reserva un pequeño trozo del set de validación como negativo de X_val."""
    val_slice_path = work_dir / "negative_features_val.npy"
    if not val_slice_path.exists():
        data = np.load(validation_path, mmap_mode="r")
        n_val = min(VAL_NEG_ROWS, max(1, data.shape[0] // 4))
        work_dir.mkdir(parents=True, exist_ok=True)
        np.save(val_slice_path, np.array(data[-n_val:]))
    return val_slice_path


def _windowed_cache_path(source_path: Path, window_steps: int) -> Path:
    return source_path.with_name(f"{source_path.stem}_win{window_steps}.npy")


def build_windowed_features(
    source_path: Path,
    window_steps: int = 16,
    chunk_windows: int = 8192,
) -> Path:
    """
    Corta un array de features "planas" (n_frames, 96) — audio continuo, un
    frame cada ~80ms, sin cortar en clips — en ventanas fijas no solapadas de
    forma (N, window_steps, 96), y lo cachea en disco junto al original.

    Por qué hace falta: los ficheros que publica openWakeWord en HuggingFace
    (`validation_set_features.npy`, el ACAV100M de modo `--full`) son
    exactamente ese tipo de array plano — el propio notebook oficial de
    openWakeWord solo los usa así para calcular falsos positivos por hora,
    troceándolos a mano con una ventana deslizante justo antes de usarlos.
    Pero `openwakeword.data.mmap_batch_generator` (la clase que alimenta el
    entrenamiento) asume en su propio constructor que TODOS los ficheros que
    recibe en `data_files` ya son arrays 3D (N, pasos, features) — si alguno
    es 2D, revienta con un `IndexError: tuple index out of range` genérico al
    intentar leer `shape[2]`. Ventanamos aquí, una sola vez y cacheado, antes
    de que esos ficheros lleguen a `mmap_batch_generator`.

    Usa un memmap de salida y procesa por bloques para no cargar en RAM
    ficheros grandes — relevante sobre todo en modo `--full`, donde el
    dataset ACAV100M completo pesa ~17 GB.
    """
    dest = _windowed_cache_path(source_path, window_steps)
    if dest.exists():
        return dest

    src = np.load(source_path, mmap_mode="r")
    if src.ndim == 3:
        # Ya viene pre-cortado (p.ej. si HuggingFace cambia el formato de
        # publicación en el futuro) — nada que hacer.
        return source_path

    n_features = src.shape[1]
    n_windows = src.shape[0] // window_steps
    if n_windows == 0:
        raise ValueError(
            f"'{source_path}' tiene solo {src.shape[0]} frames, no alcanza "
            f"para ni una ventana de {window_steps} pasos."
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.npy")
    out = open_memmap(tmp, mode="w+", dtype=np.float32, shape=(n_windows, window_steps, n_features))
    for start in range(0, n_windows, chunk_windows):
        end = min(start + chunk_windows, n_windows)
        frame_chunk = np.asarray(src[start * window_steps:end * window_steps], dtype=np.float32)
        out[start:end] = frame_chunk.reshape(end - start, window_steps, n_features)
    out.flush()
    del out
    tmp.rename(dest)
    return dest


def ensure_negative_features(
    work_dir: Path,
    mode: Literal["quick", "full"] = "quick",
    root: Path = NEGATIVE_FEATURES_ROOT,
    window_steps: int = 16,
) -> dict[str, Path]:
    """
    Prepara los tres ficheros de features negativos necesarios para entrenar,
    todos ya ventanados a (N, window_steps, 96) y listos para pasar
    directamente a `mmap_batch_generator`:

      - "train": negativos usados como clase 0 durante el entrenamiento
      - "val": pequeño negativo balanceado para la métrica de accuracy/recall
      - "false_positive_val": set de ~11.3h usado para medir falsos positivos/hora

    En modo "quick" reutiliza el set de validación como negativo de train
    (aviso: menos diverso, pero no requiere bajar 17 GB). En modo "full"
    descarga ACAV100M como negativo de entrenamiento dedicado.
    """
    validation_path = ensure_validation_features(root)
    windowed_validation_path = build_windowed_features(validation_path, window_steps)

    val_slice_path = _ensure_val_slice(validation_path, work_dir)
    windowed_val_slice_path = build_windowed_features(val_slice_path, window_steps)

    if mode == "full":
        acav_path = ensure_acav_features(root)
        train_path = build_windowed_features(acav_path, window_steps)
    else:
        train_path = windowed_validation_path

    return {
        "train": train_path,
        "val": windowed_val_slice_path,
        "false_positive_val": windowed_validation_path,
    }
