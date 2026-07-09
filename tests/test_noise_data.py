# tests/test_noise_data.py
"""
Tests para trainer/noise_data.py (RIR + ruido de fondo real para
augmentación). No dependen del paquete 'datasets' real estando instalado:
- Los tests de "ya en caché" comprueban que la función no lo importa
  siquiera si ya hay ficheros en disco (return temprano).
- El test de descarga inyecta un módulo 'datasets' falso mínimo en
  sys.modules, para no depender de la librería real ni de red.
"""
from __future__ import annotations
import sys
import types
from pathlib import Path

import numpy as np

from trainer.noise_data import (
    ensure_rir_clips,
    ensure_fma_noise_clips,
    ensure_audioset_noise_clips,
    ensure_background_noise_clips,
)


def test_ensure_rir_clips_returns_cached_without_importing_datasets(tmp_path, monkeypatch):
    dest = tmp_path / "rir"
    dest.mkdir()
    (dest / "ir_001.wav").write_bytes(b"fake")
    (dest / "ir_002.wav").write_bytes(b"fake")

    # Si 'datasets' no está instalado en absoluto, importarlo lanzaría
    # ModuleNotFoundError — nos aseguramos de que ni se intente cuando ya
    # hay clips cacheados.
    monkeypatch.setitem(sys.modules, "datasets", None)

    paths = ensure_rir_clips(dest)
    assert len(paths) == 2
    assert all(Path(p).exists() for p in paths)


def test_ensure_fma_noise_clips_returns_cached_without_importing_datasets(tmp_path, monkeypatch):
    dest = tmp_path / "noise"
    (dest / "fma").mkdir(parents=True)
    (dest / "fma" / "clip_001.wav").write_bytes(b"fake")

    monkeypatch.setitem(sys.modules, "datasets", None)

    paths = ensure_fma_noise_clips(dest)
    assert len(paths) == 1


def _fake_datasets_module(rows: list[dict]) -> types.ModuleType:
    """Construye un módulo 'datasets' mínimo con load_dataset()/cast_column()
    suficiente para que noise_data.py funcione sin la librería real."""
    mod = types.ModuleType("datasets")

    class FakeDataset:
        def __init__(self, rows):
            self._rows = rows

        def cast_column(self, *_args, **_kwargs):
            return self

        def __iter__(self):
            return iter(self._rows)

    def load_dataset(*_args, **_kwargs):
        return FakeDataset(rows)

    def Audio(*_args, **_kwargs):
        return None

    mod.load_dataset = load_dataset
    mod.Audio = Audio
    return mod


def test_ensure_rir_clips_downloads_and_writes_16k_mono_wavs(tmp_path, monkeypatch):
    dest = tmp_path / "rir"
    rows = [
        {"audio": {"path": "ir_a.wav", "array": np.zeros(1600, dtype=np.float32), "sampling_rate": 16000}},
        {"audio": {"path": "ir_b.wav", "array": np.zeros(800, dtype=np.float32), "sampling_rate": 16000}},
    ]
    monkeypatch.setitem(sys.modules, "datasets", _fake_datasets_module(rows))

    paths = ensure_rir_clips(dest)
    assert len(paths) == 2
    for p in paths:
        assert Path(p).exists()

    # Segunda llamada: ya está cacheado, no debería intentar nada más (y
    # si intentara reimportar 'datasets' de verdad, este test lo pillaría
    # porque seguimos con el módulo falso inyectado).
    paths_again = ensure_rir_clips(dest)
    assert sorted(paths_again) == sorted(paths)


def test_ensure_audioset_noise_clips_uses_parquet_dataset_not_tar(tmp_path, monkeypatch):
    """
    Regresión: la primera versión de esta función bajaba un `.tar` fijo por
    URL (bal_train09.tar) que ya no existe — el repo de HuggingFace se
    reorganizó a formato parquet. Confirma que ahora se usa
    datasets.load_dataset(..., streaming=True) igual que RIR/FMA, sin tocar
    red ni tarfile para nada.
    """
    dest = tmp_path / "noise"
    rows = [
        {"audio": {"path": "as_1.flac", "array": np.zeros(1600, dtype=np.float32), "sampling_rate": 16000}},
        {"audio": {"path": "as_2.flac", "array": np.zeros(1600, dtype=np.float32), "sampling_rate": 16000}},
    ]

    captured = {}

    def load_dataset(name, config, split, streaming):
        captured["name"] = name
        captured["config"] = config
        captured["split"] = split
        captured["streaming"] = streaming
        return _fake_datasets_module(rows).load_dataset()

    fake_mod = _fake_datasets_module(rows)
    fake_mod.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_mod)

    paths = ensure_audioset_noise_clips(dest, hours=0.01)
    assert len(paths) >= 1
    assert captured == {
        "name": "agkphysics/AudioSet", "config": "balanced", "split": "train", "streaming": True,
    }
    # Nada de tarfile/httpx de por medio: no debe haber quedado ningún .tar.
    assert not list(dest.glob("*.tar"))


def test_ensure_background_noise_clips_uses_audioset_in_both_modes_with_more_hours_when_full(tmp_path, monkeypatch):
    import trainer.noise_data as noise_data_mod

    audioset_hours_seen = []
    monkeypatch.setattr(
        noise_data_mod, "ensure_audioset_noise_clips",
        lambda dest_dir, hours=2.0: audioset_hours_seen.append(hours) or ["as1.wav"],
    )
    monkeypatch.setattr(
        noise_data_mod, "ensure_fma_noise_clips",
        lambda dest_dir, hours=0.5: ["fma1.wav"],
    )

    paths = ensure_background_noise_clips(tmp_path, full=False)
    assert set(paths) == {"as1.wav", "fma1.wav"}

    paths_full = ensure_background_noise_clips(tmp_path, full=True)
    assert set(paths_full) == {"as1.wav", "fma1.wav"}

    # Full pide más horas de AudioSet que quick.
    assert audioset_hours_seen[0] < audioset_hours_seen[1]


def test_ensure_background_noise_clips_keeps_audioset_when_fma_fails(tmp_path, monkeypatch):
    """
    Regresión del bug real: FMA falló con 'Cannot seek streaming HTTP file'
    y, al no estar aislado, se llevaba por delante el resultado de AudioSet
    que sí había funcionado. FMA es un extra opcional — su fallo no debe
    tirar nada de lo que ya se consiguió.
    """
    import trainer.noise_data as noise_data_mod

    monkeypatch.setattr(noise_data_mod, "ensure_audioset_noise_clips", lambda dest_dir, hours=2.0: ["as1.wav"])

    def boom_fma(dest_dir, hours=0.5):
        raise RuntimeError("Cannot seek streaming HTTP file")

    monkeypatch.setattr(noise_data_mod, "ensure_fma_noise_clips", boom_fma)

    paths = ensure_background_noise_clips(tmp_path, full=False)
    assert paths == ["as1.wav"]
