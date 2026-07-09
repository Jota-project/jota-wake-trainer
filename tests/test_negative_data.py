# tests/test_negative_data.py
import numpy as np
import pytest
from pathlib import Path
import trainer.negative_data as negdata


def _fake_validation_file(path: Path, n_rows: int = 100):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.random.rand(n_rows, 16, 96).astype(np.float32))


def _fake_flat_features_file(path: Path, n_frames: int = 100, n_features: int = 96):
    """
    Simula el formato REAL de validation_set_features.npy / ACAV100M tal y
    como los publica HuggingFace: una secuencia continua de frames (2D), sin
    cortar en clips/ventanas — no el formato ya-ventanado (3D) que usaban los
    fixtures anteriores.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.random.rand(n_frames, n_features).astype(np.float32))


def test_ensure_val_slice_creates_expected_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(negdata, "VAL_NEG_ROWS", 10)
    validation_path = tmp_path / "validation_set_features.npy"
    _fake_validation_file(validation_path, n_rows=100)

    work_dir = tmp_path / "work"
    result = negdata._ensure_val_slice(validation_path, work_dir)

    assert result.exists()
    data = np.load(result)
    assert data.shape == (10, 16, 96)


def test_ensure_val_slice_is_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(negdata, "VAL_NEG_ROWS", 5)
    validation_path = tmp_path / "validation_set_features.npy"
    _fake_validation_file(validation_path, n_rows=50)
    work_dir = tmp_path / "work"

    first = negdata._ensure_val_slice(validation_path, work_dir)
    mtime_first = first.stat().st_mtime

    second = negdata._ensure_val_slice(validation_path, work_dir)
    assert second == first
    assert second.stat().st_mtime == mtime_first


def test_ensure_negative_features_quick_mode_reuses_validation_set(tmp_path, monkeypatch):
    validation_path = tmp_path / "validation_set_features.npy"
    _fake_validation_file(validation_path, n_rows=40)

    monkeypatch.setattr(negdata, "ensure_validation_features", lambda root: validation_path)
    monkeypatch.setattr(
        negdata, "ensure_acav_features",
        lambda root: (_ for _ in ()).throw(AssertionError("no debería descargar ACAV100M en modo quick")),
    )

    work_dir = tmp_path / "work"
    result = negdata.ensure_negative_features(work_dir, mode="quick", root=tmp_path)

    assert result["train"] == validation_path
    assert result["false_positive_val"] == validation_path
    assert result["val"].exists()


def test_ensure_negative_features_full_mode_downloads_acav(tmp_path, monkeypatch):
    validation_path = tmp_path / "validation_set_features.npy"
    acav_path = tmp_path / "acav.npy"
    _fake_validation_file(validation_path, n_rows=40)
    _fake_validation_file(acav_path, n_rows=5)

    monkeypatch.setattr(negdata, "ensure_validation_features", lambda root: validation_path)
    monkeypatch.setattr(negdata, "ensure_acav_features", lambda root: acav_path)

    work_dir = tmp_path / "work"
    result = negdata.ensure_negative_features(work_dir, mode="full", root=tmp_path)

    assert result["train"] == acav_path
    assert result["false_positive_val"] == validation_path


# ── build_windowed_features ──────────────────────────────────────────────────
#
# Regresión directa del bug real: validation_set_features.npy y el ACAV100M
# que publica openWakeWord en HuggingFace son arrays 2D (n_frames, 96) —
# audio continuo sin cortar en clips —, NO arrays ya-ventanados 3D. Al pasar
# uno de esos directamente a openwakeword.data.mmap_batch_generator, su
# constructor revienta con "IndexError: tuple index out of range" al intentar
# leer `shape[2]` de una tupla de longitud 2. Nuestra propia validación
# (`_validate_feature_file`) lo señala de forma clara, pero el arreglo real
# es ventanar el array antes de usarlo.

def test_build_windowed_features_converts_flat_array(tmp_path):
    source = tmp_path / "validation_set_features.npy"
    _fake_flat_features_file(source, n_frames=160, n_features=96)

    result = negdata.build_windowed_features(source, window_steps=16)

    assert result != source
    data = np.load(result)
    assert data.shape == (10, 16, 96)  # 160 frames / 16 pasos = 10 ventanas


def test_build_windowed_features_drops_remainder_frames(tmp_path):
    source = tmp_path / "flat.npy"
    _fake_flat_features_file(source, n_frames=165, n_features=96)  # 165 = 10*16 + 5 sobrantes

    result = negdata.build_windowed_features(source, window_steps=16)

    data = np.load(result)
    assert data.shape == (10, 16, 96)


def test_build_windowed_features_is_cached(tmp_path):
    source = tmp_path / "flat.npy"
    _fake_flat_features_file(source, n_frames=64)

    first = negdata.build_windowed_features(source, window_steps=16)
    mtime_first = first.stat().st_mtime

    second = negdata.build_windowed_features(source, window_steps=16)

    assert second == first
    assert second.stat().st_mtime == mtime_first


def test_build_windowed_features_passthrough_when_already_3d(tmp_path):
    source = tmp_path / "already_windowed.npy"
    _fake_validation_file(source, n_rows=5)

    result = negdata.build_windowed_features(source, window_steps=16)

    assert result == source


def test_build_windowed_features_raises_when_not_enough_frames(tmp_path):
    source = tmp_path / "too_short.npy"
    _fake_flat_features_file(source, n_frames=5)

    with pytest.raises(ValueError):
        negdata.build_windowed_features(source, window_steps=16)


def test_ensure_negative_features_windows_flat_validation_set(tmp_path, monkeypatch):
    """
    Regresión end-to-end: incluso partiendo del formato REAL de HuggingFace
    (2D, sin ventanar), `ensure_negative_features` debe devolver ficheros ya
    ventanados en (N, 16, 96), listos para `mmap_batch_generator`.
    """
    validation_path = tmp_path / "validation_set_features.npy"
    _fake_flat_features_file(validation_path, n_frames=320)

    monkeypatch.setattr(negdata, "ensure_validation_features", lambda root: validation_path)
    monkeypatch.setattr(
        negdata, "ensure_acav_features",
        lambda root: (_ for _ in ()).throw(AssertionError("no debería descargar ACAV100M en modo quick")),
    )

    work_dir = tmp_path / "work"
    result = negdata.ensure_negative_features(work_dir, mode="quick", root=tmp_path)

    for key in ("train", "val", "false_positive_val"):
        data = np.load(result[key])
        assert data.ndim == 3, f"'{key}' no está ventanado: shape {data.shape}"
        assert data.shape[1] == 16
        assert data.shape[0] > 0
