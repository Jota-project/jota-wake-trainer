# tests/test_importer.py
import pytest
import numpy as np
import soundfile as sf
from pathlib import Path
from trainer.importer import scan_directory, import_clips


def make_wav(path: Path, sr: int = 16000, channels: int = 1, duration: float = 1.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (np.random.randn(int(duration * sr)) * 0.3).astype(np.float32)
    if channels > 1:
        data = np.stack([data] * channels, axis=1)
    sf.write(str(path), data, sr, subtype="PCM_16")


def test_scan_finds_valid_wavs(tmp_path):
    make_wav(tmp_path / "a.wav")
    make_wav(tmp_path / "b.wav")
    valid, invalid = scan_directory(tmp_path)
    assert len(valid) == 2
    assert len(invalid) == 0


def test_scan_rejects_wrong_format(tmp_path):
    make_wav(tmp_path / "good.wav")
    (tmp_path / "bad.m4a").write_bytes(b"fake")
    valid, invalid = scan_directory(tmp_path)
    assert len(valid) == 1
    assert len(invalid) == 1
    assert "m4a" in invalid[0][1].lower()


def test_scan_rejects_wrong_sample_rate(tmp_path):
    make_wav(tmp_path / "bad_sr.wav", sr=44100)
    valid, invalid = scan_directory(tmp_path)
    assert len(valid) == 0
    assert "44100" in invalid[0][1]


def test_scan_rejects_stereo(tmp_path):
    make_wav(tmp_path / "stereo.wav", channels=2)
    valid, invalid = scan_directory(tmp_path)
    assert not valid
    assert any("mono" in msg.lower() for _, msg in invalid)


def test_import_copies_valid_wavs(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    make_wav(src / "clip1.wav")
    make_wav(src / "clip2.wav")
    count, invalid = import_clips(src, dst)
    assert count == 2
    assert len(list(dst.glob("*.wav"))) == 2
    assert len(invalid) == 0


def test_import_numbers_clips_sequentially(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    make_wav(src / "x.wav")
    make_wav(src / "y.wav")
    import_clips(src, dst)
    names = sorted(p.name for p in dst.glob("*.wav"))
    assert names == ["001.wav", "002.wav"]


def test_import_continues_numbering_if_existing(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    dst.mkdir(parents=True)
    make_wav(dst / "001.wav")  # ya existe
    make_wav(src / "new.wav")
    import_clips(src, dst)
    names = sorted(p.name for p in dst.glob("*.wav"))
    assert "002.wav" in names
