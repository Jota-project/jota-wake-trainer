# tests/test_recorder.py
import pytest
import numpy as np
import soundfile as sf
from pathlib import Path
from unittest.mock import patch, MagicMock
from trainer.recorder import validate_clip, CONDITIONS, SAMPLE_RATE


def make_wav(path: Path, duration: float = 1.5, sr: int = 16000,
             channels: int = 1, peak: float = 0.3):
    samples = int(duration * sr)
    data = (np.random.randn(samples) * peak).astype(np.float32)
    if channels > 1:
        data = np.stack([data] * channels, axis=1)
    sf.write(str(path), data, sr, subtype="PCM_16")


def test_validate_clip_valid(tmp_path):
    wav = tmp_path / "ok.wav"
    make_wav(wav)
    valid, error = validate_clip(wav)
    assert valid
    assert error == ""


def test_validate_clip_wrong_sample_rate(tmp_path):
    wav = tmp_path / "bad_sr.wav"
    make_wav(wav, sr=44100)
    valid, error = validate_clip(wav)
    assert not valid
    assert "44100" in error


def test_validate_clip_stereo(tmp_path):
    wav = tmp_path / "stereo.wav"
    make_wav(wav, channels=2)
    valid, error = validate_clip(wav)
    assert not valid
    assert "mono" in error.lower()


def test_validate_clip_too_short(tmp_path):
    wav = tmp_path / "short.wav"
    make_wav(wav, duration=0.2)
    valid, error = validate_clip(wav)
    assert not valid
    assert "corto" in error


def test_validate_clip_too_quiet(tmp_path):
    wav = tmp_path / "quiet.wav"
    make_wav(wav, peak=0.001)
    valid, error = validate_clip(wav)
    assert not valid
    assert "bajo" in error


def test_validate_clip_saturated(tmp_path):
    wav = tmp_path / "sat.wav"
    make_wav(wav, peak=1.0)
    valid, error = validate_clip(wav)
    assert not valid
    assert "saturada" in error


def test_conditions_total_clips():
    total = sum(c["clips"] for c in CONDITIONS)
    assert total == 30


def test_conditions_count():
    assert len(CONDITIONS) == 10
