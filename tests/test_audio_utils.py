# tests/test_audio_utils.py
import numpy as np
import soundfile as sf
import pytest
from trainer.audio_utils import (
    TARGET_SAMPLE_RATE, resample_to_target, to_mono,
    write_wav_16k_mono, needs_repair, ensure_wav_16k_mono, repair_clips,
)


def _tone(sr: int, duration: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_resample_to_target_noop_when_already_correct():
    data = _tone(16000)
    out = resample_to_target(data, 16000)
    assert out is data


def test_resample_to_target_changes_length_proportionally():
    data = _tone(22050, duration=1.0)
    out = resample_to_target(data, 22050, target_sr=16000)
    expected_len = round(len(data) * 16000 / 22050)
    assert abs(len(out) - expected_len) <= 1


def test_to_mono_averages_channels():
    stereo = np.stack([np.ones(100), np.zeros(100)], axis=1).astype(np.float32)
    mono = to_mono(stereo)
    assert mono.shape == (100,)
    assert np.allclose(mono, 0.5)


def test_to_mono_passthrough_for_1d():
    data = _tone(16000)
    assert to_mono(data) is data


def test_write_wav_16k_mono_writes_correct_format(tmp_path):
    out = tmp_path / "clip.wav"
    data = _tone(22050)
    write_wav_16k_mono(data, 22050, out)
    info = sf.info(str(out))
    assert info.samplerate == TARGET_SAMPLE_RATE
    assert info.channels == 1


def test_needs_repair_false_for_correct_file(tmp_path):
    out = tmp_path / "ok.wav"
    sf.write(str(out), _tone(16000), 16000, subtype="PCM_16")
    assert needs_repair(out) is False


def test_needs_repair_true_for_wrong_sample_rate(tmp_path):
    out = tmp_path / "bad.wav"
    sf.write(str(out), _tone(22050), 22050, subtype="PCM_16")
    assert needs_repair(out) is True


def test_ensure_wav_16k_mono_fixes_in_place(tmp_path):
    """
    Caso real: una voz Piper como es_ES-davefx-medium escribe su WAV a
    22050 Hz. Tras `ensure_wav_16k_mono`, el mismo fichero debe quedar en
    16kHz mono, en el mismo path.
    """
    out = tmp_path / "davefx.wav"
    sf.write(str(out), _tone(22050), 22050, subtype="PCM_16")

    was_repaired = ensure_wav_16k_mono(out)

    assert was_repaired is True
    info = sf.info(str(out))
    assert info.samplerate == 16000
    assert info.channels == 1


def test_ensure_wav_16k_mono_is_noop_when_already_correct(tmp_path):
    out = tmp_path / "ok.wav"
    sf.write(str(out), _tone(16000), 16000, subtype="PCM_16")
    assert ensure_wav_16k_mono(out) is False


def test_repair_clips_fixes_mixed_sample_rates(tmp_path):
    """
    Regresión directa del bug: un dataset con clips Piper de varias voces
    (algunas a 16000 Hz, otras a 22050 Hz) debe quedar homogéneo tras
    `repair_clips`, sin perder ningún clip válido.
    """
    good = tmp_path / "good.wav"
    bad1 = tmp_path / "bad1.wav"
    bad2 = tmp_path / "bad2.wav"
    sf.write(str(good), _tone(16000), 16000, subtype="PCM_16")
    sf.write(str(bad1), _tone(22050), 22050, subtype="PCM_16")
    sf.write(str(bad2), _tone(22050), 22050, subtype="PCM_16")

    usable, n_repaired = repair_clips([good, bad1, bad2])

    assert n_repaired == 2
    assert usable == [good, bad1, bad2]
    for p in usable:
        info = sf.info(str(p))
        assert info.samplerate == 16000
        assert info.channels == 1


def test_repair_clips_skips_unreadable_files(tmp_path):
    good = tmp_path / "good.wav"
    corrupt = tmp_path / "corrupt.wav"
    sf.write(str(good), _tone(16000), 16000, subtype="PCM_16")
    corrupt.write_bytes(b"not a real wav file")

    usable, n_repaired = repair_clips([good, corrupt])

    assert usable == [good]
    assert n_repaired == 0
