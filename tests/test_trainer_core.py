# tests/test_trainer_core.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from trainer.trainer_core import collect_positive_clips, TrainingConfig, run_training


def test_collect_positive_clips_finds_wavs(tmp_path):
    person_dir = tmp_path / "Alfonso"
    person_dir.mkdir()
    (person_dir / "001.wav").touch()
    (person_dir / "002.wav").touch()
    (tmp_path / "ignored.txt").touch()

    clips = collect_positive_clips(tmp_path)
    assert len(clips) == 2
    assert all(str(c).endswith(".wav") for c in clips)


def test_collect_positive_clips_recurses_subdirectories(tmp_path):
    (tmp_path / "persona1").mkdir()
    (tmp_path / "persona2").mkdir()
    (tmp_path / "persona1" / "001.wav").touch()
    (tmp_path / "persona2" / "001.wav").touch()
    (tmp_path / "persona2" / "002.wav").touch()

    clips = collect_positive_clips(tmp_path)
    assert len(clips) == 3


def test_collect_positive_clips_empty_dir(tmp_path):
    assert collect_positive_clips(tmp_path) == []


def test_training_config_defaults():
    cfg = TrainingConfig(model_name="ok_jota", output_dir=Path("models"))
    assert cfg.epochs == 100
    assert cfg.batch_size == 32
    assert cfg.learning_rate == 1e-3


def test_run_training_calls_openwakeword(tmp_path):
    (tmp_path / "positivos" / "Alfonso").mkdir(parents=True)
    (tmp_path / "positivos" / "Alfonso" / "001.wav").touch()
    models_path = tmp_path / "models"
    models_path.mkdir()

    cfg = TrainingConfig(model_name="ok_jota", output_dir=models_path)

    with patch("trainer.trainer_core.oww_train") as mock_train:
        mock_train.train_custom_model = MagicMock()
        run_training(positivos_path=tmp_path / "positivos", config=cfg)
        mock_train.train_custom_model.assert_called_once()
