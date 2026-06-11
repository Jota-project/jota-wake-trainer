# tests/test_evaluator.py
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from trainer.evaluator import EvaluationResult, evaluate_model


def test_evaluation_result_defaults():
    r = EvaluationResult(precision=0.94, recall=0.91, false_positives=0, threshold=0.3)
    assert r.precision == 0.94
    assert r.false_positives == 0


def test_evaluate_model_returns_result(tmp_path):
    model_path = tmp_path / "ok_jota.tflite"
    model_path.touch()

    positivos = tmp_path / "positivos"
    positivos.mkdir()
    (positivos / "001.wav").touch()

    mock_oww = MagicMock()
    mock_oww.predict.return_value = {"ok_jota": [0.8, 0.9, 0.1, 0.05]}

    with patch("trainer.evaluator.openwakeword") as mock_module, \
         patch("trainer.evaluator.sf") as mock_sf:
        mock_sf.read.return_value = (np.zeros(16000, dtype="int16"), 16000)
        mock_module.Model.return_value = mock_oww
        result = evaluate_model(model_path=model_path, positivos_path=positivos)

    assert isinstance(result, EvaluationResult)
    assert 0.0 <= result.precision <= 1.0
    assert 0.0 <= result.recall <= 1.0
