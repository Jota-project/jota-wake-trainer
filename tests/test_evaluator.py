# tests/test_evaluator.py
"""
Tests para trainer/evaluator.py.

Regresión del bug real reportado por el usuario al ejecutar
'wake-trainer evaluate ok_jota':

    TypeError: 'float' object is not iterable

en `max(scores.get(model_key, [0.0]))`. Causa: `openwakeword.Model.predict()`
devuelve un float escalar por modelo (el máximo ya colapsado si le pasas un
clip entero de una vez — ver `Model.predict`, línea
`predictions[mdl] = prediction[0][0][0]`), no una lista de scores por frame.
El `[0.0]` de fallback sí era iterable, así que el bug pasó desapercibido
hasta que hubo una predicción de verdad con la que reventar — el test
original de este fichero (`mock_oww.predict.return_value = {"ok_jota": [0.8,
0.9, ...]}`) reproducía justo esa suposición equivocada y por eso no lo
pilló. El fix usa `model.predict_clip()` (que sí itera frame a frame y
devuelve una lista de dicts) vía `_max_score_over_clip`, con `model.reset()`
antes de cada clip para no arrastrar estado del clip anterior.

Estos tests no cargan openwakeword de verdad — usan un `FakeModel` mínimo
que reproduce el contrato relevante (reset()/predict_clip()) para poder
probar evaluate_model() y _max_score_over_clip() sin red ni un .tflite real.
"""
from __future__ import annotations
from types import SimpleNamespace

import numpy as np

import trainer.evaluator as evaluator_mod
from trainer.evaluator import evaluate_model, _max_score_over_clip, EvaluationResult


class _FakeTfliteInterpreter:
    """Clase con el mismo 'apellido' que tflite_runtime.interpreter.Interpreter,
    para que _detect_runtime_used la reconozca como tflite."""
    pass


class _FakeOnnxInferenceSession:
    """Clase con el mismo 'apellido' que onnxruntime.InferenceSession, para
    que _detect_runtime_used la reconozca como el fallback a onnx."""
    pass


class FakeModel:
    """
    Simula openwakeword.Model lo justo para los tests: predict_clip()
    devuelve una lista de dicts (uno por 'frame'), con un score fijado por
    clip vía `scores_by_clip_len` (indexado por len(data), para poder dar
    distintos scores a distintos clips en el mismo test). `.models` imita el
    diccionario real de openwakeword para que _detect_runtime_used funcione.
    """
    def __init__(self, scores_by_clip_len: dict[int, float], model_key: str, runtime: str = "tflite"):
        self.scores_by_clip_len = scores_by_clip_len
        self.model_key = model_key
        self.reset_calls = 0
        self.predict_clip_calls = []
        loaded = _FakeOnnxInferenceSession() if runtime == "onnx" else _FakeTfliteInterpreter()
        self.models = {model_key: loaded}

    def reset(self):
        self.reset_calls += 1

    def predict_clip(self, data, **kwargs):
        self.predict_clip_calls.append(len(data))
        score = self.scores_by_clip_len.get(len(data), 0.0)
        # Simula varios frames, con el score objetivo apareciendo en uno de ellos.
        return [{self.model_key: 0.0}, {self.model_key: score}, {self.model_key: 0.0}]


def test_evaluation_result_defaults():
    r = EvaluationResult(
        precision=0.94, recall=0.91, false_positives=0, threshold=0.3,
        false_positive_note="0 de 40 clips negativos sintéticos (~35 s)",
    )
    assert r.precision == 0.94
    assert r.false_positives == 0


def test_max_score_over_clip_returns_scalar_not_error():
    """
    Regresión directa del bug: antes de este fix, si `scores.get(model_key)`
    devolvía un float real (no el default [0.0]), `max(float)` lanzaba
    TypeError. Aquí el 'modelo' siempre devuelve un score real.
    """
    model = FakeModel(scores_by_clip_len={100: 0.87}, model_key="ok_jota")
    data = np.zeros(100, dtype=np.int16)

    result = _max_score_over_clip(model, data, "ok_jota")

    assert result == 0.87
    assert model.reset_calls == 1


def test_max_score_over_clip_resets_state_before_each_call():
    model = FakeModel(scores_by_clip_len={}, model_key="ok_jota")
    data = np.zeros(10, dtype=np.int16)

    _max_score_over_clip(model, data, "ok_jota")
    _max_score_over_clip(model, data, "ok_jota")

    assert model.reset_calls == 2


def test_max_score_over_clip_defaults_to_zero_when_key_missing():
    model = FakeModel(scores_by_clip_len={}, model_key="otro_modelo")
    data = np.zeros(10, dtype=np.int16)

    assert _max_score_over_clip(model, data, "ok_jota") == 0.0


def _fake_sf_read(clip_lengths: dict[str, int]):
    def read(path, dtype, always_2d):
        length = clip_lengths[path]
        return np.zeros(length, dtype=np.int16), 16000
    return read


def test_evaluate_model_does_not_raise_and_computes_recall(tmp_path, monkeypatch):
    positivos = tmp_path / "positivos"
    positivos.mkdir()
    clip1 = positivos / "001.wav"
    clip2 = positivos / "002.wav"
    clip1.write_bytes(b"")
    clip2.write_bytes(b"")

    model_path = tmp_path / "ok_jota.tflite"
    model_path.write_bytes(b"fake")

    # clip1 dispara (score alto), clip2 no (score bajo).
    fake_model = FakeModel(
        scores_by_clip_len={16000: 0.9, 8000: 0.1},
        model_key="ok_jota",
    )
    clip_lengths = {str(clip1): 16000, str(clip2): 8000}

    monkeypatch.setattr(
        evaluator_mod, "openwakeword",
        SimpleNamespace(Model=lambda **kwargs: fake_model),
    )
    monkeypatch.setattr(evaluator_mod, "sf", SimpleNamespace(read=_fake_sf_read(clip_lengths)))

    result = evaluate_model(model_path=model_path, positivos_path=positivos, negativos_path=None, threshold=0.3)

    assert isinstance(result, EvaluationResult)
    assert result.recall == 0.5  # 1 de 2 clips positivos por encima del umbral
    assert "silencio sintético" in result.false_positive_note


def test_evaluate_model_uses_negative_clips_when_available(tmp_path, monkeypatch):
    positivos = tmp_path / "positivos"
    positivos.mkdir()
    clip_pos = positivos / "001.wav"
    clip_pos.write_bytes(b"")

    negativos = tmp_path / "negativos"
    negativos.mkdir()
    clip_neg = negativos / "neg_001.wav"
    clip_neg.write_bytes(b"")

    model_path = tmp_path / "ok_jota.tflite"
    model_path.write_bytes(b"fake")

    fake_model = FakeModel(
        scores_by_clip_len={16000: 0.9, 4000: 0.5},  # el negativo también dispara (falso positivo)
        model_key="ok_jota",
    )
    clip_lengths = {str(clip_pos): 16000, str(clip_neg): 4000}

    monkeypatch.setattr(
        evaluator_mod, "openwakeword",
        SimpleNamespace(Model=lambda **kwargs: fake_model),
    )
    monkeypatch.setattr(evaluator_mod, "sf", SimpleNamespace(read=_fake_sf_read(clip_lengths)))

    result = evaluate_model(model_path=model_path, positivos_path=positivos, negativos_path=negativos, threshold=0.3)

    assert result.false_positives == 1
    assert "clips negativos sintéticos" in result.false_positive_note


def test_evaluate_model_reports_threshold_sweep_and_performance(tmp_path, monkeypatch):
    """
    El usuario pidió explícitamente ver sensibilidad a distintos thresholds,
    porcentaje de acierto y rendimiento/velocidad — no solo un único par
    precisión/recall al threshold por defecto.
    """
    positivos = tmp_path / "positivos"
    positivos.mkdir()
    clip_pos = positivos / "001.wav"
    clip_pos.write_bytes(b"")

    negativos = tmp_path / "negativos"
    negativos.mkdir()
    clip_neg = negativos / "neg_001.wav"
    clip_neg.write_bytes(b"")

    model_path = tmp_path / "ok_jota.tflite"
    model_path.write_bytes(b"fake")

    # Positivo con score 0.6: pasa thresholds <=0.6, falla en los más altos.
    # Negativo con score 0.2: solo cuenta como falso positivo en thresholds muy bajos.
    fake_model = FakeModel(
        scores_by_clip_len={16000: 0.6, 4000: 0.2},
        model_key="ok_jota",
        runtime="tflite",
    )
    clip_lengths = {str(clip_pos): 16000, str(clip_neg): 4000}

    monkeypatch.setattr(
        evaluator_mod, "openwakeword",
        SimpleNamespace(Model=lambda **kwargs: fake_model),
    )
    monkeypatch.setattr(evaluator_mod, "sf", SimpleNamespace(read=_fake_sf_read(clip_lengths)))

    result = evaluate_model(
        model_path=model_path, positivos_path=positivos, negativos_path=negativos,
        threshold=0.5, threshold_sweep=(0.1, 0.5, 0.9),
    )

    assert [s.threshold for s in result.threshold_sweep] == [0.1, 0.5, 0.9]

    low = result.threshold_sweep[0]   # 0.1: positivo (0.6) y negativo (0.2) pasan ambos
    assert low.recall == 1.0
    assert low.false_positives == 1

    mid = result.threshold_sweep[1]   # 0.5: solo el positivo (0.6) pasa
    assert mid.recall == 1.0
    assert mid.false_positives == 0

    high = result.threshold_sweep[2]  # 0.9: ni el positivo (0.6) pasa
    assert high.recall == 0.0

    assert result.performance is not None
    assert result.performance.runtime_used == "tflite"
    assert result.performance.total_audio_seconds > 0
    assert result.performance.real_time_factor > 0


def test_evaluate_model_detects_onnx_fallback_runtime(tmp_path, monkeypatch):
    """
    openwakeword.Model puede pedir tflite y caer solo a onnx si falta
    tflite_runtime, sin avisar más que con un logging.warning fácil de
    perder. El informe debe decir la verdad sobre qué runtime se usó.
    """
    positivos = tmp_path / "positivos"
    positivos.mkdir()
    clip_pos = positivos / "001.wav"
    clip_pos.write_bytes(b"")

    model_path = tmp_path / "ok_jota.tflite"
    model_path.write_bytes(b"fake")

    fake_model = FakeModel(scores_by_clip_len={16000: 0.9}, model_key="ok_jota", runtime="onnx")
    clip_lengths = {str(clip_pos): 16000}

    monkeypatch.setattr(
        evaluator_mod, "openwakeword",
        SimpleNamespace(Model=lambda **kwargs: fake_model),
    )
    monkeypatch.setattr(evaluator_mod, "sf", SimpleNamespace(read=_fake_sf_read(clip_lengths)))

    result = evaluate_model(model_path=model_path, positivos_path=positivos, negativos_path=None)

    assert result.performance.runtime_used == "onnx"
