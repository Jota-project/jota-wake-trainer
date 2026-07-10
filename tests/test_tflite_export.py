# tests/test_tflite_export.py
"""
Tests para trainer/tflite_export.py, en particular
`convert_onnx_to_tflite_in_subprocess` — la función que aísla la conversión
ONNX->TFLite en un proceso `python` nuevo para evitar el hang real observado
en producción: llamar a `convert_onnx_to_tflite` en el mismo proceso que ya
había usado `torch` para entrenar se queda colgado sin responder ni a
Ctrl+C, mientras que en un proceso nuevo (sin torch cargado) termina en
segundos. Estos tests no lanzan un subproceso real de verdad (serían lentos
y dependerían de tensorflow/onnx2tf instalados) — monkeypatchean
`subprocess.run` para comprobar el manejo de cada resultado posible.
"""
from __future__ import annotations
import subprocess

import pytest

from trainer.tflite_export import (
    convert_onnx_to_tflite_in_subprocess,
    TFLiteExportError,
)


def test_raises_without_spawning_subprocess_if_onnx_missing(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: called.append(1))

    with pytest.raises(TFLiteExportError, match="No existe el modelo ONNX"):
        convert_onnx_to_tflite_in_subprocess(tmp_path / "no_existe.onnx", tmp_path / "out.tflite")

    assert called == []


def test_success_returns_tflite_path(tmp_path, monkeypatch):
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"fake-onnx")
    tflite_path = tmp_path / "model.tflite"

    def fake_run(cmd, capture_output, text, timeout):
        # Simula que el subproceso hizo su trabajo y escribió el .tflite.
        tflite_path.write_bytes(b"fake-tflite")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = convert_onnx_to_tflite_in_subprocess(onnx_path, tflite_path)
    assert result == tflite_path
    assert tflite_path.exists()


def test_timeout_raises_tflite_export_error_mentioning_isolation(tmp_path, monkeypatch):
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"fake-onnx")

    def fake_run(cmd, capture_output, text, timeout):
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(TFLiteExportError, match="proceso aislado"):
        convert_onnx_to_tflite_in_subprocess(onnx_path, tmp_path / "out.tflite", timeout=1.0)


def test_subprocess_failure_with_tflite_export_error_marker_extracts_message(tmp_path, monkeypatch):
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"fake-onnx")

    def fake_run(cmd, capture_output, text, timeout):
        return subprocess.CompletedProcess(
            cmd, returncode=2, stdout="",
            stderr="algo de ruido\nTFLITE_EXPORT_ERROR: faltan dependencias de verdad\n",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(TFLiteExportError, match="faltan dependencias de verdad"):
        convert_onnx_to_tflite_in_subprocess(onnx_path, tmp_path / "out.tflite")


def test_subprocess_generic_failure_includes_stderr_tail(tmp_path, monkeypatch):
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"fake-onnx")

    def fake_run(cmd, capture_output, text, timeout):
        return subprocess.CompletedProcess(
            cmd, returncode=1, stdout="", stderr="Traceback...\nRuntimeError: boom\n",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(TFLiteExportError, match="falló en el subproceso"):
        convert_onnx_to_tflite_in_subprocess(onnx_path, tmp_path / "out.tflite")


def test_success_but_missing_output_file_raises(tmp_path, monkeypatch):
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"fake-onnx")

    def fake_run(cmd, capture_output, text, timeout):
        # Código 0 pero, por lo que sea, el .tflite no aparece.
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(TFLiteExportError, match="no generó"):
        convert_onnx_to_tflite_in_subprocess(onnx_path, tmp_path / "out.tflite")


# ── Regresión real: ejes de entrada invertidos por onnx2tf ─────────────────
#
# Bug confirmado en producción (openWakeWord vía Wyoming/Home Assistant, ver
# commit que introduce este test): onnx2tf trata por defecto cualquier
# input 3D como si siguiera la convención de imagen NCW de ONNX y lo
# reordena a NWC — para el input de openWakeWord, (batch, ventanas_tiempo,
# features) = (1, 16, 96), eso produce un .tflite con (1, 96, 16). El
# consumidor calcula el nº de ventanas del eje equivocado, nunca llega al
# mínimo esperado, y el hilo de detección se cuelga para siempre sin
# procesar nada — sin ningún error visible. Afectaba a TODOS los modelos
# generados por este repo hasta ahora.
#
# Estos tests requieren onnx2tf/tensorflow de verdad (extra "tflite") — se
# omiten limpiamente si no están instalados, igual que test_recorder.py con
# sounddevice (ver tests/conftest.py). El CI actual no instala el extra
# "tflite" (es pesado — tensorflow completo), así que ahí se omiten; se
# ejecutan en local con 'pip install -e ".[tflite]"'.
onnx = pytest.importorskip("onnx", reason="requiere el extra 'tflite'")
pytest.importorskip("onnx2tf", reason="requiere el extra 'tflite'")
tf = pytest.importorskip("tensorflow", reason="requiere el extra 'tflite'")

from trainer.tflite_export import convert_onnx_to_tflite  # noqa: E402


def _build_fake_openwakeword_onnx(path, input_name="onnx::Flatten_0"):
    """
    Construye un .onnx mínimo con la misma forma de input que el modelo DNN
    real de openWakeWord — (1, 16, 96): 16 ventanas de 96 features de
    embeddings — y una cadena Flatten + MatMul + Sigmoid equivalente a la
    capa final real, lo bastante fiel como para disparar el mismo camino de
    inferencia de ejes en onnx2tf que el modelo de verdad.
    """
    from onnx import helper, TensorProto, numpy_helper
    import numpy as np

    n_windows, n_features = 16, 96
    flat_dim = n_windows * n_features

    x = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, n_windows, n_features])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])

    weight = numpy_helper.from_array(
        np.random.randn(flat_dim, 1).astype(np.float32), name="weight"
    )
    flatten_node = helper.make_node("Flatten", [input_name], ["flat"], axis=1)
    matmul_node = helper.make_node("MatMul", ["flat", "weight"], ["logits"])
    sigmoid_node = helper.make_node("Sigmoid", ["logits"], ["output"])

    graph = helper.make_graph(
        [flatten_node, matmul_node, sigmoid_node],
        "fake_openwakeword",
        [x], [y], initializer=[weight],
    )
    model = helper.make_model(graph, producer_name="test")
    model.opset_import[0].version = 13
    onnx.save(model, str(path))


def test_convert_onnx_to_tflite_preserves_input_axis_order(tmp_path):
    onnx_path = tmp_path / "model.onnx"
    tflite_path = tmp_path / "model.tflite"
    _build_fake_openwakeword_onnx(onnx_path)

    result = convert_onnx_to_tflite(onnx_path, tflite_path)

    assert result == tflite_path
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    actual_shape = tuple(interpreter.get_input_details()[0]["shape"].tolist())
    assert actual_shape == (1, 16, 96), (
        f"El .tflite tiene el input con forma {actual_shape} — se esperaba (1, 16, 96) "
        "(mismo orden que el .onnx de origen). Si esto falla, es EXACTAMENTE el bug de "
        "producción que este test existe para detectar: onnx2tf ha vuelto a invertir "
        "los ejes de entrada."
    )


def test_convert_onnx_to_tflite_raises_clearly_if_axis_order_cannot_be_fixed(tmp_path, monkeypatch):
    """
    Si ninguna variante de nombre candidata logra preservar el shape (por
    ejemplo, porque onnx2tf cambió de comportamiento en una versión nueva),
    la función debe fallar con un TFLiteExportError explicativo — nunca
    copiar en silencio un .tflite con los ejes equivocados a su destino.
    """
    import trainer.tflite_export as mod

    onnx_path = tmp_path / "model.onnx"
    tflite_path = tmp_path / "model.tflite"
    _build_fake_openwakeword_onnx(onnx_path)

    # Fuerza que solo se intente el nombre crudo del tensor (sin las
    # variantes saneadas) — ya sabemos que ese candidato por sí solo no
    # preserva el shape correcto, así que esto ejercita de verdad la ruta
    # de "agotados los candidatos, ninguno ha funcionado".
    monkeypatch.setattr(mod, "_name_candidates", lambda name: [name])

    with pytest.raises(TFLiteExportError, match="No se pudo generar un .tflite"):
        convert_onnx_to_tflite(onnx_path, tflite_path)

    assert not tflite_path.exists()
