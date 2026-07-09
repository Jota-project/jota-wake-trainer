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
