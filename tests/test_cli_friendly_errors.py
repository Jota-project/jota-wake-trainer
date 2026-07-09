# tests/test_cli_friendly_errors.py
"""
Verifica que pasar un model_name inexistente (en particular una ruta, el
error real que reportó el usuario al hacer 'wake-trainer evaluate
projects/ok_jota/models/ok_jota.tflite') da un mensaje corto en pantalla en
vez de un traceback de varias pantallas — para varios comandos, no solo el
que se reportó, ya que todos comparten el mismo fallo de fondo
(load_project con un nombre inválido).
"""
from __future__ import annotations
import pytest
from typer.testing import CliRunner

from trainer.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate_projects_root(tmp_path, monkeypatch):
    import trainer.state as state_mod
    monkeypatch.setattr(state_mod, "PROJECTS_ROOT", tmp_path)
    yield


@pytest.mark.parametrize("command", ["status", "record", "synthesize", "train", "evaluate", "convert"])
def test_missing_project_path_argument_does_not_show_traceback(command):
    result = runner.invoke(app, [command, "projects/ok_jota/models/ok_jota.tflite"])

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "No se encontró el proyecto" in result.output
    assert "¿Quisiste decir" in result.output or "espera solo el NOMBRE" in result.output


def test_evaluate_missing_project_matches_reported_bug_exactly():
    result = runner.invoke(app, ["evaluate", "projects/ok_jota/models/ok_jota.tflite"])

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "FileNotFoundError" not in result.output
    assert "No se encontró el proyecto" in result.output
