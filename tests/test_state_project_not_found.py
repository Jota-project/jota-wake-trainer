# tests/test_state_project_not_found.py
"""
Tests para trainer/state.py::load_project cuando el proyecto no existe.

Regresión del bug real: pasar una ruta ('projects/ok_jota/models/ok_jota.tflite')
en vez del nombre del proyecto ('ok_jota') hacía que Path.read_text() lanzara
un FileNotFoundError crudo con un path sin sentido
('projects/projects/ok_jota/models/ok_jota.tflite/session.json'), que se
propagaba como traceback completo hasta la terminal. Ahora load_project
detecta el caso y lanza ProjectNotFoundError con un mensaje ya pensado para
mostrarse tal cual al usuario (ver trainer/cli.py::_friendly_errors).
"""
from __future__ import annotations
import pytest

from trainer.state import (
    load_project,
    ProjectNotFoundError,
    create_project,
    available_project_names,
)


def test_load_project_missing_raises_project_not_found_error(tmp_path, monkeypatch):
    import trainer.state as state_mod
    monkeypatch.setattr(state_mod, "PROJECTS_ROOT", tmp_path)

    with pytest.raises(ProjectNotFoundError, match="No se encontró el proyecto 'no_existe'"):
        load_project("no_existe")


def test_message_lists_available_projects_when_any_exist(tmp_path, monkeypatch):
    import trainer.state as state_mod
    monkeypatch.setattr(state_mod, "PROJECTS_ROOT", tmp_path)
    create_project("ok jota", "ok_jota", ["Sito"])

    with pytest.raises(ProjectNotFoundError, match="Proyectos existentes: ok_jota"):
        load_project("otro_nombre")


def test_message_says_no_projects_when_none_exist(tmp_path, monkeypatch):
    import trainer.state as state_mod
    monkeypatch.setattr(state_mod, "PROJECTS_ROOT", tmp_path)

    with pytest.raises(ProjectNotFoundError, match="No hay ningún proyecto creado todavía"):
        load_project("cualquier_cosa")


def test_message_detects_path_passed_instead_of_name_and_guesses_intent(tmp_path, monkeypatch):
    import trainer.state as state_mod
    monkeypatch.setattr(state_mod, "PROJECTS_ROOT", tmp_path)
    create_project("ok jota", "ok_jota", ["Sito"])

    with pytest.raises(ProjectNotFoundError) as exc_info:
        load_project("projects/ok_jota/models/ok_jota.tflite")

    msg = str(exc_info.value)
    assert "espera solo el NOMBRE del proyecto" in msg
    assert "¿Quisiste decir 'ok_jota'?" in msg


def test_message_does_not_suggest_guess_for_plain_wrong_name(tmp_path, monkeypatch):
    import trainer.state as state_mod
    monkeypatch.setattr(state_mod, "PROJECTS_ROOT", tmp_path)
    create_project("ok jota", "ok_jota", ["Sito"])

    with pytest.raises(ProjectNotFoundError) as exc_info:
        load_project("okjota_typo")

    msg = str(exc_info.value)
    assert "espera solo el NOMBRE" not in msg
    assert "Proyectos existentes: ok_jota" in msg


def test_available_project_names_ignores_dirs_without_session_json(tmp_path, monkeypatch):
    import trainer.state as state_mod
    monkeypatch.setattr(state_mod, "PROJECTS_ROOT", tmp_path)
    create_project("ok jota", "ok_jota", ["Sito"])
    (tmp_path / "carpeta_suelta").mkdir()

    assert available_project_names() == ["ok_jota"]
