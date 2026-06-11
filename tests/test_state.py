import pytest
from pathlib import Path
from datetime import datetime
import trainer.state as state_mod
from trainer.state import (
    Project, Voice, SynthesisState, TrainingState, TtsSource,
    create_project, load_project, save_project, list_projects,
)


@pytest.fixture(autouse=True)
def tmp_projects(tmp_path, monkeypatch):
    monkeypatch.setattr(state_mod, "PROJECTS_ROOT", tmp_path / "projects")
    return tmp_path / "projects"


def test_create_project_creates_directories(tmp_projects):
    p = create_project("ok jota", "ok_jota", ["Alfonso", "María"])
    assert (tmp_projects / "ok_jota" / "data" / "positivos" / "Alfonso").exists()
    assert (tmp_projects / "ok_jota" / "data" / "positivos" / "María").exists()
    assert (tmp_projects / "ok_jota" / "data" / "sintetizados").exists()
    assert (tmp_projects / "ok_jota" / "models").exists()


def test_create_project_saves_session(tmp_projects):
    create_project("ok jota", "ok_jota", ["Alfonso"])
    assert (tmp_projects / "ok_jota" / "session.json").exists()


def test_load_project_roundtrip(tmp_projects):
    create_project("hey asistente", "hey_asistente", ["Ana", "Luis"])
    loaded = load_project("hey_asistente")
    assert loaded.wake_word == "hey asistente"
    assert len(loaded.voices) == 2
    assert loaded.voices[0].name == "Ana"
    assert loaded.voices[0].status == "pending"


def test_save_project_persists_voice_state(tmp_projects):
    p = create_project("ok jota", "ok_jota", ["Alfonso"])
    p.voices[0].status = "done"
    p.voices[0].clips = 30
    save_project(p)
    loaded = load_project("ok_jota")
    assert loaded.voices[0].status == "done"
    assert loaded.voices[0].clips == 30


def test_list_projects_empty(tmp_projects):
    assert list_projects() == []


def test_list_projects_returns_all(tmp_projects):
    create_project("ok jota", "ok_jota", ["Alfonso"])
    create_project("hey bot", "hey_bot", ["María"])
    names = [p.model_name for p in list_projects()]
    assert "ok_jota" in names
    assert "hey_bot" in names


def test_project_paths(tmp_projects):
    p = create_project("ok jota", "ok_jota", ["Alfonso"])
    assert p.root == tmp_projects / "ok_jota"
    assert p.positivos_path == tmp_projects / "ok_jota" / "data" / "positivos"
    assert p.sintetizados_path == tmp_projects / "ok_jota" / "data" / "sintetizados"
    assert p.models_path == tmp_projects / "ok_jota" / "models"
