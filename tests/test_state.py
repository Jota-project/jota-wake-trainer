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


def test_create_project_raises_if_exists(tmp_projects):
    create_project("ok jota", "ok_jota", ["Alfonso"])
    with pytest.raises(FileExistsError):
        create_project("ok jota diferente", "ok_jota", ["María"])


from trainer.state import calculate_dataset


def test_calculate_dataset_no_synthesis(tmp_projects):
    p = create_project("ok jota", "ok_jota", ["Alfonso", "María"])
    stats = calculate_dataset(p)
    assert stats["real_clips"] == 60        # 2 personas × 30
    assert stats["synth_clips"] == 0
    assert stats["real_augmented"] == 600   # 60 × 10
    assert stats["total"] == 600
    assert stats["meets_minimum"] is False  # < 1000


def test_calculate_dataset_with_piper_source(tmp_projects):
    p = create_project("ok jota", "ok_jota", ["Alfonso", "María", "Carlos"])
    p.synthesis.sources.append(TtsSource(
        type="piper",
        selected_voices=["voz_a", "voz_b", "voz_c", "voz_d", "voz_e", "voz_f"],
        speeds=[0.8, 0.9, 1.0, 1.1, 1.2],
    ))
    stats = calculate_dataset(p)
    assert stats["real_clips"] == 90        # 3 × 30
    assert stats["synth_clips"] == 30       # 6 voces × 5 velocidades
    assert stats["total"] == 1200           # (90 + 30) × 10
    assert stats["meets_minimum"] is True


def test_calculate_dataset_multiple_sources(tmp_projects):
    p = create_project("ok jota", "ok_jota", ["Alfonso"])
    p.synthesis.sources = [
        TtsSource(type="piper", selected_voices=["a", "b"], speeds=[1.0]),
        TtsSource(type="openai", selected_voices=["Rachel", "Josh"], speeds=[0.9, 1.0, 1.1]),
    ]
    stats = calculate_dataset(p)
    # real: 1 × 30 = 30
    # synth: piper(2×1=2) + openai(2×3=6) = 8
    # total: (30+8) × 10 = 380
    assert stats["synth_clips"] == 8
    assert stats["total"] == 380
