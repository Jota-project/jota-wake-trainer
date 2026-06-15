# tests/test_workflows_training.py
from __future__ import annotations
from pathlib import Path
import pytest
import trainer.trainer_core
import trainer.workflows.training as train_mod
from trainer.state import Project, Voice, SynthesisState, TrainingState


def _make_project(model_path: str | None = None):
    return Project(
        wake_word="ok jota",
        model_name="ok_jota",
        created_at="2026-01-01T00:00:00+00:00",
        voices=[Voice(name="Alfonso")],
        synthesis=SynthesisState(),
        training=TrainingState(),
        model_path=model_path,
    )


def test_train_project_aborts_when_n(monkeypatch):
    monkeypatch.setattr(train_mod, "calculate_dataset", lambda p: {"total": 500})
    monkeypatch.setattr(train_mod, "ask_choice", lambda *a, **kw: "n")
    monkeypatch.setattr(train_mod, "save_project", lambda p: None)

    project = _make_project()
    train_mod.train_project(project)

    assert project.training.status == "pending"


def test_train_project_sets_done_on_success(monkeypatch):
    monkeypatch.setattr(train_mod, "calculate_dataset", lambda p: {"total": 500})
    monkeypatch.setattr(train_mod, "ask_choice", lambda *a, **kw: "s")
    monkeypatch.setattr(train_mod, "save_project", lambda p: None)
    monkeypatch.setattr(
        trainer.trainer_core, "run_training",
        lambda path, cfg: Path("models/ok_jota.tflite"),
    )

    project = _make_project()
    train_mod.train_project(project)

    assert project.training.status == "done"
    assert project.model_path == "models/ok_jota.tflite"


def test_train_project_sets_pending_on_failure(monkeypatch):
    monkeypatch.setattr(train_mod, "calculate_dataset", lambda p: {"total": 500})
    monkeypatch.setattr(train_mod, "ask_choice", lambda *a, **kw: "s")
    monkeypatch.setattr(train_mod, "save_project", lambda p: None)
    monkeypatch.setattr(
        trainer.trainer_core, "run_training",
        lambda path, cfg: (_ for _ in ()).throw(RuntimeError("sin GPU")),
    )

    project = _make_project()
    train_mod.train_project(project)

    assert project.training.status == "pending"


def test_evaluate_project_skips_when_no_model_path(monkeypatch):
    monkeypatch.setattr(train_mod, "save_project", lambda p: None)

    project = _make_project(model_path=None)
    train_mod.evaluate_project(project)  # no debe lanzar excepción
