# tests/test_workflows_recording.py
from __future__ import annotations
import pytest
import trainer.importer
import trainer.recorder
import trainer.workflows.recording as rec_mod
from trainer.state import Project, Voice, SynthesisState, TrainingState


def _make_project():
    return Project(
        wake_word="ok jota",
        model_name="ok_jota",
        created_at="2026-01-01T00:00:00+00:00",
        voices=[Voice(name="Alfonso")],
        synthesis=SynthesisState(),
        training=TrainingState(),
    )


def test_record_or_import_skips_when_d(monkeypatch):
    monkeypatch.setattr(rec_mod, "explain", lambda *a, **kw: None)
    monkeypatch.setattr(rec_mod, "ask_choice", lambda *a, **kw: "d")
    monkeypatch.setattr(rec_mod, "save_project", lambda p: None)

    project = _make_project()
    voice = project.voices[0]
    rec_mod.record_or_import_voice(project, voice)

    assert voice.status == "pending"
    assert voice.mode is None


def test_record_or_import_import_updates_voice(monkeypatch, tmp_path):
    monkeypatch.setattr(rec_mod, "explain", lambda *a, **kw: None)
    monkeypatch.setattr(rec_mod, "ask_choice", lambda *a, **kw: "i")
    monkeypatch.setattr(rec_mod, "ask", lambda *a, **kw: str(tmp_path))
    monkeypatch.setattr(rec_mod, "save_project", lambda p: None)
    monkeypatch.setattr(trainer.importer, "import_clips", lambda src, dst: (35, []))

    project = _make_project()
    voice = project.voices[0]
    rec_mod.record_or_import_voice(project, voice)

    assert voice.mode == "import"
    assert voice.clips == 35
    assert voice.status == "done"


def test_record_or_import_record_updates_voice(monkeypatch):
    monkeypatch.setattr(rec_mod, "explain", lambda *a, **kw: None)
    monkeypatch.setattr(rec_mod, "ask_choice", lambda *a, **kw: "g")
    monkeypatch.setattr(rec_mod, "save_project", lambda p: None)
    monkeypatch.setattr(trainer.recorder, "record_voice", lambda path, word: 30)

    project = _make_project()
    voice = project.voices[0]
    rec_mod.record_or_import_voice(project, voice)

    assert voice.mode == "record"
    assert voice.clips == 30
    assert voice.status == "done"


def test_record_or_import_partial_import_sets_done(monkeypatch, tmp_path):
    monkeypatch.setattr(rec_mod, "explain", lambda *a, **kw: None)
    monkeypatch.setattr(rec_mod, "ask_choice", lambda *a, **kw: "i")
    monkeypatch.setattr(rec_mod, "ask", lambda *a, **kw: str(tmp_path))
    monkeypatch.setattr(rec_mod, "save_project", lambda p: None)
    monkeypatch.setattr(trainer.importer, "import_clips", lambda src, dst: (5, ["bad.mp3"]))

    project = _make_project()
    voice = project.voices[0]
    rec_mod.record_or_import_voice(project, voice)

    assert voice.clips == 5
    assert voice.status == "done"
