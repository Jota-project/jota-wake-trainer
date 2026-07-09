# tests/test_ui_voice_selection.py
from __future__ import annotations
import pytest
from trainer.ui import voice_selection as vs_mod

VOICES_RAW = [
    {"name": "Alice", "voice_id": "alice_id"},
    {"name": "Bob",   "voice_id": "bob_id"},
    {"name": "Carol", "voice_id": "carol_id"},
]
PIPER_VOICES = ["piper/voices/es_ES.onnx", "piper/voices/en_US.onnx"]


def test_select_openai_todas(monkeypatch):
    monkeypatch.setattr(vs_mod, "ask", lambda *a, **kw: "todas")
    assert vs_mod.select_openai_voices(VOICES_RAW) == ["alice_id", "bob_id", "carol_id"]


def test_select_openai_indices(monkeypatch):
    monkeypatch.setattr(vs_mod, "ask", lambda *a, **kw: "1,3")
    assert vs_mod.select_openai_voices(VOICES_RAW) == ["alice_id", "carol_id"]


def test_select_openai_invalid_fallback(monkeypatch):
    monkeypatch.setattr(vs_mod, "ask", lambda *a, **kw: "no_es_numero")
    assert vs_mod.select_openai_voices(VOICES_RAW) == ["alice_id", "bob_id", "carol_id"]


def test_select_piper_todas(monkeypatch):
    monkeypatch.setattr(vs_mod, "ask", lambda *a, **kw: "todas")
    assert vs_mod.select_piper_voices(PIPER_VOICES) == PIPER_VOICES


def test_select_piper_indices(monkeypatch):
    monkeypatch.setattr(vs_mod, "ask", lambda *a, **kw: "2")
    assert vs_mod.select_piper_voices(PIPER_VOICES) == ["piper/voices/en_US.onnx"]
