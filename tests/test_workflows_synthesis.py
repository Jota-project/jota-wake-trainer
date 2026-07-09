# tests/test_workflows_synthesis.py
from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, patch
import trainer.workflows.synthesis as syn_mod
from trainer.providers import ProviderConfig
from trainer.state import Project, SynthesisState, TrainingState, TtsSource


def _make_project(sources=None):
    return Project(
        wake_word="ok jota",
        model_name="ok_jota",
        created_at="2026-01-01T00:00:00+00:00",
        voices=[],
        synthesis=SynthesisState(sources=sources or []),
        training=TrainingState(),
    )


def test_run_synthesize_step_skips_configure_when_sources_exist(monkeypatch):
    project = _make_project(sources=[TtsSource(type="openai", url="http://x", selected_voices=["a"])])
    monkeypatch.setattr(syn_mod, "load_project", lambda name: project)
    monkeypatch.setattr(syn_mod, "synthesize_project", lambda p: None)

    called = {"configure": False}
    monkeypatch.setattr(syn_mod, "configure_synthesis", lambda p: called.__setitem__("configure", True))

    syn_mod.run_synthesize_step("ok_jota")

    assert called["configure"] is False


def test_run_synthesize_step_calls_configure_when_no_sources(monkeypatch):
    project = _make_project(sources=[])
    monkeypatch.setattr(syn_mod, "load_project", lambda name: project)
    monkeypatch.setattr(syn_mod, "synthesize_project", lambda p: None)

    called = {"configure": False}
    monkeypatch.setattr(syn_mod, "configure_synthesis", lambda p: called.__setitem__("configure", True))

    syn_mod.run_synthesize_step("ok_jota")

    assert called["configure"] is True


def test_run_synthesize_step_add_provider_forces_configure_even_with_sources(monkeypatch):
    project = _make_project(sources=[TtsSource(type="openai", url="http://x", selected_voices=["a"])])
    monkeypatch.setattr(syn_mod, "load_project", lambda name: project)
    monkeypatch.setattr(syn_mod, "synthesize_project", lambda p: None)

    called = {"configure": False}
    monkeypatch.setattr(syn_mod, "configure_synthesis", lambda p: called.__setitem__("configure", True))

    syn_mod.run_synthesize_step("ok_jota", add_provider=True)

    assert called["configure"] is True


def test_provider_to_tts_source_uses_existing_voices(monkeypatch):
    monkeypatch.setattr(syn_mod, "ask", lambda *a, **kw: "0.9,1.0")

    provider = ProviderConfig(
        name="test", type="openai", url="http://x", voices=["Alice", "Bob"]
    )
    result = syn_mod._provider_to_tts_source(provider)

    assert result is not None
    assert result.selected_voices == ["Alice", "Bob"]
    assert result.speeds == [0.9, 1.0]


def test_provider_to_tts_source_returns_none_when_openai_no_voices(monkeypatch):
    monkeypatch.setattr(syn_mod, "ask", lambda *a, **kw: "")

    with patch("trainer.synthesizer.list_voices_openai", new=AsyncMock(return_value=[])):
        provider = ProviderConfig(
            name="test", type="openai", url="http://x", voices=[]
        )
        result = syn_mod._provider_to_tts_source(provider)

    assert result is None


def test_provider_to_tts_source_returns_none_when_piper_no_models(monkeypatch):
    monkeypatch.setattr(syn_mod, "ask", lambda *a, **kw: "1.0")

    with patch("trainer.synthesizer.list_voices_piper", return_value=[]):
        provider = ProviderConfig(
            name="piper_local", type="piper", voices_dir="piper/voices", voices=[]
        )
        result = syn_mod._provider_to_tts_source(provider)

    assert result is None
