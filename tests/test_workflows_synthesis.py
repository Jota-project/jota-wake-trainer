# tests/test_workflows_synthesis.py
from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, patch
import trainer.workflows.synthesis as syn_mod
from trainer.providers import ProviderConfig


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
