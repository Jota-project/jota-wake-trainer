import pytest
from pathlib import Path
import trainer.providers as providers_mod
from trainer.providers import (
    ProviderConfig, load_providers, save_providers,
    add_or_update_provider, remove_provider, get_provider,
)


@pytest.fixture(autouse=True)
def tmp_providers_file(tmp_path, monkeypatch):
    monkeypatch.setattr(providers_mod, "PROVIDERS_FILE", tmp_path / "providers.json")


def test_load_providers_returns_empty_when_file_missing():
    assert load_providers() == []


def test_add_provider_creates_file():
    p = ProviderConfig(name="jspeaker", type="openai", url="http://localhost:5500", voices=["Kaia"], speeds=[1.0])
    add_or_update_provider(p)
    loaded = load_providers()
    assert len(loaded) == 1
    assert loaded[0].name == "jspeaker"


def test_add_or_update_does_not_duplicate():
    p = ProviderConfig(name="jspeaker", type="openai", url="http://localhost:5500")
    add_or_update_provider(p)
    p2 = ProviderConfig(name="jspeaker", type="openai", url="http://localhost:9999")
    add_or_update_provider(p2)
    loaded = load_providers()
    assert len(loaded) == 1
    assert loaded[0].url == "http://localhost:9999"


def test_remove_provider_returns_true_when_found():
    p = ProviderConfig(name="elevenlabs", type="openai", url="https://api.elevenlabs.io/v1")
    add_or_update_provider(p)
    assert remove_provider("elevenlabs") is True
    assert load_providers() == []


def test_remove_provider_returns_false_when_not_found():
    assert remove_provider("nonexistent") is False


def test_get_provider_returns_correct():
    p = ProviderConfig(name="jspeaker", type="openai", url="http://localhost:5500")
    add_or_update_provider(p)
    found = get_provider("jspeaker")
    assert found is not None
    assert found.url == "http://localhost:5500"


def test_get_provider_returns_none_when_missing():
    assert get_provider("missing") is None


def test_roundtrip_preserves_all_fields():
    p = ProviderConfig(
        name="elevenlabs",
        type="openai",
        url="https://api.elevenlabs.io/v1",
        token_env="ELEVENLABS_API_KEY",
        voices=["Rachel", "Josh"],
        speeds=[0.8, 1.0, 1.2],
    )
    add_or_update_provider(p)
    loaded = get_provider("elevenlabs")
    assert loaded.token_env == "ELEVENLABS_API_KEY"
    assert loaded.voices == ["Rachel", "Josh"]
    assert loaded.speeds == [0.8, 1.0, 1.2]
