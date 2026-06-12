import pytest
from typer.testing import CliRunner
import trainer.providers as providers_mod
from trainer.providers import ProviderConfig, load_providers, add_or_update_provider
from trainer.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def tmp_providers(tmp_path, monkeypatch):
    monkeypatch.setattr(providers_mod, "PROVIDERS_FILE", tmp_path / "providers.json")


def test_providers_list_empty():
    result = runner.invoke(app, ["providers", "list"])
    assert result.exit_code == 0
    assert "providers add" in result.output


def test_providers_list_shows_configured():
    add_or_update_provider(
        ProviderConfig(name="jspeaker", type="openai", url="http://localhost:5500", voices=["Kaia"])
    )
    result = runner.invoke(app, ["providers", "list"])
    assert result.exit_code == 0
    assert "jspeaker" in result.output


def test_providers_add_with_all_flags():
    result = runner.invoke(app, [
        "providers", "add",
        "--name", "jspeaker",
        "--type", "openai",
        "--url", "http://localhost:5500",
        "--voice", "Kaia",
        "--speed", "1.0",
    ])
    assert result.exit_code == 0
    providers = load_providers()
    assert len(providers) == 1
    assert providers[0].name == "jspeaker"
    assert providers[0].voices == ["Kaia"]
    assert providers[0].speeds == [1.0]


def test_providers_add_with_token_env():
    result = runner.invoke(app, [
        "providers", "add",
        "--name", "elevenlabs",
        "--type", "openai",
        "--url", "https://api.elevenlabs.io/v1",
        "--token-env", "ELEVENLABS_API_KEY",
    ])
    assert result.exit_code == 0
    p = load_providers()[0]
    assert p.token_env == "ELEVENLABS_API_KEY"
    assert p.voices == []


def test_providers_remove_existing():
    add_or_update_provider(
        ProviderConfig(name="jspeaker", type="openai", url="http://localhost:5500")
    )
    result = runner.invoke(app, ["providers", "remove", "jspeaker"])
    assert result.exit_code == 0
    assert load_providers() == []


def test_providers_remove_nonexistent():
    result = runner.invoke(app, ["providers", "remove", "nonexistent"])
    assert result.exit_code == 1


def test_providers_add_invalid_type():
    result = runner.invoke(app, [
        "providers", "add",
        "--name", "test",
        "--type", "invalido",
        "--url", "http://example.com",
    ])
    assert result.exit_code == 1
