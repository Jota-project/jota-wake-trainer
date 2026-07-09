from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Literal
import json

PROVIDERS_FILE = Path("configs/providers.local.json")


@dataclass
class ProviderConfig:
    name: str
    type: Literal["piper", "openai", "google"]
    url: Optional[str] = None
    token_env: Optional[str] = None
    token_header: Optional[str] = None
    voices: list[str] = field(default_factory=list)
    speeds: list[float] = field(default_factory=lambda: [0.8, 0.9, 1.0, 1.1, 1.2])
    binary: Optional[str] = None
    voices_dir: Optional[str] = None

    def __post_init__(self):
        if not self.speeds:
            self.speeds = [0.8, 0.9, 1.0, 1.1, 1.2]


def load_providers() -> list[ProviderConfig]:
    """Devuelve lista vacía si el fichero no existe."""
    if not PROVIDERS_FILE.exists():
        return []
    data = json.loads(PROVIDERS_FILE.read_text())
    return [ProviderConfig(**p) for p in data.get("providers", [])]


def save_providers(providers: list[ProviderConfig]) -> None:
    """Crea el fichero (y configs/) si no existe."""
    PROVIDERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROVIDERS_FILE.write_text(
        json.dumps({"providers": [asdict(p) for p in providers]}, indent=2, ensure_ascii=False)
    )


def add_or_update_provider(provider: ProviderConfig) -> None:
    """Añade si no existe; actualiza si el nombre ya está."""
    providers = load_providers()
    for i, p in enumerate(providers):
        if p.name == provider.name:
            providers[i] = provider
            save_providers(providers)
            return
    providers.append(provider)
    save_providers(providers)


def remove_provider(name: str) -> bool:
    """Elimina por nombre. Devuelve False si no existe."""
    providers = load_providers()
    new_list = [p for p in providers if p.name != name]
    if len(new_list) == len(providers):
        return False
    save_providers(new_list)
    return True


def get_provider(name: str) -> ProviderConfig | None:
    """Busca por nombre exacto."""
    for p in load_providers():
        if p.name == name:
            return p
    return None
