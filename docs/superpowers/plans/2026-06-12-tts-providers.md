# TTS Providers — Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Añadir preconfiguración global de providers TTS (ElevenLabs, Jspeaker u otros) guardada en `configs/providers.local.json`, con subcomando CLI `wake-trainer providers` y un wizard interactivo que los detecta automáticamente al configurar síntesis.

**Architecture:** Nuevo módulo `trainer/providers.py` con `ProviderConfig` (dataclass) y CRUD sobre `configs/providers.local.json` (ignorado por git). El wizard principal `_wizard_configure_synthesis` en `trainer/wizard.py` consulta los providers globales y los convierte a `TtsSource` por proyecto. El CLI añade un grupo `providers` con subcomandos `list`, `add`, `remove`.

**Tech Stack:** Python 3.11+, Typer (CLI), Rich (UI), dataclasses + json (persistencia), pytest + typer.testing.CliRunner (tests).

---

## Mapa de archivos

**Crear:**
```
trainer/providers.py
tests/test_providers.py
tests/test_providers_cli.py
```

**Modificar:**
- `trainer/cli.py` — añadir `providers_app` Typer sub-group con subcomandos `list`, `add`, `remove`
- `trainer/wizard.py` — reemplazar `_wizard_configure_synthesis`; añadir `_wizard_add_provider` y `_provider_to_tts_source`
- `.gitignore` — añadir `configs/providers.local.json`

---

## Dependencias entre tasks

```
Task 1 (providers.py)
  └─→ Task 2 (CLI)
  └─→ Task 3 (wizard)
        └─→ Task 4 (gitignore + verificación final)
```

---

## Task 1: providers.py — CRUD de providers globales

**Files:**
- Create: `trainer/providers.py`
- Create: `tests/test_providers.py`

- [ ] **Step 1: Escribir tests**

```python
# tests/test_providers.py
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
```

- [ ] **Step 2: Verificar que los tests fallan**

```bash
pytest tests/test_providers.py -v
```

Esperado: `ImportError` — `trainer/providers.py` no existe.

- [ ] **Step 3: Implementar `trainer/providers.py`**

```python
# trainer/providers.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Literal
import json

PROVIDERS_FILE = Path("configs/providers.local.json")


@dataclass
class ProviderConfig:
    name: str
    type: Literal["piper", "openai"]
    url: Optional[str] = None
    token_env: Optional[str] = None
    voices: list[str] = field(default_factory=list)
    speeds: list[float] = field(default_factory=lambda: [0.8, 0.9, 1.0, 1.1, 1.2])
    binary: Optional[str] = None
    voices_dir: Optional[str] = None


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
```

- [ ] **Step 4: Ejecutar tests**

```bash
pytest tests/test_providers.py -v
```

Esperado: 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add trainer/providers.py tests/test_providers.py
git commit -m "feat: providers — CRUD de providers TTS globales en configs/providers.local.json"
```

---

## Task 2: CLI — subcomando `providers`

**Files:**
- Create: `tests/test_providers_cli.py`
- Modify: `trainer/cli.py`

- [ ] **Step 1: Escribir tests**

```python
# tests/test_providers_cli.py
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
```

- [ ] **Step 2: Verificar que los tests fallan**

```bash
pytest tests/test_providers_cli.py -v
```

Esperado: error porque el subcomando `providers` no existe todavía.

- [ ] **Step 3: Añadir el grupo `providers` a `trainer/cli.py`**

Añadir justo después de la línea `console = Console()` (línea 13), antes del `@app.callback`:

```python
# ─── Subgrupo providers ────────────────────────────────────────────────────────

providers_app = typer.Typer(
    name="providers",
    help="Gestiona providers TTS globales preconfigurados.",
    add_completion=False,
)
app.add_typer(providers_app)


@providers_app.command("list")
def providers_list():
    """Lista todos los providers TTS configurados."""
    from trainer.providers import load_providers
    from rich.table import Table
    from rich import box as rich_box

    providers = load_providers()
    if not providers:
        console.print(
            "[dim]No hay providers configurados. "
            "Usa [bold]wake-trainer providers add[/bold] para añadir uno.[/dim]"
        )
        return

    table = Table(box=rich_box.SIMPLE_HEAD)
    table.add_column("Nombre", style="bold")
    table.add_column("Tipo")
    table.add_column("URL / Directorio")
    table.add_column("Token", justify="center")
    table.add_column("Voces", justify="right")
    table.add_column("Velocidades")

    for p in providers:
        location = p.url or (f"{p.voices_dir}" if p.voices_dir else "—")
        token_str = "✅" if p.token_env else "—"
        voices_str = str(len(p.voices)) if p.voices else "auto"
        speeds_str = ", ".join(str(s) for s in p.speeds)
        table.add_row(p.name, p.type, location, token_str, voices_str, speeds_str)

    console.print(table)


@providers_app.command("add")
def providers_add(
    name: str = typer.Option(None, "--name", "-n", help="Nombre del provider."),
    type_: str = typer.Option(None, "--type", "-t", metavar="TYPE", help="Tipo: piper u openai."),
    url: str = typer.Option(None, "--url", help="URL del endpoint (openai)."),
    token_env: str = typer.Option(None, "--token-env", help="Variable de entorno del token."),
    voices: list[str] = typer.Option([], "--voice", help="Voz (repetible)."),
    speeds: list[float] = typer.Option([], "--speed", help="Velocidad (repetible)."),
):
    """Añade o actualiza un provider TTS global. Sin flags: lanza wizard interactivo."""
    if not name or not type_:
        from trainer.wizard import _wizard_add_provider
        _wizard_add_provider()
        return

    if type_ not in ("piper", "openai"):
        console.print(f"[red]Tipo inválido: '{type_}'. Usa 'piper' u 'openai'.[/red]")
        raise typer.Exit(1)

    from trainer.providers import ProviderConfig, add_or_update_provider
    provider = ProviderConfig(
        name=name,
        type=type_,
        url=url,
        token_env=token_env,
        voices=voices,
        speeds=speeds if speeds else [0.8, 0.9, 1.0, 1.1, 1.2],
    )
    add_or_update_provider(provider)
    console.print(f"  [green]✅ Provider '{name}' guardado.[/green]")


@providers_app.command("remove")
def providers_remove(
    name: str = typer.Argument(..., help="Nombre del provider a eliminar."),
):
    """Elimina un provider TTS global."""
    from trainer.providers import remove_provider
    if not remove_provider(name):
        console.print(f"[red]Provider '{name}' no encontrado.[/red]")
        raise typer.Exit(1)
    console.print(f"  [green]✅ Provider '{name}' eliminado.[/green]")
```

- [ ] **Step 4: Ejecutar tests**

```bash
pytest tests/test_providers_cli.py -v
```

Esperado: 7 tests PASS.

- [ ] **Step 5: Verificar en terminal**

```bash
wake-trainer providers --help
wake-trainer providers list
```

Esperado: ayuda con subcomandos `list`, `add`, `remove`, y mensaje "No hay providers configurados".

- [ ] **Step 6: Commit**

```bash
git add trainer/cli.py tests/test_providers_cli.py
git commit -m "feat: CLI providers — subcomandos list, add, remove con flags y wizard"
```

---

## Task 3: Wizard — integración con providers globales

**Files:**
- Modify: `trainer/wizard.py`

No hay tests unitarios nuevos para esta task — la lógica interactiva del wizard no es testeable con CliRunner (requiere stdin). La cobertura viene de los tests de Task 1 y Task 2.

- [ ] **Step 1: Añadir `_wizard_add_provider` al final de la sección de síntesis en `wizard.py`**

Añadir después de `_configure_openai_source` (busca la línea `# ─── Grabación / importación`):

```python
def _wizard_add_provider() -> None:
    """Wizard interactivo para añadir un provider TTS global."""
    import asyncio
    from trainer.providers import ProviderConfig, add_or_update_provider

    console.print(Panel("[bold]Nuevo provider TTS[/bold]", box=box.ROUNDED))

    name = ask("Nombre del provider (ej: elevenlabs, jspeaker)")
    if not name:
        return

    type_ = ask_choice("Tipo", ["openai", "piper"], default="openai")

    if type_ == "openai":
        url = ask("URL del endpoint (ej: https://api.elevenlabs.io/v1)")
        token_env_raw = ask("Variable de entorno del token (vacío si no hay autenticación)")
        token_env = token_env_raw.strip() or None

        token = os.environ.get(token_env, "") if token_env else ""
        console.print("  Consultando voces disponibles...")
        from trainer.synthesizer import list_voices_openai
        voices_raw = asyncio.run(list_voices_openai(url, token))

        if voices_raw:
            console.print(f"  ✓ {len(voices_raw)} voces encontradas")
            for i, v in enumerate(voices_raw[:20], 1):
                vname = v.get("name") or v.get("voice_id") or str(v)
                console.print(f"    {i}. {vname}")
            raw = ask("Selecciona voces (números separados por coma, o 'todas')", default="todas")
            if raw.lower() == "todas":
                voices = [v.get("name") or v.get("voice_id") for v in voices_raw]
            else:
                try:
                    indices = [int(x.strip()) - 1 for x in raw.split(",")]
                    voices = [
                        (voices_raw[i].get("name") or voices_raw[i].get("voice_id"))
                        for i in indices if 0 <= i < len(voices_raw)
                    ]
                except ValueError:
                    voices = [v.get("name") or v.get("voice_id") for v in voices_raw]
        else:
            console.print("  [yellow]No se encontraron voces automáticamente.[/yellow]")
            raw = ask("Introduce las voces manualmente (separadas por coma, vacío si solo hay una)")
            voices = [v.strip() for v in raw.split(",") if v.strip()] if raw.strip() else []

        provider = ProviderConfig(
            name=name,
            type="openai",
            url=url,
            token_env=token_env,
            voices=voices,
        )

    else:  # piper
        from trainer.synthesizer import list_voices_piper
        voices_dir = ask("Directorio de voces Piper", default="piper/voices")
        binary = ask("Ruta al binario piper", default="piper/piper")
        available = list_voices_piper(voices_dir)
        if not available:
            console.print(f"  [red]No se encontraron modelos .onnx en {voices_dir}[/red]")
            return
        console.print("  Voces disponibles:")
        for i, v in enumerate(available, 1):
            console.print(f"    {i}. {Path(v).stem}")
        raw = ask("Selecciona voces (números separados por coma, o 'todas')", default="todas")
        if raw.lower() == "todas":
            voices = available
        else:
            try:
                indices = [int(x.strip()) - 1 for x in raw.split(",")]
                voices = [available[i] for i in indices if 0 <= i < len(available)]
            except ValueError:
                voices = available
        provider = ProviderConfig(
            name=name,
            type="piper",
            binary=binary,
            voices_dir=voices_dir,
            voices=voices,
        )

    raw_speeds = ask(
        "Velocidades por defecto (ej: 0.8,1.0,1.2)",
        default=",".join(str(s) for s in provider.speeds),
    )
    try:
        provider.speeds = [float(s.strip()) for s in raw_speeds.split(",") if s.strip()]
    except ValueError:
        pass

    add_or_update_provider(provider)
    console.print(f"  [green]✅ Provider '{name}' guardado en configs/providers.local.json[/green]")
```

- [ ] **Step 2: Añadir `_provider_to_tts_source` justo después de `_wizard_add_provider`**

```python
def _provider_to_tts_source(provider: "ProviderConfig", project: Project) -> TtsSource | None:
    """Convierte un ProviderConfig global en un TtsSource para el proyecto."""
    import asyncio
    from trainer.providers import ProviderConfig as _PC  # import local para evitar circular

    if provider.voices:
        selected = provider.voices
    else:
        from trainer.synthesizer import list_voices_openai
        token = os.environ.get(provider.token_env, "") if provider.token_env else ""
        console.print(f"  Consultando voces de [bold]{provider.name}[/bold]...")
        voices_raw = asyncio.run(list_voices_openai(provider.url, token))
        if voices_raw:
            console.print(f"  ✓ {len(voices_raw)} voces disponibles")
            for i, v in enumerate(voices_raw[:20], 1):
                vname = v.get("name") or v.get("voice_id") or str(v)
                console.print(f"    {i}. {vname}")
            raw = ask("Selecciona voces (números, o 'todas')", default="todas")
            if raw.lower() == "todas":
                selected = [v.get("name") or v.get("voice_id") for v in voices_raw]
            else:
                try:
                    indices = [int(x.strip()) - 1 for x in raw.split(",")]
                    selected = [
                        (voices_raw[i].get("name") or voices_raw[i].get("voice_id"))
                        for i in indices if 0 <= i < len(voices_raw)
                    ]
                except ValueError:
                    selected = [v.get("name") or v.get("voice_id") for v in voices_raw]
        else:
            console.print("  [yellow]No se pudieron obtener voces automáticamente.[/yellow]")
            raw = ask("Introduce las voces manualmente (separadas por coma)")
            selected = [v.strip() for v in raw.split(",") if v.strip()]

    if not selected:
        return None

    raw_speeds = ask(
        f"Velocidades para {provider.name} en este proyecto",
        default=",".join(str(s) for s in provider.speeds),
    )
    try:
        speeds = [float(s.strip()) for s in raw_speeds.split(",") if s.strip()]
    except ValueError:
        speeds = provider.speeds

    return TtsSource(
        type=provider.type,
        url=provider.url,
        token_env=provider.token_env,
        binary=provider.binary,
        voices_dir=provider.voices_dir,
        selected_voices=selected,
        speeds=speeds,
    )
```

- [ ] **Step 3: Reemplazar el cuerpo de `_wizard_configure_synthesis` (líneas 133-154 del wizard actual)**

El bloque actual es:

```python
def _wizard_configure_synthesis(project: Project):
    while True:
        source_type = ask_choice(
            "Tipo de fuente TTS",
            ["piper", "openai", "ninguna"],
            default="piper",
        )
        if source_type == "ninguna":
            break

        if source_type == "piper":
            source = _configure_piper_source()
        else:
            source = _configure_openai_source()

        if source and source.selected_voices:
            project.synthesis.sources.append(source)
            save_project(project)

        more = ask_choice("¿Añadir otra fuente TTS?", ["s", "n"], default="n")
        if more != "s":
            break
```

Reemplazarlo con:

```python
def _wizard_configure_synthesis(project: Project):
    from trainer.providers import load_providers

    providers = load_providers()

    if not providers:
        add_now = ask_choice(
            "No hay providers TTS configurados. ¿Configurar uno ahora?",
            ["s", "n"],
            default="s",
        )
        if add_now == "s":
            _wizard_add_provider()
            providers = load_providers()

    if providers:
        while True:
            console.print("\n  [bold]Providers TTS disponibles:[/bold]")
            for i, p in enumerate(providers, 1):
                voices_info = f"{len(p.voices)} voces" if p.voices else "voces auto"
                console.print(f"    {i}. [bold]{p.name}[/bold] ({p.type}) — {voices_info}")
            console.print(f"    {len(providers) + 1}. Añadir nuevo provider ahora")
            console.print(f"    0. Omitir síntesis TTS")

            raw = ask("Selecciona provider(s) para este proyecto (números separados por coma)")
            if raw.strip() == "0":
                return

            try:
                indices = [int(x.strip()) for x in raw.split(",") if x.strip()]
            except ValueError:
                console.print("  [red]Entrada inválida.[/red]")
                continue

            for idx in indices:
                if idx == len(providers) + 1:
                    _wizard_add_provider()
                    providers = load_providers()
                elif 1 <= idx <= len(providers):
                    source = _provider_to_tts_source(providers[idx - 1], project)
                    if source and source.selected_voices:
                        project.synthesis.sources.append(source)
                        save_project(project)

            more = ask_choice("¿Añadir otro provider al proyecto?", ["s", "n"], default="n")
            if more != "s":
                break
    else:
        # Flujo manual de respaldo — sin providers configurados y usuario declinó
        while True:
            source_type = ask_choice(
                "Tipo de fuente TTS",
                ["piper", "openai", "ninguna"],
                default="piper",
            )
            if source_type == "ninguna":
                break
            if source_type == "piper":
                source = _configure_piper_source()
            else:
                source = _configure_openai_source()
            if source and source.selected_voices:
                project.synthesis.sources.append(source)
                save_project(project)
            more = ask_choice("¿Añadir otra fuente TTS?", ["s", "n"], default="n")
            if more != "s":
                break
```

- [ ] **Step 4: Verificar que el módulo importa sin errores**

```bash
python3 -c "from trainer.wizard import _wizard_add_provider, _provider_to_tts_source, _wizard_configure_synthesis; print('wizard OK')"
```

Esperado: `wizard OK`

- [ ] **Step 5: Ejecutar suite completa para verificar que no hay regresiones**

```bash
pytest tests/ -v --tb=short
```

Esperado: todos los tests anteriores siguen PASS (el wizard modificado no tiene tests unitarios nuevos, pero no debe romper nada).

- [ ] **Step 6: Commit**

```bash
git add trainer/wizard.py
git commit -m "feat: wizard — integración de providers globales en configuración de síntesis"
```

---

## Task 4: .gitignore y verificación final

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Añadir regla para `configs/providers.local.json` al `.gitignore`**

Añadir al final del `.gitignore` existente:

```
# Config local de providers TTS (contiene referencias a API keys — no versionar)
configs/providers.local.json
```

- [ ] **Step 2: Ejecutar la suite completa de tests**

```bash
pytest tests/ -v --tb=short
```

Esperado: todos los tests PASS (incluyendo los 8 nuevos de `test_providers.py` y los 7 de `test_providers_cli.py`).

- [ ] **Step 3: Smoke test del CLI**

```bash
wake-trainer providers list
wake-trainer providers add --name jspeaker --type openai --url http://localhost:5500 --voice Kaia --speed 1.0
wake-trainer providers list
wake-trainer providers add --name elevenlabs --type openai --url https://api.elevenlabs.io/v1 --token-env ELEVENLABS_API_KEY
wake-trainer providers list
```

Esperado: tabla con dos providers. Verificar que `configs/providers.local.json` existe con el contenido correcto.

```bash
cat configs/providers.local.json
```

- [ ] **Step 4: Verificar que `providers.local.json` está en .gitignore**

```bash
git status
```

Esperado: `configs/providers.local.json` NO aparece como untracked (está ignorado por git).

- [ ] **Step 5: Commit final**

```bash
git add .gitignore
git commit -m "chore: ignorar configs/providers.local.json en git"
```

---

## Resumen de dependencias entre tasks

```
Task 1 (providers.py) ──→ Task 2 (CLI providers)
                      ──→ Task 3 (wizard integration)
                                └──→ Task 4 (gitignore + verificación)
```
