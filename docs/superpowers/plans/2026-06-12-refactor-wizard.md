# Refactoring Wizard — Split wizard.py en ui/ y workflows/ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactorizar `trainer/wizard.py` (615 líneas, 18 funciones) en módulos cohesionados sin cambiar ningún comportamiento externo.

**Architecture:** Separación en dos capas: `trainer/ui/` contiene componentes presentacionales puros (sin decisiones de negocio), `trainer/workflows/` contiene la orquestación que coordina pasos y muta estado. `trainer/wizard.py` queda como coordinador de ~90 líneas que solo gestiona el flujo de entrada del wizard principal. Las 54 pruebas existentes deben pasar en cada commit.

**Tech Stack:** Python 3.11+, Typer, Rich, pytest, dataclasses

---

## Estructura de ficheros

### Nuevos
- `trainer/ui/voice_selection.py` — Selección interactiva de voces (openai y piper); sin estado
- `trainer/ui/tables.py` — Renderizado de tablas Rich (providers); sin estado
- `trainer/workflows/__init__.py` — Re-exporta las funciones públicas de los submódulos
- `trainer/workflows/recording.py` — Grabación e importación de clips; muta `Project`
- `trainer/workflows/synthesis.py` — Configuración TTS, síntesis, provider wizard; muta `Project`
- `trainer/workflows/training.py` — Entrenamiento y evaluación; muta `Project`
- `tests/test_ui_voice_selection.py` — 5 tests para la selección de voces
- `tests/test_ui_tables.py` — 2 tests para `print_providers_table`

### Modificados
- `trainer/wizard.py` — reducido de 615 a ~90 líneas; importa de workflows
- `trainer/cli.py` — usa `print_providers_table` y `add_provider_interactive` directamente

---

## Task 1: trainer/ui/voice_selection.py

**Files:**
- Create: `trainer/ui/voice_selection.py`
- Create: `tests/test_ui_voice_selection.py`

- [ ] **Step 1: Escribir los tests que fallan**

```python
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
    assert vs_mod.select_openai_voices(VOICES_RAW) == ["Alice", "Bob", "Carol"]


def test_select_openai_indices(monkeypatch):
    monkeypatch.setattr(vs_mod, "ask", lambda *a, **kw: "1,3")
    assert vs_mod.select_openai_voices(VOICES_RAW) == ["Alice", "Carol"]


def test_select_openai_invalid_fallback(monkeypatch):
    monkeypatch.setattr(vs_mod, "ask", lambda *a, **kw: "no_es_numero")
    assert vs_mod.select_openai_voices(VOICES_RAW) == ["Alice", "Bob", "Carol"]


def test_select_piper_todas(monkeypatch):
    monkeypatch.setattr(vs_mod, "ask", lambda *a, **kw: "todas")
    assert vs_mod.select_piper_voices(PIPER_VOICES) == PIPER_VOICES


def test_select_piper_indices(monkeypatch):
    monkeypatch.setattr(vs_mod, "ask", lambda *a, **kw: "2")
    assert vs_mod.select_piper_voices(PIPER_VOICES) == ["piper/voices/en_US.onnx"]
```

- [ ] **Step 2: Verificar que los tests fallan**

```
python3 -m pytest tests/test_ui_voice_selection.py -v
```
Esperado: 5 FAILED con `ModuleNotFoundError` o `AttributeError`

- [ ] **Step 3: Implementar `trainer/ui/voice_selection.py`**

```python
# trainer/ui/voice_selection.py
from __future__ import annotations
from pathlib import Path
from rich.console import Console
from trainer.ui.prompts import ask

console = Console()


def select_openai_voices(voices_raw: list[dict]) -> list[str]:
    console.print(f"  ✓ {len(voices_raw)} voces encontradas")
    for i, v in enumerate(voices_raw[:20], 1):
        name = v.get("name") or v.get("voice_id") or str(v)
        console.print(f"    {i}. {name}")
    raw = ask("Selecciona voces (números separados por coma, o 'todas')", default="todas")
    if raw.lower() == "todas":
        return [v.get("name") or v.get("voice_id") for v in voices_raw]
    try:
        indices = [int(x.strip()) - 1 for x in raw.split(",")]
        return [
            (voices_raw[i].get("name") or voices_raw[i].get("voice_id"))
            for i in indices if 0 <= i < len(voices_raw)
        ]
    except ValueError:
        return [v.get("name") or v.get("voice_id") for v in voices_raw]


def select_piper_voices(available: list[str]) -> list[str]:
    console.print("  Voces disponibles:")
    for i, v in enumerate(available, 1):
        console.print(f"    {i}. {Path(v).stem}")
    raw = ask("Selecciona voces (números separados por coma, o 'todas')", default="todas")
    if raw.lower() == "todas":
        return available
    try:
        indices = [int(x.strip()) - 1 for x in raw.split(",")]
        return [available[i] for i in indices if 0 <= i < len(available)]
    except ValueError:
        return available
```

- [ ] **Step 4: Verificar que los 5 tests pasan**

```
python3 -m pytest tests/test_ui_voice_selection.py -v
```
Esperado: 5 PASSED

- [ ] **Step 5: Suite completa sin regresiones**

```
python3 -m pytest tests/ -v --tb=short
```
Esperado: 59 PASSED (54 previos + 5 nuevos)

- [ ] **Step 6: Commit**

```bash
git add trainer/ui/voice_selection.py tests/test_ui_voice_selection.py
git commit -m "feat: extraer selección de voces a trainer/ui/voice_selection.py"
```

---

## Task 2: trainer/ui/tables.py

**Files:**
- Create: `trainer/ui/tables.py`
- Create: `tests/test_ui_tables.py`

- [ ] **Step 1: Escribir los tests que fallan**

```python
# tests/test_ui_tables.py
from __future__ import annotations
from io import StringIO
import pytest
from rich.console import Console
from trainer.providers import ProviderConfig


def _make_console():
    buf = StringIO()
    return Console(file=buf, width=120, highlight=False), buf


def test_print_providers_table_empty():
    import trainer.ui.tables as tables_mod
    con, buf = _make_console()
    tables_mod.console = con
    tables_mod.print_providers_table([])
    assert "providers add" in buf.getvalue()


def test_print_providers_table_shows_provider():
    import trainer.ui.tables as tables_mod
    con, buf = _make_console()
    tables_mod.console = con
    p = ProviderConfig(name="elevenlabs", type="openai", url="https://api.elevenlabs.io/v1")
    tables_mod.print_providers_table([p])
    assert "elevenlabs" in buf.getvalue()
```

- [ ] **Step 2: Verificar que los tests fallan**

```
python3 -m pytest tests/test_ui_tables.py -v
```
Esperado: 2 FAILED

- [ ] **Step 3: Implementar `trainer/ui/tables.py`**

```python
# trainer/ui/tables.py
from __future__ import annotations
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


def print_providers_table(providers: list) -> None:
    if not providers:
        console.print(
            "[dim]No hay providers configurados. "
            "Usa [bold]wake-trainer providers add[/bold] para añadir uno.[/dim]"
        )
        return
    table = Table(box=box.SIMPLE_HEAD)
    table.add_column("Nombre", style="bold")
    table.add_column("Tipo")
    table.add_column("URL / Directorio")
    table.add_column("Token", justify="center")
    table.add_column("Voces", justify="right")
    table.add_column("Velocidades")
    for p in providers:
        location = p.url or (str(p.voices_dir) if p.voices_dir else "—")
        token_str = "✅" if p.token_env else "—"
        voices_str = str(len(p.voices)) if p.voices else "auto"
        speeds_str = ", ".join(str(s) for s in p.speeds)
        table.add_row(p.name, p.type, location, token_str, voices_str, speeds_str)
    console.print(table)
```

- [ ] **Step 4: Verificar que los 2 tests pasan**

```
python3 -m pytest tests/test_ui_tables.py -v
```
Esperado: 2 PASSED

- [ ] **Step 5: Suite completa sin regresiones**

```
python3 -m pytest tests/ -v --tb=short
```
Esperado: 61 PASSED

- [ ] **Step 6: Commit**

```bash
git add trainer/ui/tables.py tests/test_ui_tables.py
git commit -m "feat: extraer tabla de providers a trainer/ui/tables.py"
```

---

## Task 3: trainer/workflows/recording.py

**Files:**
- Create: `trainer/workflows/__init__.py` (vacío por ahora)
- Create: `trainer/workflows/recording.py`
- Modify: (sin cambios en wizard.py todavía — se hace en Task 7)

- [ ] **Step 1: Crear el paquete**

```python
# trainer/workflows/__init__.py
```
(fichero vacío)

- [ ] **Step 2: Crear `trainer/workflows/recording.py`**

```python
# trainer/workflows/recording.py
from __future__ import annotations
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich import box

from trainer.state import Project, save_project, load_project
from trainer.ui.prompts import ask, ask_choice, explain

console = Console()


def show_preparation_summary(project: Project) -> None:
    pending = [v for v in project.voices if v.status == "pending"]
    console.print(Panel(
        "\n".join(
            f"  {'•'} [bold]{v.name}[/bold] — "
            + ("30 clips en 10 condiciones" if v.mode == "record" else
               "importar ficheros WAV" if v.mode == "import" else "sin definir")
            for v in pending
        ),
        title="Preparación para grabación",
        box=box.ROUNDED,
    ))
    console.print()


def record_or_import_voice(project: Project, voice) -> None:
    explain(
        f"Para la voz de [bold]{voice.name}[/bold], elige cómo obtener las grabaciones:\n"
        f"  [bold]g[/bold] — Grabar ahora con el micrófono de este dispositivo\n"
        f"  [bold]i[/bold] — Importar ficheros WAV que {voice.name} te haya enviado\n"
        f"  [bold]d[/bold] — Dejar para más tarde"
    )
    action = ask_choice(f"Voz de «{voice.name}»", ["g", "i", "d"], default="g")

    if action == "d":
        return

    voice.mode = "record" if action == "g" else "import"

    if action == "g":
        from trainer.recorder import record_voice
        clips = record_voice(project.positivos_path / voice.name, project.wake_word)
        voice.clips = clips
        if clips >= 30:
            voice.status = "done"

    elif action == "i":
        src_raw = ask("Ruta de la carpeta con los WAVs")
        src_dir = Path(src_raw).expanduser()
        from trainer.importer import import_clips
        count, invalid = import_clips(src_dir, project.positivos_path / voice.name)
        console.print(f"  [green]✅ {count} clips importados[/green]")
        if invalid:
            console.print(f"  [yellow]⚠️  {len(invalid)} ficheros ignorados[/yellow]")
        voice.clips = count
        if count >= 30:
            voice.status = "done"
        elif count > 0:
            voice.status = "done"

    save_project(project)


def run_record_step(model_name: str, voice_name: str | None) -> None:
    project = load_project(model_name)
    if voice_name:
        voice = next((v for v in project.voices if v.name == voice_name), None)
        if not voice:
            console.print(f"[red]Voz '{voice_name}' no encontrada en el proyecto.[/red]")
            return
        record_or_import_voice(project, voice)
    else:
        pending = [v for v in project.voices if v.status == "pending"]
        for voice in pending:
            record_or_import_voice(project, voice)


def run_import_step(model_name: str, voice_name: str | None, directory: str | None) -> None:
    project = load_project(model_name)
    voice = next((v for v in project.voices if v.name == voice_name), None) if voice_name else None
    if voice_name and not voice:
        console.print(f"[red]Voz '{voice_name}' no encontrada.[/red]")
        return
    src_dir = Path(directory).expanduser() if directory else Path(ask("Carpeta con WAVs")).expanduser()
    target_dir = project.positivos_path / (voice.name if voice else "import")
    from trainer.importer import import_clips
    count, invalid = import_clips(src_dir, target_dir)
    console.print(f"  [green]✅ {count} clips importados[/green]")
    if invalid:
        console.print(f"  [yellow]⚠️  {len(invalid)} ficheros ignorados[/yellow]")
```

- [ ] **Step 3: Verificar que la suite sigue pasando (sin cambiar wizard.py aún)**

```
python3 -m pytest tests/ -v --tb=short
```
Esperado: 61 PASSED (wizard.py no cambia hasta Task 7)

- [ ] **Step 4: Commit**

```bash
git add trainer/workflows/__init__.py trainer/workflows/recording.py
git commit -m "feat: extraer grabación e importación a trainer/workflows/recording.py"
```

---

## Task 4: trainer/workflows/training.py

**Files:**
- Create: `trainer/workflows/training.py`

- [ ] **Step 1: Crear `trainer/workflows/training.py`**

```python
# trainer/workflows/training.py
from __future__ import annotations
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich import box

from trainer.state import Project, save_project, load_project, calculate_dataset
from trainer.ui.prompts import ask_choice

console = Console()


def train_project(project: Project) -> None:
    from trainer.trainer_core import run_training, TrainingConfig

    console.print(Panel("[bold]Entrenamiento[/bold]", box=box.ROUNDED))
    stats = calculate_dataset(project)
    console.print(f"  Dataset estimado: [bold]{stats['total']}[/bold] muestras")

    confirm = ask_choice("¿Iniciar entrenamiento?", ["s", "n"], default="s")
    if confirm != "s":
        return

    project.training.status = "in_progress"
    save_project(project)

    cfg = TrainingConfig(
        model_name=project.model_name,
        output_dir=project.models_path,
    )

    try:
        tflite_path = run_training(project.positivos_path, cfg)
        project.training.status = "done"
        project.model_path = str(tflite_path)
        save_project(project)
        console.print(f"\n  [green]✓ Modelo guardado en {tflite_path}[/green]")
    except Exception as exc:
        project.training.status = "pending"
        save_project(project)
        console.print(f"\n  [red]✗ Entrenamiento fallido: {exc}[/red]")


def evaluate_project(project: Project) -> None:
    from trainer.evaluator import evaluate_model
    if not project.model_path:
        console.print("[yellow]No hay modelo entrenado para evaluar.[/yellow]")
        return

    console.print(Panel("[bold]Evaluación del modelo[/bold]", box=box.ROUNDED))
    result = evaluate_model(
        model_path=Path(project.model_path),
        positivos_path=project.positivos_path,
    )
    console.print(f"""
  Modelo:            [bold]{project.model_path}[/bold]
  Precisión:         [bold]{result.precision*100:.1f}%[/bold]
  Recall:            [bold]{result.recall*100:.1f}%[/bold]
  Falsos positivos:  [bold]{result.false_positives}[/bold] en 10 s de silencio
  Threshold:         [bold]{result.threshold}[/bold]
    """)

    if result.passed():
        console.print("  [green]✅ El modelo supera los umbrales mínimos.[/green]")
    else:
        console.print("  [yellow]⚠️  Considera grabar más muestras y reentrenar.[/yellow]")


def run_train_step(model_name: str) -> None:
    project = load_project(model_name)
    train_project(project)


def run_evaluate_step(model_name: str) -> None:
    project = load_project(model_name)
    evaluate_project(project)
```

- [ ] **Step 2: Suite completa sin regresiones**

```
python3 -m pytest tests/ -v --tb=short
```
Esperado: 61 PASSED

- [ ] **Step 3: Commit**

```bash
git add trainer/workflows/training.py
git commit -m "feat: extraer entrenamiento y evaluación a trainer/workflows/training.py"
```

---

## Task 5: trainer/workflows/synthesis.py

**Files:**
- Create: `trainer/workflows/synthesis.py`

Depende de: Task 1 (`trainer/ui/voice_selection.py`)

- [ ] **Step 1: Crear `trainer/workflows/synthesis.py`**

```python
# trainer/workflows/synthesis.py
from __future__ import annotations
import os
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich import box

from trainer.state import Project, TtsSource, save_project, load_project
from trainer.ui.prompts import ask, ask_choice
from trainer.ui.voice_selection import select_openai_voices, select_piper_voices

console = Console()


def configure_piper_source() -> TtsSource | None:
    from trainer.synthesizer import list_voices_piper
    voices_dir = ask("Directorio de voces Piper", default="piper/voices")
    available = list_voices_piper(voices_dir)
    if not available:
        console.print(f"  [red]No se encontraron modelos .onnx en {voices_dir}[/red]")
        return None
    selected = select_piper_voices(available)
    return TtsSource(type="piper", voices_dir=voices_dir, selected_voices=selected)


def configure_openai_source() -> TtsSource | None:
    import asyncio
    from trainer.synthesizer import list_voices_openai

    url = ask("URL del endpoint (ej: https://api.elevenlabs.io/v1)")
    token_env = ask("Variable de entorno del token (ej: ELEVENLABS_API_KEY)")
    token = os.environ.get(token_env, "")
    if not token:
        token = console.input("  Token (se usará solo ahora, no se guarda): ").strip()

    console.print("  Consultando voces disponibles...")
    voices_raw = asyncio.run(list_voices_openai(url, token))

    if voices_raw:
        selected = select_openai_voices(voices_raw)
    else:
        console.print("  [yellow]No se pudo obtener la lista de voces. Entrada manual.[/yellow]")
        raw = ask("Nombres de las voces (separados por coma)")
        selected = [v.strip() for v in raw.split(",") if v.strip()]

    return TtsSource(type="openai", url=url, token_env=token_env, selected_voices=selected)


def add_provider_interactive() -> None:
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
            voices = select_openai_voices(voices_raw)
        else:
            console.print("  [yellow]No se encontraron voces automáticamente.[/yellow]")
            raw = ask("Introduce las voces manualmente (separadas por coma, vacío si solo hay una)")
            voices = [v.strip() for v in raw.split(",") if v.strip()] if raw.strip() else []

        provider = ProviderConfig(name=name, type="openai", url=url, token_env=token_env, voices=voices)

    else:  # piper
        from trainer.synthesizer import list_voices_piper
        voices_dir = ask("Directorio de voces Piper", default="piper/voices")
        binary = ask("Ruta al binario piper", default="piper/piper")
        available = list_voices_piper(voices_dir)
        if not available:
            console.print(f"  [red]No se encontraron modelos .onnx en {voices_dir}[/red]")
            return
        voices = select_piper_voices(available)
        provider = ProviderConfig(
            name=name, type="piper", binary=binary, voices_dir=voices_dir, voices=voices
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


def _provider_to_tts_source(provider) -> TtsSource | None:
    import asyncio

    if provider.voices:
        selected = provider.voices
    elif provider.type == "openai":
        from trainer.synthesizer import list_voices_openai
        token = os.environ.get(provider.token_env, "") if provider.token_env else ""
        console.print(f"  Consultando voces de [bold]{provider.name}[/bold]...")
        voices_raw = asyncio.run(list_voices_openai(provider.url, token))
        if voices_raw:
            console.print(f"  ✓ {len(voices_raw)} voces disponibles")
            selected = select_openai_voices(voices_raw)
        else:
            console.print("  [yellow]No se pudieron obtener voces automáticamente.[/yellow]")
            raw = ask("Introduce las voces manualmente (separadas por coma)")
            selected = [v.strip() for v in raw.split(",") if v.strip()]
    else:  # piper
        from trainer.synthesizer import list_voices_piper
        console.print(f"  Escaneando modelos en {provider.voices_dir}...")
        available = list_voices_piper(provider.voices_dir or "piper/voices")
        if not available:
            console.print(f"  [red]No se encontraron modelos .onnx en {provider.voices_dir}[/red]")
            selected = []
        else:
            console.print(f"  ✓ {len(available)} modelos encontrados")
            selected = select_piper_voices(available)

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


def configure_synthesis(project: Project) -> None:
    from trainer.providers import load_providers

    providers = load_providers()

    if not providers:
        add_now = ask_choice(
            "No hay providers TTS configurados. ¿Configurar uno ahora?",
            ["s", "n"],
            default="s",
        )
        if add_now == "s":
            add_provider_interactive()
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
                    add_provider_interactive()
                    providers = load_providers()
                elif 1 <= idx <= len(providers):
                    source = _provider_to_tts_source(providers[idx - 1])
                    if source and source.selected_voices:
                        project.synthesis.sources.append(source)
                        save_project(project)

            more = ask_choice("¿Añadir otro provider al proyecto?", ["s", "n"], default="n")
            if more != "s":
                break
    else:
        while True:
            source_type = ask_choice(
                "Tipo de fuente TTS",
                ["piper", "openai", "ninguna"],
                default="piper",
            )
            if source_type == "ninguna":
                break
            if source_type == "piper":
                source = configure_piper_source()
            else:
                source = configure_openai_source()
            if source and source.selected_voices:
                project.synthesis.sources.append(source)
                save_project(project)
            more = ask_choice("¿Añadir otra fuente TTS?", ["s", "n"], default="n")
            if more != "s":
                break


def synthesize_project(project: Project) -> None:
    from trainer.synthesizer import run_synthesis
    console.print("\n[bold]Generando muestras sintéticas...[/bold]")
    generated = run_synthesis(project)
    console.print(f"  [green]✓ {generated} clips generados[/green]")


def run_synthesize_step(model_name: str) -> None:
    project = load_project(model_name)
    if not project.synthesis.sources:
        configure_synthesis(project)
    synthesize_project(project)
```

- [ ] **Step 2: Suite completa sin regresiones**

```
python3 -m pytest tests/ -v --tb=short
```
Esperado: 61 PASSED

- [ ] **Step 3: Commit**

```bash
git add trainer/workflows/synthesis.py
git commit -m "feat: extraer síntesis y provider wizard a trainer/workflows/synthesis.py"
```

---

## Task 6: trainer/workflows/__init__.py — Re-exportaciones

**Files:**
- Modify: `trainer/workflows/__init__.py`

Depende de: Tasks 3, 4, 5

- [ ] **Step 1: Actualizar `trainer/workflows/__init__.py`**

Reemplazar el fichero vacío con:

```python
# trainer/workflows/__init__.py
from trainer.workflows.recording import run_record_step, run_import_step
from trainer.workflows.synthesis import run_synthesize_step
from trainer.workflows.training import run_train_step, run_evaluate_step

__all__ = [
    "run_record_step",
    "run_import_step",
    "run_synthesize_step",
    "run_train_step",
    "run_evaluate_step",
]
```

- [ ] **Step 2: Suite completa sin regresiones**

```
python3 -m pytest tests/ -v --tb=short
```
Esperado: 61 PASSED

- [ ] **Step 3: Commit**

```bash
git add trainer/workflows/__init__.py
git commit -m "feat: re-exportar funciones públicas desde trainer/workflows/__init__.py"
```

---

## Task 7: Reducir trainer/wizard.py a ~90 líneas

**Files:**
- Modify: `trainer/wizard.py`

Depende de: Tasks 3, 4, 5, 6

El objetivo es eliminar todo el código que ya vive en `workflows/`, dejando solo el flujo del wizard principal y los re-exports de compatibilidad hacia atrás.

- [ ] **Step 1: Reemplazar `trainer/wizard.py` con la versión reducida**

El nuevo contenido completo del fichero (sustituye todo el contenido actual):

```python
# trainer/wizard.py
from __future__ import annotations
from rich.console import Console
from rich.panel import Panel
from rich import box

from trainer.state import (
    Project, create_project, list_projects, calculate_dataset,
)
from trainer.ui.panels import (
    print_header, print_project_list, print_project_status, print_dataset_summary,
)
from trainer.ui.prompts import explain, ask, ask_choice, ask_int
from trainer.workflows.recording import show_preparation_summary, record_or_import_voice
from trainer.workflows.synthesis import configure_synthesis, synthesize_project
from trainer.workflows.training import train_project, evaluate_project
from trainer.workflows import (
    run_record_step, run_import_step, run_synthesize_step,
    run_train_step, run_evaluate_step,
)

console = Console()


def run_wizard():
    print_header()
    projects = list_projects()

    if not projects:
        console.print("\n[dim]No hay proyectos. Creando uno nuevo...[/dim]\n")
        project = _wizard_new_project()
        if project:
            _wizard_continue(project)
        return

    print_project_list(projects)
    console.print()
    choice = console.input(
        "  Elige un proyecto [1-{n}], [n] nuevo, [q] salir: ".format(n=len(projects))
    ).strip().lower()

    if choice == "q":
        return
    if choice == "n":
        project = _wizard_new_project()
        if project:
            _wizard_continue(project)
        return

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(projects):
            _wizard_continue(projects[idx])
    except ValueError:
        console.print("[red]Opción no válida.[/red]")


def _wizard_continue(project: Project):
    print_project_status(project)
    console.print()

    pending_voices = [v for v in project.voices if v.status == "pending"]
    if pending_voices:
        show_preparation_summary(project)
        for voice in pending_voices:
            record_or_import_voice(project, voice)

    if project.synthesis.status == "pending" and project.synthesis.sources:
        synthesize_project(project)

    if project.synthesis.status != "done":
        configure_synthesis(project)
        synthesize_project(project)

    if project.training.status == "pending":
        train_project(project)

    if project.training.status == "done" and not project.model_path:
        evaluate_project(project)


def _wizard_new_project() -> Project | None:
    console.print(Panel("[bold]Nuevo proyecto de wake word[/bold]", box=box.ROUNDED))

    wake_word = ask("Frase de la wake word").lower()
    if not wake_word:
        return None

    suggested_name = wake_word.replace(" ", "_").replace("'", "")
    model_name = ask("Nombre del modelo", default=suggested_name)
    if not model_name:
        return None

    explain(
        "Para que el modelo funcione bien con las personas que lo van a usar,\n"
        "necesitamos grabar su voz directamente. Cada persona graba [bold]30 clips[/bold]\n"
        "en 10 condiciones distintas (distancia, ruido, velocidad).\n"
        "Cuantas más personas graben, mejor reconocerá el modelo sus voces."
    )
    n_voices = ask_int("¿Cuántas personas van a grabar?", minimum=1)
    voice_names = []
    for i in range(1, n_voices + 1):
        name = ask(f"  Nombre de la persona {i}")
        voice_names.append(name or f"Persona {i}")

    project = create_project(wake_word, model_name, voice_names)

    explain(
        "Además de las voces reales, generaremos muestras sintéticas con servicios\n"
        "text-to-speech (TTS). Esto hace el modelo más robusto para personas que\n"
        "no hayan grabado su voz, cubriendo distintos acentos, géneros y estilos."
    )
    use_tts = ask_choice("¿Añadir fuentes TTS ahora?", ["s", "n"], default="s")
    if use_tts == "s":
        configure_synthesis(project)

    print_dataset_summary(project)
    stats = calculate_dataset(project)
    if not stats["meets_minimum"]:
        console.print("\n  [yellow]⚠️  El dataset estimado es menor de 1.000 muestras.[/yellow]")
        console.print("  Considera añadir más personas o fuentes TTS antes de entrenar.\n")

    confirm = ask_choice("¿Crear proyecto?", ["s", "n"], default="s")
    if confirm != "s":
        import shutil
        shutil.rmtree(project.root, ignore_errors=True)
        return None

    return project


def _wizard_add_provider() -> None:
    from trainer.workflows.synthesis import add_provider_interactive
    add_provider_interactive()
```

- [ ] **Step 2: Suite completa sin regresiones**

```
python3 -m pytest tests/ -v --tb=short
```
Esperado: 61 PASSED — si falla alguno, revisar imports antes de continuar.

- [ ] **Step 3: Verificar que wizard.py tiene menos de 110 líneas**

```bash
wc -l trainer/wizard.py
```
Esperado: ≤ 110

- [ ] **Step 4: Commit**

```bash
git add trainer/wizard.py
git commit -m "refactor: reducir wizard.py a coordinador de flujo principal (~95 líneas)"
```

---

## Task 8: Actualizar trainer/cli.py

**Files:**
- Modify: `trainer/cli.py`

Depende de: Tasks 2, 5, 6

- [ ] **Step 1: Actualizar los imports y cuerpos en `trainer/cli.py`**

Tres cambios:

**a) Al inicio del fichero, añadir import de `print_providers_table`** — insertar después del import existente de `panels`:

```python
from trainer.ui.tables import print_providers_table
```

**b) Reemplazar el cuerpo completo de `providers_list`** (actualmente líneas 27–56):

```python
@providers_app.command("list")
def providers_list():
    """Lista todos los providers TTS configurados."""
    from trainer.providers import load_providers
    print_providers_table(load_providers())
```

**c) En `providers_add`, reemplazar el bloque que importa y llama a `_wizard_add_provider`** (actualmente):

```python
        from trainer.wizard import _wizard_add_provider
        _wizard_add_provider()
```

Por:

```python
        from trainer.workflows.synthesis import add_provider_interactive
        add_provider_interactive()
```

- [ ] **Step 2: Suite completa sin regresiones**

```
python3 -m pytest tests/ -v --tb=short
```
Esperado: 61 PASSED

- [ ] **Step 3: Verificar que los tests CLI de providers siguen pasando en concreto**

```
python3 -m pytest tests/test_providers_cli.py -v
```
Esperado: 7 PASSED

- [ ] **Step 4: Commit final**

```bash
git add trainer/cli.py
git commit -m "refactor: cli.py usa print_providers_table y add_provider_interactive directamente"
```

---

## Resultado final

Tras los 8 tasks:

| Fichero | Líneas antes | Líneas después |
|---------|-------------|----------------|
| `trainer/wizard.py` | 615 | ~95 |
| `trainer/cli.py` | 167 | ~140 |
| `trainer/ui/voice_selection.py` | — | ~35 |
| `trainer/ui/tables.py` | — | ~30 |
| `trainer/workflows/recording.py` | — | ~70 |
| `trainer/workflows/synthesis.py` | — | ~175 |
| `trainer/workflows/training.py` | — | ~60 |
| `trainer/workflows/__init__.py` | — | ~10 |

**Tests:** 61 (54 previos + 5 voice_selection + 2 tables), todos pasando.

**Regla de diseño vigente:**
- `trainer/ui/` → muestra cosas, no toma decisiones, no muta estado
- `trainer/workflows/` → coordina pasos, llama a ui y a business logic, muta `Project`
- `trainer/wizard.py` → flujo de entrada del wizard interactivo; re-exporta las funciones CLI públicas
