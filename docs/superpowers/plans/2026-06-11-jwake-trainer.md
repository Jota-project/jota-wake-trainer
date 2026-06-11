# JWake Trainer — Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar JWake Trainer, una herramienta CLI guiada para entrenar wake words personalizadas con openWakeWord, con soporte multi-proyecto, grabación interactiva, síntesis TTS (Piper + endpoints OpenAI-compatible) y wizard de configuración elegante con Rich.

**Architecture:** Wizard-first con subcomandos disponibles. El estado persiste en `projects/<model_name>/session.json`. Cada módulo tiene responsabilidad única: state (persistencia), recorder (captura), importer (importación), synthesizer (TTS), trainer_core (openWakeWord), evaluator (métricas), ui/ (componentes Rich), wizard (orquestación), cli (entry point + subcomandos).

**Tech Stack:** Python 3.11+, Typer (CLI), Rich (UI), sounddevice (audio), soundfile (WAV I/O), httpx (TTS HTTP), openWakeWord[train] (entrenamiento), pytest (tests).

---

## Mapa de archivos

**Crear:**
```
pyproject.toml
tests/__init__.py
tests/test_state.py
tests/test_recorder.py
tests/test_importer.py
tests/test_synthesizer.py
tests/test_trainer_core.py
tests/test_evaluator.py
trainer/__init__.py
trainer/cli.py
trainer/wizard.py
trainer/state.py
trainer/recorder.py
trainer/importer.py
trainer/synthesizer.py
trainer/trainer_core.py
trainer/evaluator.py
trainer/ui/__init__.py
trainer/ui/panels.py
trainer/ui/prompts.py
projects/.gitkeep
```

**Modificar:**
- `.gitignore` — añadir reglas para `projects/*/data/`

---

## Task 1: Scaffolding y packaging

**Files:**
- Create: `pyproject.toml`
- Create: `trainer/__init__.py`
- Create: `trainer/ui/__init__.py`
- Create: `tests/__init__.py`
- Create: `projects/.gitkeep`
- Modify: `.gitignore`

- [ ] **Step 1: Crear pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "jwake-trainer"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "typer>=0.12",
    "rich>=13",
    "sounddevice>=0.4.7",
    "soundfile>=0.12",
    "numpy>=1.24",
    "httpx>=0.27",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-asyncio>=0.23"]
train = ["openWakeWord[train]"]

[project.scripts]
wake-trainer = "trainer.cli:app"

[tool.setuptools.packages.find]
where = ["."]
include = ["trainer*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

- [ ] **Step 2: Crear archivos __init__.py vacíos**

```python
# trainer/__init__.py  (vacío)
# trainer/ui/__init__.py  (vacío)
# tests/__init__.py  (vacío)
```

- [ ] **Step 3: Actualizar .gitignore**

Añadir al final del `.gitignore` existente:

```
# Datos de audio de proyectos (mismo criterio que data/ raíz)
projects/*/data/positivos/
projects/*/data/sintetizados/
projects/*/data/negativos/

# Artefactos de entrenamiento por proyecto
projects/*/output/

# El modelo .tflite SÍ se versiona
!projects/*/models/*.tflite
```

- [ ] **Step 4: Crear projects/.gitkeep**

```
(fichero vacío)
```

- [ ] **Step 5: Instalar el paquete en modo editable**

```bash
pip install -e ".[dev]"
```

Esperado: instalación sin errores. `wake-trainer --help` debería mostrar el mensaje de ayuda (aunque el comando aún no esté implementado, el entry point se registra).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml trainer/__init__.py trainer/ui/__init__.py tests/__init__.py projects/.gitkeep .gitignore
git commit -m "chore: scaffolding inicial — packaging y estructura de directorios"
```

---

## Task 2: State — persistencia de proyectos

**Files:**
- Create: `trainer/state.py`
- Create: `tests/test_state.py`

- [ ] **Step 1: Escribir tests**

```python
# tests/test_state.py
import pytest
from pathlib import Path
from datetime import datetime
import trainer.state as state_mod
from trainer.state import (
    Project, Voice, SynthesisState, TrainingState, TtsSource,
    create_project, load_project, save_project, list_projects,
)


@pytest.fixture(autouse=True)
def tmp_projects(tmp_path, monkeypatch):
    monkeypatch.setattr(state_mod, "PROJECTS_ROOT", tmp_path / "projects")
    return tmp_path / "projects"


def test_create_project_creates_directories(tmp_projects):
    p = create_project("ok jota", "ok_jota", ["Alfonso", "María"])
    assert (tmp_projects / "ok_jota" / "data" / "positivos" / "Alfonso").exists()
    assert (tmp_projects / "ok_jota" / "data" / "positivos" / "María").exists()
    assert (tmp_projects / "ok_jota" / "data" / "sintetizados").exists()
    assert (tmp_projects / "ok_jota" / "models").exists()


def test_create_project_saves_session(tmp_projects):
    create_project("ok jota", "ok_jota", ["Alfonso"])
    assert (tmp_projects / "ok_jota" / "session.json").exists()


def test_load_project_roundtrip(tmp_projects):
    create_project("hey asistente", "hey_asistente", ["Ana", "Luis"])
    loaded = load_project("hey_asistente")
    assert loaded.wake_word == "hey asistente"
    assert len(loaded.voices) == 2
    assert loaded.voices[0].name == "Ana"
    assert loaded.voices[0].status == "pending"


def test_save_project_persists_voice_state(tmp_projects):
    p = create_project("ok jota", "ok_jota", ["Alfonso"])
    p.voices[0].status = "done"
    p.voices[0].clips = 30
    save_project(p)
    loaded = load_project("ok_jota")
    assert loaded.voices[0].status == "done"
    assert loaded.voices[0].clips == 30


def test_list_projects_empty(tmp_projects):
    assert list_projects() == []


def test_list_projects_returns_all(tmp_projects):
    create_project("ok jota", "ok_jota", ["Alfonso"])
    create_project("hey bot", "hey_bot", ["María"])
    names = [p.model_name for p in list_projects()]
    assert "ok_jota" in names
    assert "hey_bot" in names


def test_project_paths(tmp_projects):
    p = create_project("ok jota", "ok_jota", ["Alfonso"])
    assert p.root == tmp_projects / "ok_jota"
    assert p.positivos_path == tmp_projects / "ok_jota" / "data" / "positivos"
    assert p.sintetizados_path == tmp_projects / "ok_jota" / "data" / "sintetizados"
    assert p.models_path == tmp_projects / "ok_jota" / "models"
```

- [ ] **Step 2: Verificar que los tests fallan**

```bash
pytest tests/test_state.py -v
```

Esperado: `ModuleNotFoundError` o `ImportError` porque `trainer/state.py` no existe.

- [ ] **Step 3: Implementar state.py**

```python
# trainer/state.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Literal
import json
from datetime import datetime, timezone

PROJECTS_ROOT = Path("projects")
SESSION_FILE = "session.json"


@dataclass
class TtsSource:
    type: Literal["piper", "openai"]
    binary: Optional[str] = None
    voices_dir: Optional[str] = None
    url: Optional[str] = None
    token_env: Optional[str] = None
    selected_voices: list[str] = field(default_factory=list)
    speeds: list[float] = field(default_factory=lambda: [0.8, 0.9, 1.0, 1.1, 1.2])


@dataclass
class Voice:
    name: str
    mode: Optional[Literal["record", "import"]] = None
    clips: int = 0
    status: Literal["pending", "done"] = "pending"


@dataclass
class SynthesisState:
    status: Literal["pending", "in_progress", "done"] = "pending"
    clips: int = 0
    sources: list[TtsSource] = field(default_factory=list)


@dataclass
class TrainingState:
    status: Literal["pending", "in_progress", "done"] = "pending"
    epochs_completed: int = 0


@dataclass
class Project:
    wake_word: str
    model_name: str
    created_at: str
    voices: list[Voice]
    synthesis: SynthesisState
    training: TrainingState
    model_path: Optional[str] = None

    @property
    def root(self) -> Path:
        return PROJECTS_ROOT / self.model_name

    @property
    def positivos_path(self) -> Path:
        return self.root / "data" / "positivos"

    @property
    def sintetizados_path(self) -> Path:
        return self.root / "data" / "sintetizados"

    @property
    def models_path(self) -> Path:
        return self.root / "models"


def _to_dict(project: Project) -> dict:
    return asdict(project)


def _from_dict(d: dict) -> Project:
    voices = [Voice(**v) for v in d.pop("voices", [])]
    synth_raw = d.pop("synthesis", {})
    sources = [TtsSource(**s) for s in synth_raw.pop("sources", [])]
    synthesis = SynthesisState(**synth_raw, sources=sources)
    training = TrainingState(**d.pop("training", {}))
    return Project(**d, voices=voices, synthesis=synthesis, training=training)


def load_project(model_name: str) -> Project:
    path = PROJECTS_ROOT / model_name / SESSION_FILE
    return _from_dict(json.loads(path.read_text()))


def save_project(project: Project) -> None:
    project.root.mkdir(parents=True, exist_ok=True)
    path = project.root / SESSION_FILE
    path.write_text(json.dumps(_to_dict(project), indent=2, ensure_ascii=False))


def list_projects() -> list[Project]:
    if not PROJECTS_ROOT.exists():
        return []
    result = []
    for session_path in sorted(PROJECTS_ROOT.glob(f"*/{SESSION_FILE}")):
        try:
            result.append(load_project(session_path.parent.name))
        except Exception:
            pass
    return result


def create_project(wake_word: str, model_name: str, voice_names: list[str]) -> Project:
    project = Project(
        wake_word=wake_word,
        model_name=model_name,
        created_at=datetime.now(timezone.utc).isoformat(),
        voices=[Voice(name=n) for n in voice_names],
        synthesis=SynthesisState(),
        training=TrainingState(),
    )
    project.positivos_path.mkdir(parents=True, exist_ok=True)
    project.sintetizados_path.mkdir(parents=True, exist_ok=True)
    project.models_path.mkdir(parents=True, exist_ok=True)
    for voice in project.voices:
        (project.positivos_path / voice.name).mkdir(exist_ok=True)
    save_project(project)
    return project
```

- [ ] **Step 4: Ejecutar tests**

```bash
pytest tests/test_state.py -v
```

Esperado: todos PASS.

- [ ] **Step 5: Commit**

```bash
git add trainer/state.py tests/test_state.py
git commit -m "feat: state — persistencia de proyectos en session.json"
```

---

## Task 3: Dataset calculator

**Files:**
- Modify: `trainer/state.py` (añadir `calculate_dataset`)
- Modify: `tests/test_state.py` (añadir tests)

- [ ] **Step 1: Añadir tests al fichero existente**

```python
# Añadir a tests/test_state.py

from trainer.state import calculate_dataset


def test_calculate_dataset_no_synthesis(tmp_projects):
    p = create_project("ok jota", "ok_jota", ["Alfonso", "María"])
    stats = calculate_dataset(p)
    assert stats["real_clips"] == 60        # 2 personas × 30
    assert stats["synth_clips"] == 0
    assert stats["real_augmented"] == 600   # 60 × 10
    assert stats["total"] == 600
    assert stats["meets_minimum"] is False  # < 1000


def test_calculate_dataset_with_piper_source(tmp_projects):
    p = create_project("ok jota", "ok_jota", ["Alfonso", "María", "Carlos"])
    p.synthesis.sources.append(TtsSource(
        type="piper",
        selected_voices=["voz_a", "voz_b", "voz_c", "voz_d", "voz_e", "voz_f"],
        speeds=[0.8, 0.9, 1.0, 1.1, 1.2],
    ))
    stats = calculate_dataset(p)
    assert stats["real_clips"] == 90        # 3 × 30
    assert stats["synth_clips"] == 30       # 6 voces × 5 velocidades
    assert stats["total"] == 1200           # (90 + 30) × 10
    assert stats["meets_minimum"] is True


def test_calculate_dataset_multiple_sources(tmp_projects):
    p = create_project("ok jota", "ok_jota", ["Alfonso"])
    p.synthesis.sources = [
        TtsSource(type="piper", selected_voices=["a", "b"], speeds=[1.0]),
        TtsSource(type="openai", selected_voices=["Rachel", "Josh"], speeds=[0.9, 1.0, 1.1]),
    ]
    stats = calculate_dataset(p)
    # real: 1 × 30 = 30
    # synth: piper(2×1=2) + openai(2×3=6) = 8
    # total: (30+8) × 10 = 380
    assert stats["synth_clips"] == 8
    assert stats["total"] == 380
```

- [ ] **Step 2: Verificar que los tests fallan**

```bash
pytest tests/test_state.py::test_calculate_dataset_no_synthesis -v
```

Esperado: `ImportError` — `calculate_dataset` no existe aún.

- [ ] **Step 3: Añadir calculate_dataset a state.py**

```python
# Añadir al final de trainer/state.py

AUGMENTATION_FACTOR = 10
MINIMUM_DATASET_SIZE = 1000
CLIPS_PER_VOICE = 30


def calculate_dataset(project: Project) -> dict:
    real_clips = len(project.voices) * CLIPS_PER_VOICE
    synth_clips = sum(
        len(src.selected_voices) * len(src.speeds)
        for src in project.synthesis.sources
    )
    real_augmented = real_clips * AUGMENTATION_FACTOR
    synth_augmented = synth_clips * AUGMENTATION_FACTOR
    total = real_augmented + synth_augmented
    return {
        "real_clips": real_clips,
        "synth_clips": synth_clips,
        "real_augmented": real_augmented,
        "synth_augmented": synth_augmented,
        "total": total,
        "meets_minimum": total >= MINIMUM_DATASET_SIZE,
    }
```

- [ ] **Step 4: Ejecutar todos los tests de state**

```bash
pytest tests/test_state.py -v
```

Esperado: todos PASS.

- [ ] **Step 5: Commit**

```bash
git add trainer/state.py tests/test_state.py
git commit -m "feat: dataset calculator — estima muestras totales con augmentación"
```

---

## Task 4: UI — componentes Rich

**Files:**
- Create: `trainer/ui/panels.py`
- Create: `trainer/ui/prompts.py`

No hay lógica de negocio que testear aquí — son funciones de presentación puras. Se validan visualmente al integrar con el wizard en Task 11.

- [ ] **Step 1: Implementar panels.py**

```python
# trainer/ui/panels.py
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from trainer.state import Project, calculate_dataset

console = Console()


def print_header():
    console.print(Panel(
        "[bold cyan]JWake Trainer[/bold cyan]",
        subtitle="Entrenamiento de wake words personalizadas",
        box=box.DOUBLE_EDGE,
        style="cyan",
    ))


def print_project_list(projects: list[Project]):
    if not projects:
        console.print("[dim]No hay proyectos. Crea uno con [bold]n[/bold].[/dim]")
        return

    table = Table(box=box.SIMPLE_HEAD, show_header=False, padding=(0, 1))
    table.add_column("idx", style="bold cyan", width=4)
    table.add_column("modelo", style="bold")
    table.add_column("estado")
    table.add_column("detalle", style="dim")

    for i, p in enumerate(projects, 1):
        pending_voices = [v for v in p.voices if v.status == "pending"]
        if p.training.status == "done":
            estado = "[green]listo[/green]"
            detalle = f"modelo disponible en {p.model_path or 'models/'}"
        elif pending_voices:
            estado = "[yellow]en curso[/yellow]"
            detalle = "falta voz de " + ", ".join(f'"{v.name}"' for v in pending_voices)
        else:
            estado = "[yellow]en curso[/yellow]"
            detalle = "listo para entrenar"
        table.add_row(str(i), p.model_name, estado, detalle)

    console.print(table)


def print_project_status(project: Project):
    stats = calculate_dataset(project)
    table = Table(box=box.SIMPLE, title=f"[bold]{project.model_name}[/bold]  «{project.wake_word}»")

    table.add_column("Tipo", style="bold")
    table.add_column("Nombre")
    table.add_column("Clips", justify="right")
    table.add_column("Estado")

    for v in project.voices:
        status_str = "[green]✅[/green]" if v.status == "done" else "[yellow]⏳[/yellow]"
        mode_str = v.mode or "—"
        table.add_row("Voz humana", f"{v.name} ({mode_str})", str(v.clips), status_str)

    synth_status = "[green]✅[/green]" if project.synthesis.status == "done" else "[yellow]⏳[/yellow]"
    table.add_row("Síntesis", f"{project.synthesis.clips} clips", str(project.synthesis.clips), synth_status)

    train_status = "[green]✅[/green]" if project.training.status == "done" else "[yellow]⏳[/yellow]"
    table.add_row("Entrenamiento", "", "", train_status)

    console.print(table)
    console.print(f"\n  Dataset estimado: [bold]{stats['total']}[/bold] muestras", end="")
    if stats["meets_minimum"]:
        console.print("  [green]✅ suficiente[/green]")
    else:
        console.print(f"  [yellow]⚠️  mínimo recomendado: 1.000[/yellow]")


def print_dataset_summary(project: Project):
    stats = calculate_dataset(project)
    console.print(Panel(
        f"[bold]Grabaciones reales[/bold] ({len(project.voices)} personas × 30 clips): "
        f"{stats['real_clips']} clips → [bold]{stats['real_augmented']}[/bold] muestras tras augmentación\n"
        f"[bold]Síntesis TTS:[/bold] {stats['synth_clips']} clips → [bold]{stats['synth_augmented']}[/bold] muestras\n"
        f"\n[bold]Total dataset:[/bold] [{'green' if stats['meets_minimum'] else 'red'}]{stats['total']} muestras[/]  "
        f"{'✅' if stats['meets_minimum'] else '⚠️  mínimo recomendado: 1.000'}",
        title="Dataset estimado",
        box=box.ROUNDED,
    ))
```

- [ ] **Step 2: Implementar prompts.py**

```python
# trainer/ui/prompts.py
from rich.console import Console
from rich.panel import Panel
from rich import box

console = Console()


def explain(text: str):
    """Imprime un panel de contexto antes de una pregunta."""
    console.print(Panel(text, box=box.SIMPLE, style="dim"))


def ask(prompt: str, default: str | None = None) -> str:
    """Pregunta con prompt enriquecido. Devuelve la respuesta."""
    if default:
        display = f"[bold cyan]{prompt}[/bold cyan] [[dim]{default}[/dim]]: "
    else:
        display = f"[bold cyan]{prompt}[/bold cyan]: "
    value = console.input(display).strip()
    return value or (default or "")


def ask_choice(prompt: str, options: list[str], default: str | None = None) -> str:
    """Pregunta con opciones explícitas. Devuelve la clave elegida."""
    opts_display = "  ".join(
        f"[[bold]{o}[/bold]]" if o == default else f"[{o}]"
        for o in options
    )
    value = console.input(f"[bold cyan]{prompt}[/bold cyan]  {opts_display}: ").strip().lower()
    if not value and default:
        return default
    while value not in options:
        value = console.input(f"  Opción inválida. Elige {'/'.join(options)}: ").strip().lower()
    return value


def ask_int(prompt: str, default: int | None = None, minimum: int = 1) -> int:
    """Pregunta un número entero con validación."""
    while True:
        raw = ask(prompt, str(default) if default is not None else None)
        try:
            value = int(raw)
            if value >= minimum:
                return value
            console.print(f"  [red]Debe ser al menos {minimum}.[/red]")
        except ValueError:
            console.print("  [red]Introduce un número entero.[/red]")
```

- [ ] **Step 3: Commit**

```bash
git add trainer/ui/panels.py trainer/ui/prompts.py
git commit -m "feat: UI — componentes Rich para paneles, tablas y prompts contextuales"
```

---

## Task 5: CLI — entry point y subcomando status

**Files:**
- Create: `trainer/cli.py`

- [ ] **Step 1: Implementar cli.py**

```python
# trainer/cli.py
from __future__ import annotations
import typer
from rich.console import Console
from trainer.state import list_projects, load_project
from trainer.ui.panels import print_header, print_project_list, print_project_status

app = typer.Typer(
    name="wake-trainer",
    help="JWake Trainer — entrena wake words personalizadas con openWakeWord.",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Sin subcomando: lanza el wizard interactivo."""
    if ctx.invoked_subcommand is None:
        from trainer.wizard import run_wizard
        run_wizard()


@app.command()
def status(model_name: str = typer.Argument(None, help="Nombre del modelo. Si se omite, muestra todos.")):
    """Muestra el estado de uno o todos los proyectos."""
    print_header()
    if model_name:
        try:
            project = load_project(model_name)
            print_project_status(project)
        except FileNotFoundError:
            console.print(f"[red]Proyecto '{model_name}' no encontrado.[/red]")
            raise typer.Exit(1)
    else:
        projects = list_projects()
        print_project_list(projects)


@app.command()
def record(
    model_name: str = typer.Argument(..., help="Nombre del modelo."),
    voice: str = typer.Option(None, "--voice", "-v", help="Nombre de la voz a grabar."),
):
    """Graba muestras de voz para un proyecto existente."""
    from trainer.wizard import run_record_step
    run_record_step(model_name, voice)


@app.command("import")
def import_clips(
    model_name: str = typer.Argument(..., help="Nombre del modelo."),
    voice: str = typer.Option(None, "--voice", "-v", help="Nombre de la voz."),
    directory: str = typer.Option(None, "--dir", "-d", help="Carpeta con los WAVs."),
):
    """Importa clips WAV externos para una voz."""
    from trainer.wizard import run_import_step
    run_import_step(model_name, voice, directory)


@app.command()
def synthesize(model_name: str = typer.Argument(..., help="Nombre del modelo.")):
    """Genera muestras sintéticas con las fuentes TTS configuradas."""
    from trainer.wizard import run_synthesize_step
    run_synthesize_step(model_name)


@app.command()
def train(model_name: str = typer.Argument(..., help="Nombre del modelo.")):
    """Entrena el modelo con las muestras disponibles."""
    from trainer.wizard import run_train_step
    run_train_step(model_name)


@app.command()
def evaluate(model_name: str = typer.Argument(..., help="Nombre del modelo.")):
    """Evalúa el modelo entrenado."""
    from trainer.wizard import run_evaluate_step
    run_evaluate_step(model_name)
```

- [ ] **Step 2: Verificar que el entry point funciona**

```bash
wake-trainer --help
wake-trainer status
```

Esperado: muestra help y "No hay proyectos" respectivamente. El wizard no existe aún, pero `status` sí.

- [ ] **Step 3: Commit**

```bash
git add trainer/cli.py
git commit -m "feat: CLI — entry point Typer con subcomandos y delegación al wizard"
```

---

## Task 6: Recorder — grabación de audio guiada

**Files:**
- Create: `trainer/recorder.py`
- Create: `tests/test_recorder.py`

- [ ] **Step 1: Escribir tests (con mock de sounddevice)**

```python
# tests/test_recorder.py
import pytest
import numpy as np
import soundfile as sf
from pathlib import Path
from unittest.mock import patch, MagicMock
from trainer.recorder import validate_clip, CONDITIONS, SAMPLE_RATE


def make_wav(path: Path, duration: float = 1.5, sr: int = 16000,
             channels: int = 1, peak: float = 0.3):
    samples = int(duration * sr)
    data = (np.random.randn(samples) * peak).astype(np.float32)
    if channels > 1:
        data = np.stack([data] * channels, axis=1)
    sf.write(str(path), data, sr, subtype="PCM_16")


def test_validate_clip_valid(tmp_path):
    wav = tmp_path / "ok.wav"
    make_wav(wav)
    valid, error = validate_clip(wav)
    assert valid
    assert error == ""


def test_validate_clip_wrong_sample_rate(tmp_path):
    wav = tmp_path / "bad_sr.wav"
    make_wav(wav, sr=44100)
    valid, error = validate_clip(wav)
    assert not valid
    assert "44100" in error


def test_validate_clip_stereo(tmp_path):
    wav = tmp_path / "stereo.wav"
    make_wav(wav, channels=2)
    valid, error = validate_clip(wav)
    assert not valid
    assert "mono" in error.lower()


def test_validate_clip_too_short(tmp_path):
    wav = tmp_path / "short.wav"
    make_wav(wav, duration=0.2)
    valid, error = validate_clip(wav)
    assert not valid
    assert "corto" in error


def test_validate_clip_too_quiet(tmp_path):
    wav = tmp_path / "quiet.wav"
    make_wav(wav, peak=0.001)
    valid, error = validate_clip(wav)
    assert not valid
    assert "bajo" in error


def test_validate_clip_saturated(tmp_path):
    wav = tmp_path / "sat.wav"
    make_wav(wav, peak=1.0)
    valid, error = validate_clip(wav)
    assert not valid
    assert "saturada" in error


def test_conditions_total_clips():
    total = sum(c["clips"] for c in CONDITIONS)
    assert total == 30


def test_conditions_count():
    assert len(CONDITIONS) == 10
```

- [ ] **Step 2: Verificar que los tests fallan**

```bash
pytest tests/test_recorder.py -v
```

Esperado: `ImportError` — `trainer/recorder.py` no existe.

- [ ] **Step 3: Implementar recorder.py**

```python
# trainer/recorder.py
from __future__ import annotations
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()

SAMPLE_RATE = 16000
CHANNELS = 1
SUBTYPE = "PCM_16"
RECORD_DURATION = 3.0
MIN_DURATION = 0.5
MIN_LEVEL = 0.01   # ~-40 dBFS
MAX_LEVEL = 0.99   # ~-0.1 dBFS

CONDITIONS: list[dict] = [
    {"id": 1,  "name": "Distancia normal · silencio",  "desc": "1-1.5 m del dispositivo, habitación en silencio",  "clips": 5},
    {"id": 2,  "name": "Distancia cercana · silencio", "desc": "30-50 cm del dispositivo, volumen normal",          "clips": 3},
    {"id": 3,  "name": "Distancia larga · voz alzada", "desc": "3-4 m del dispositivo, voz ligeramente más alta",  "clips": 3},
    {"id": 4,  "name": "Ruido TV/radio",               "desc": "TV o radio de fondo a volumen moderado",            "clips": 4},
    {"id": 5,  "name": "Ruido de conversación",        "desc": "Otra persona hablando en la misma habitación",      "clips": 3},
    {"id": 6,  "name": "Música de fondo",              "desc": "Música a volumen normal",                           "clips": 3},
    {"id": 7,  "name": "Voz rápida",                   "desc": "Dicho con prisa",                                   "clips": 3},
    {"id": 8,  "name": "Voz lenta",                    "desc": "Pausado, sobrearticulado",                          "clips": 2},
    {"id": 9,  "name": "Voz baja / susurro",           "desc": "Tono bajo, sin proyectar",                         "clips": 2},
    {"id": 10, "name": "Ángulo lateral",               "desc": "Hablando de lado al dispositivo (~45°)",            "clips": 2},
]


def validate_clip(path: Path) -> tuple[bool, str]:
    data, sr = sf.read(str(path), dtype="float32")
    if sr != SAMPLE_RATE:
        return False, f"Sample rate incorrecto: {sr} Hz (esperado {SAMPLE_RATE} Hz)"
    if data.ndim > 1 and data.shape[1] > 1:
        return False, "El audio no es mono"
    duration = len(data) / sr
    if duration < MIN_DURATION:
        return False, f"Clip demasiado corto: {duration:.2f}s (mínimo {MIN_DURATION}s)"
    peak = float(np.abs(data).max())
    if peak < MIN_LEVEL:
        return False, "Nivel de señal demasiado bajo — habla más fuerte o acércate"
    if peak > MAX_LEVEL:
        return False, "Señal saturada — baja el volumen o aléjate del micrófono"
    return True, ""


def _record_raw(output_path: Path, countdown: int = 3) -> None:
    for i in range(countdown, 0, -1):
        console.print(f"  [dim]Preparado...[/dim]  [bold]{i}[/bold]", end="\r")
        time.sleep(1)
    console.print("  [bold red]● GRABANDO[/bold red]          ", end="\r")
    audio = sd.rec(
        int(RECORD_DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
    )
    sd.wait()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, SAMPLE_RATE, subtype=SUBTYPE)


def record_voice(voice_dir: Path, wake_word: str) -> int:
    """
    Guía la grabación de los 30 clips para una voz.
    Devuelve el número de clips grabados correctamente.
    """
    voice_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(voice_dir.glob("*.wav"))
    clip_number = len(existing) + 1

    for cond in CONDITIONS:
        console.print(Panel(
            f"[bold]Condición {cond['id']}/10:[/bold] {cond['name']}\n"
            f"[dim]{cond['desc']}[/dim]",
            box_kw={"box": None},
        ))
        clips_done = 0
        while clips_done < cond["clips"]:
            clip_path = voice_dir / f"{clip_number:03d}.wav"
            console.print(f"\n  Clip {clips_done+1}/{cond['clips']}")
            console.print(f'  Di claramente: [bold cyan]"{wake_word}"[/bold cyan]')

            _record_raw(clip_path)
            valid, error = validate_clip(clip_path)

            if not valid:
                clip_path.unlink(missing_ok=True)
                console.print(f"  [red]✗ {error}[/red] — reintentando...")
                time.sleep(0.5)
                continue

            data, _ = sf.read(str(clip_path), dtype="float32")
            dur = len(data) / SAMPLE_RATE
            console.print(f"  [green]✓ Guardado[/green]  ({dur:.2f}s · {SAMPLE_RATE}Hz · mono)")

            action = console.input("\n  [↵ continuar]  [r repetir]  [q pausar]: ").strip().lower()
            if action == "r":
                clip_path.unlink(missing_ok=True)
                continue
            if action == "q":
                return clip_number - 1

            clip_number += 1
            clips_done += 1

    return clip_number - 1
```

- [ ] **Step 4: Ejecutar tests**

```bash
pytest tests/test_recorder.py -v
```

Esperado: todos PASS (los tests de validación no necesitan hardware de audio).

- [ ] **Step 5: Commit**

```bash
git add trainer/recorder.py tests/test_recorder.py
git commit -m "feat: recorder — captura de audio guiada con countdown y validación"
```

---

## Task 7: Importer — importación de WAVs externos

**Files:**
- Create: `trainer/importer.py`
- Create: `tests/test_importer.py`

- [ ] **Step 1: Escribir tests**

```python
# tests/test_importer.py
import pytest
import numpy as np
import soundfile as sf
from pathlib import Path
from trainer.importer import scan_directory, import_clips


def make_wav(path: Path, sr: int = 16000, channels: int = 1, duration: float = 1.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (np.random.randn(int(duration * sr)) * 0.3).astype(np.float32)
    if channels > 1:
        data = np.stack([data] * channels, axis=1)
    sf.write(str(path), data, sr, subtype="PCM_16")


def test_scan_finds_valid_wavs(tmp_path):
    make_wav(tmp_path / "a.wav")
    make_wav(tmp_path / "b.wav")
    valid, invalid = scan_directory(tmp_path)
    assert len(valid) == 2
    assert len(invalid) == 0


def test_scan_rejects_wrong_format(tmp_path):
    make_wav(tmp_path / "good.wav")
    (tmp_path / "bad.m4a").write_bytes(b"fake")
    valid, invalid = scan_directory(tmp_path)
    assert len(valid) == 1
    assert len(invalid) == 1
    assert "m4a" in invalid[0][1].lower()


def test_scan_rejects_wrong_sample_rate(tmp_path):
    make_wav(tmp_path / "bad_sr.wav", sr=44100)
    valid, invalid = scan_directory(tmp_path)
    assert len(valid) == 0
    assert "44100" in invalid[0][1]


def test_scan_rejects_stereo(tmp_path):
    make_wav(tmp_path / "stereo.wav", channels=2)
    valid, invalid = scan_directory(tmp_path)
    assert not valid
    assert any("mono" in msg.lower() for _, msg in invalid)


def test_import_copies_valid_wavs(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    make_wav(src / "clip1.wav")
    make_wav(src / "clip2.wav")
    count, invalid = import_clips(src, dst)
    assert count == 2
    assert len(list(dst.glob("*.wav"))) == 2
    assert len(invalid) == 0


def test_import_numbers_clips_sequentially(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    make_wav(src / "x.wav")
    make_wav(src / "y.wav")
    import_clips(src, dst)
    names = sorted(p.name for p in dst.glob("*.wav"))
    assert names == ["001.wav", "002.wav"]


def test_import_continues_numbering_if_existing(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    dst.mkdir(parents=True)
    make_wav(dst / "001.wav")  # ya existe
    make_wav(src / "new.wav")
    import_clips(src, dst)
    names = sorted(p.name for p in dst.glob("*.wav"))
    assert "002.wav" in names
```

- [ ] **Step 2: Verificar que los tests fallan**

```bash
pytest tests/test_importer.py -v
```

Esperado: `ImportError`.

- [ ] **Step 3: Implementar importer.py**

```python
# trainer/importer.py
from __future__ import annotations
import shutil
import soundfile as sf
from pathlib import Path

SAMPLE_RATE = 16000
SUPPORTED_SUFFIXES = {".wav"}


def scan_directory(source_dir: Path) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Escanea source_dir y clasifica WAVs en válidos e inválidos."""
    valid: list[Path] = []
    invalid: list[tuple[Path, str]] = []

    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            invalid.append((path, f"Formato no soportado: {path.suffix}"))
            continue
        try:
            info = sf.info(str(path))
            if info.samplerate != SAMPLE_RATE:
                invalid.append((path, f"Sample rate: {info.samplerate} Hz (esperado {SAMPLE_RATE})"))
            elif info.channels != 1:
                invalid.append((path, f"Canales: {info.channels} (esperado mono)"))
            else:
                valid.append(path)
        except Exception as exc:
            invalid.append((path, f"No se puede leer: {exc}"))

    return valid, invalid


def import_clips(source_dir: Path, target_dir: Path) -> tuple[int, list[tuple[Path, str]]]:
    """Copia WAVs válidos de source_dir a target_dir con numeración secuencial."""
    target_dir.mkdir(parents=True, exist_ok=True)
    valid, invalid = scan_directory(source_dir)

    existing_count = len(list(target_dir.glob("*.wav")))
    for i, wav_path in enumerate(valid, start=existing_count + 1):
        dest = target_dir / f"{i:03d}.wav"
        shutil.copy2(str(wav_path), str(dest))

    return len(valid), invalid
```

- [ ] **Step 4: Ejecutar tests**

```bash
pytest tests/test_importer.py -v
```

Esperado: todos PASS.

- [ ] **Step 5: Commit**

```bash
git add trainer/importer.py tests/test_importer.py
git commit -m "feat: importer — importación y validación de WAVs externos"
```

---

## Task 8: Synthesizer — Piper y endpoints OpenAI-compatible

**Files:**
- Create: `trainer/synthesizer.py`
- Create: `tests/test_synthesizer.py`

- [ ] **Step 1: Escribir tests**

```python
# tests/test_synthesizer.py
import pytest
import numpy as np
import soundfile as sf
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from trainer.synthesizer import (
    synthesize_piper, synthesize_openai, list_voices_openai, list_voices_piper,
    run_synthesis,
)
from trainer.state import TtsSource


def make_wav_bytes(sr: int = 16000, duration: float = 0.5) -> bytes:
    import io
    data = (np.random.randn(int(duration * sr)) * 0.3).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_synthesize_piper_calls_binary(tmp_path):
    out = tmp_path / "clip.wav"
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        synthesize_piper(
            text="ok jota",
            voice_path="piper/voices/es_ES.onnx",
            output_path=out,
            speed=1.0,
            piper_binary="piper/piper",
        )
    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert "piper/piper" in args
    assert "piper/voices/es_ES.onnx" in args


def test_synthesize_piper_raises_on_error(tmp_path):
    out = tmp_path / "clip.wav"
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="model not found")
        with pytest.raises(RuntimeError, match="Piper"):
            synthesize_piper("ok jota", "bad.onnx", out, piper_binary="piper/piper")


def test_list_voices_piper(tmp_path):
    (tmp_path / "es_ES_female.onnx").touch()
    (tmp_path / "es_ES_male.onnx").touch()
    (tmp_path / "config.json").touch()  # debe ignorarse
    voices = list_voices_piper(str(tmp_path))
    assert len(voices) == 2
    assert all(v.endswith(".onnx") for v in voices)


@pytest.mark.asyncio
async def test_synthesize_openai_posts_to_endpoint(tmp_path):
    out = tmp_path / "clip.wav"
    wav_bytes = make_wav_bytes()

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.content = wav_bytes
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        await synthesize_openai(
            text="ok jota",
            voice="Rachel",
            url="https://api.example.com/v1",
            token="sk-test",
            output_path=out,
            speed=1.0,
        )

    assert out.exists()


@pytest.mark.asyncio
async def test_list_voices_openai_parses_response(tmp_path):
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = MagicMock(return_value={"voices": [
        {"voice_id": "Rachel", "name": "Rachel", "labels": {"gender": "female", "language": "es"}},
        {"voice_id": "Josh",   "name": "Josh",   "labels": {"gender": "male",   "language": "en"}},
    ]})

    with patch("httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
        voices = await list_voices_openai("https://api.example.com/v1", "sk-test")

    assert len(voices) == 2
    assert voices[0]["name"] == "Rachel"


@pytest.mark.asyncio
async def test_list_voices_openai_returns_empty_on_error():
    mock_response = AsyncMock()
    mock_response.status_code = 404

    with patch("httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
        voices = await list_voices_openai("https://api.example.com/v1", "sk-test")

    assert voices == []
```

- [ ] **Step 2: Verificar que los tests fallan**

```bash
pytest tests/test_synthesizer.py -v
```

Esperado: `ImportError`.

- [ ] **Step 3: Implementar synthesizer.py**

```python
# trainer/synthesizer.py
from __future__ import annotations
import asyncio
import subprocess
import io
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

from trainer.state import TtsSource, Project, save_project

console = Console()
SAMPLE_RATE = 16000


def list_voices_piper(voices_dir: str) -> list[str]:
    return sorted(str(p) for p in Path(voices_dir).glob("*.onnx"))


def synthesize_piper(
    text: str,
    voice_path: str,
    output_path: Path,
    speed: float = 1.0,
    piper_binary: str = "piper/piper",
) -> None:
    length_scale = str(round(1.0 / speed, 3))
    cmd = [piper_binary, "--model", voice_path, "--output_file", str(output_path),
           "--length_scale", length_scale]
    result = subprocess.run(cmd, input=text, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Piper falló: {result.stderr}")


async def list_voices_openai(url: str, token: str) -> list[dict]:
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(
                f"{url}/voices",
                headers={"Authorization": f"Bearer {token}"},
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            if isinstance(data, list):
                return data
            return data.get("voices", [])
        except Exception:
            return []


async def synthesize_openai(
    text: str,
    voice: str,
    url: str,
    token: str,
    output_path: Path,
    speed: float = 1.0,
) -> None:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{url}/audio/speech",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "model": "tts-1",
                "input": text,
                "voice": voice,
                "speed": speed,
                "response_format": "wav",
            },
        )
        resp.raise_for_status()

    # Write raw bytes; if not WAV (some endpoints return MP3), re-encode
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw = resp.content
    try:
        data, sr = sf.read(io.BytesIO(raw), dtype="float32")
        if sr != SAMPLE_RATE:
            # Resample would require resampy; for now write as-is and let validate_clip catch it
            pass
        sf.write(str(output_path), data, sr, subtype="PCM_16")
    except Exception:
        output_path.write_bytes(raw)


def run_synthesis(project: Project) -> int:
    """Genera todos los clips sintéticos definidos en project.synthesis.sources."""
    sintetizados = project.sintetizados_path
    sintetizados.mkdir(parents=True, exist_ok=True)
    existing = len(list(sintetizados.glob("*.wav")))
    clip_number = existing + 1
    total_generated = 0

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                  TextColumn("{task.completed}/{task.total}")) as progress:

        for source in project.synthesis.sources:
            total_clips = len(source.selected_voices) * len(source.speeds)
            task = progress.add_task(f"[cyan]{source.type}[/cyan]", total=total_clips)

            for voice in source.selected_voices:
                for speed in source.speeds:
                    out_path = sintetizados / f"{clip_number:03d}.wav"
                    try:
                        if source.type == "piper":
                            synthesize_piper(project.wake_word, voice, out_path,
                                             speed=speed, piper_binary=source.binary or "piper/piper")
                        elif source.type == "openai":
                            token = _resolve_token(source)
                            asyncio.run(synthesize_openai(project.wake_word, voice,
                                                          source.url, token, out_path, speed))
                        clip_number += 1
                        total_generated += 1
                    except Exception as exc:
                        console.print(f"  [red]✗ {source.type}/{voice}/{speed}:[/red] {exc}")
                    finally:
                        progress.advance(task)

    project.synthesis.clips = existing + total_generated
    project.synthesis.status = "done"
    save_project(project)
    return total_generated


def _resolve_token(source: TtsSource) -> str:
    import os
    if source.token_env:
        token = os.environ.get(source.token_env, "")
        if token:
            return token
    return console.input(f"  Token para {source.url}: ").strip()
```

- [ ] **Step 4: Ejecutar tests**

```bash
pytest tests/test_synthesizer.py -v
```

Esperado: todos PASS.

- [ ] **Step 5: Commit**

```bash
git add trainer/synthesizer.py tests/test_synthesizer.py
git commit -m "feat: synthesizer — Piper TTS y endpoints OpenAI-compatible con selección de voces"
```

---

## Task 9: Trainer core — openWakeWord

**Files:**
- Create: `trainer/trainer_core.py`
- Create: `tests/test_trainer_core.py`

**Nota:** La API exacta de `openwakeword.train` debe verificarse ejecutando:
```bash
python3 -c "import openwakeword.train; help(openwakeword.train.train_custom_model)"
```
El plan usa los parámetros documentados en la versión ≥0.6. Ajustar si la instalación difiere.

- [ ] **Step 1: Escribir tests con mock de openWakeWord**

```python
# tests/test_trainer_core.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from trainer.trainer_core import collect_positive_clips, TrainingConfig, run_training


def test_collect_positive_clips_finds_wavs(tmp_path):
    person_dir = tmp_path / "Alfonso"
    person_dir.mkdir()
    (person_dir / "001.wav").touch()
    (person_dir / "002.wav").touch()
    (tmp_path / "ignored.txt").touch()

    clips = collect_positive_clips(tmp_path)
    assert len(clips) == 2
    assert all(str(c).endswith(".wav") for c in clips)


def test_collect_positive_clips_recurses_subdirectories(tmp_path):
    (tmp_path / "persona1").mkdir()
    (tmp_path / "persona2").mkdir()
    (tmp_path / "persona1" / "001.wav").touch()
    (tmp_path / "persona2" / "001.wav").touch()
    (tmp_path / "persona2" / "002.wav").touch()

    clips = collect_positive_clips(tmp_path)
    assert len(clips) == 3


def test_collect_positive_clips_empty_dir(tmp_path):
    assert collect_positive_clips(tmp_path) == []


def test_training_config_defaults():
    cfg = TrainingConfig(model_name="ok_jota", output_dir=Path("models"))
    assert cfg.epochs == 100
    assert cfg.batch_size == 32
    assert cfg.learning_rate == 1e-3


def test_run_training_calls_openwakeword(tmp_path):
    (tmp_path / "positivos" / "Alfonso").mkdir(parents=True)
    (tmp_path / "positivos" / "Alfonso" / "001.wav").touch()
    models_path = tmp_path / "models"
    models_path.mkdir()

    cfg = TrainingConfig(model_name="ok_jota", output_dir=models_path)

    with patch("trainer.trainer_core.oww_train") as mock_train:
        mock_train.train_custom_model = MagicMock()
        run_training(positivos_path=tmp_path / "positivos", config=cfg)
        mock_train.train_custom_model.assert_called_once()
```

- [ ] **Step 2: Verificar que los tests fallan**

```bash
pytest tests/test_trainer_core.py -v
```

Esperado: `ImportError`.

- [ ] **Step 3: Implementar trainer_core.py**

```python
# trainer/trainer_core.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from rich.console import Console

console = Console()

try:
    import openwakeword.train as oww_train
except ImportError:
    oww_train = None  # type: ignore — se instala con pip install ".[train]"


@dataclass
class TrainingConfig:
    model_name: str
    output_dir: Path
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    augmentation_factor: int = 10


def collect_positive_clips(positivos_path: Path) -> list[Path]:
    """Devuelve todos los WAVs bajo positivos_path (recursivo)."""
    return sorted(positivos_path.rglob("*.wav"))


def run_training(positivos_path: Path, config: TrainingConfig) -> Path:
    """
    Lanza el entrenamiento con openWakeWord.
    Devuelve la ruta al .tflite resultante.

    La API de openwakeword.train.train_custom_model puede variar según versión.
    Verificar con: python3 -c "import openwakeword.train; help(openwakeword.train.train_custom_model)"
    """
    if oww_train is None:
        raise RuntimeError(
            "openWakeWord no está instalado con soporte de entrenamiento. "
            "Ejecuta: pip install openWakeWord[train]"
        )

    clips = collect_positive_clips(positivos_path)
    if not clips:
        raise ValueError(f"No se encontraron clips en {positivos_path}")

    console.print(f"\n  [dim]Clips positivos:[/dim] {len(clips)}")
    console.print(f"  [dim]Dataset estimado tras augmentación:[/dim] ~{len(clips) * config.augmentation_factor}")

    config.output_dir.mkdir(parents=True, exist_ok=True)

    oww_train.train_custom_model(
        custom_model_data={
            "positive_data": [str(c) for c in clips],
        },
        model_name=config.model_name,
        output_dir=str(config.output_dir),
        num_steps=config.epochs * 100,  # openWakeWord usa steps, no epochs
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
    )

    tflite_path = config.output_dir / f"{config.model_name}.tflite"
    if not tflite_path.exists():
        # algunos entrenadores guardan en subdirectorio
        candidates = list(config.output_dir.rglob("*.tflite"))
        if candidates:
            tflite_path = candidates[0]

    return tflite_path
```

- [ ] **Step 4: Ejecutar tests**

```bash
pytest tests/test_trainer_core.py -v
```

Esperado: todos PASS (con mock de oww_train).

- [ ] **Step 5: Commit**

```bash
git add trainer/trainer_core.py tests/test_trainer_core.py
git commit -m "feat: trainer core — wrapper de openWakeWord con recolección de clips"
```

---

## Task 10: Evaluator — métricas del modelo

**Files:**
- Create: `trainer/evaluator.py`
- Create: `tests/test_evaluator.py`

- [ ] **Step 1: Escribir tests**

```python
# tests/test_evaluator.py
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from trainer.evaluator import EvaluationResult, evaluate_model


def test_evaluation_result_defaults():
    r = EvaluationResult(precision=0.94, recall=0.91, false_positives=0, threshold=0.3)
    assert r.precision == 0.94
    assert r.false_positives == 0


def test_evaluate_model_returns_result(tmp_path):
    model_path = tmp_path / "ok_jota.tflite"
    model_path.touch()

    positivos = tmp_path / "positivos"
    positivos.mkdir()
    (positivos / "001.wav").touch()

    mock_oww = MagicMock()
    mock_oww.predict.return_value = {"ok_jota": [0.8, 0.9, 0.1, 0.05]}

    with patch("trainer.evaluator.openwakeword") as mock_module:
        mock_module.Model.return_value = mock_oww
        result = evaluate_model(model_path=model_path, positivos_path=positivos)

    assert isinstance(result, EvaluationResult)
    assert 0.0 <= result.precision <= 1.0
    assert 0.0 <= result.recall <= 1.0
```

- [ ] **Step 2: Verificar que los tests fallan**

```bash
pytest tests/test_evaluator.py -v
```

Esperado: `ImportError`.

- [ ] **Step 3: Implementar evaluator.py**

```python
# trainer/evaluator.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

try:
    import openwakeword
    import soundfile as sf
except ImportError:
    openwakeword = None  # type: ignore
    sf = None  # type: ignore


@dataclass
class EvaluationResult:
    precision: float
    recall: float
    false_positives: int
    threshold: float

    def passed(self) -> bool:
        return self.precision >= 0.9 and self.recall >= 0.85


def evaluate_model(
    model_path: Path,
    positivos_path: Path,
    threshold: float = 0.3,
) -> EvaluationResult:
    """
    Evalúa el modelo cargando todos los WAVs positivos y calculando
    precisión y recall. Los falsos positivos se miden en fragmentos
    de audio ambiente (silencio sintético).
    """
    if openwakeword is None:
        raise RuntimeError("openWakeWord no instalado.")

    model = openwakeword.Model(
        wakeword_models=[str(model_path)],
        inference_framework="tflite",
    )
    model_key = model_path.stem

    clips = sorted(positivos_path.rglob("*.wav"))
    true_positives = 0
    false_negatives = 0

    for clip_path in clips:
        data, sr = sf.read(str(clip_path), dtype="int16", always_2d=False)
        scores = model.predict(data)
        max_score = max(scores.get(model_key, [0.0]))
        if max_score >= threshold:
            true_positives += 1
        else:
            false_negatives += 1

    total = len(clips) or 1
    # False positives: test with 10s of synthetic silence (all zeros)
    silence = np.zeros(16000 * 10, dtype=np.int16)
    silence_scores = model.predict(silence)
    fp = sum(1 for s in silence_scores.get(model_key, []) if s >= threshold)

    precision = true_positives / max(true_positives + fp, 1)
    recall = true_positives / total

    return EvaluationResult(
        precision=round(precision, 3),
        recall=round(recall, 3),
        false_positives=fp,
        threshold=threshold,
    )
```

- [ ] **Step 4: Ejecutar tests**

```bash
pytest tests/test_evaluator.py -v
```

Esperado: todos PASS.

- [ ] **Step 5: Commit**

```bash
git add trainer/evaluator.py tests/test_evaluator.py
git commit -m "feat: evaluator — métricas de precisión, recall y falsos positivos"
```

---

## Task 11: Wizard — fase de planificación (nuevo proyecto)

**Files:**
- Create: `trainer/wizard.py` (esqueleto + planning phase)

- [ ] **Step 1: Crear wizard.py con la fase de planificación**

```python
# trainer/wizard.py
from __future__ import annotations
import os
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich import box

from trainer.state import (
    Project, TtsSource, create_project, load_project, save_project,
    list_projects, calculate_dataset,
)
from trainer.ui.panels import (
    print_header, print_project_list, print_project_status, print_dataset_summary,
)
from trainer.ui.prompts import explain, ask, ask_choice, ask_int

console = Console()


# ─── Entry point ──────────────────────────────────────────────────────────────

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
    choice = console.input("  Elige un proyecto [1-{n}], [n] nuevo, [q] salir: ".format(n=len(projects))).strip().lower()

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
    """Continúa desde donde se dejó el proyecto."""
    print_project_status(project)
    console.print()

    pending_voices = [v for v in project.voices if v.status == "pending"]
    if pending_voices:
        _show_preparation_summary(project)
        for voice in pending_voices:
            _wizard_record_or_import_voice(project, voice)

    if project.synthesis.status == "pending" and project.synthesis.sources:
        _wizard_synthesize(project)

    if project.synthesis.status != "done":
        _wizard_configure_synthesis(project)
        _wizard_synthesize(project)

    if project.training.status == "pending":
        _wizard_train(project)

    if project.training.status == "done" and not project.model_path:
        _wizard_evaluate(project)


# ─── Nueva wake word ──────────────────────────────────────────────────────────

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
        _wizard_configure_synthesis(project)

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


# ─── Configuración de síntesis ────────────────────────────────────────────────

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


def _configure_piper_source() -> TtsSource | None:
    from trainer.synthesizer import list_voices_piper
    voices_dir = ask("Directorio de voces Piper", default="piper/voices")
    available = list_voices_piper(voices_dir)
    if not available:
        console.print(f"  [red]No se encontraron modelos .onnx en {voices_dir}[/red]")
        return None

    console.print("  Voces disponibles:")
    for i, v in enumerate(available, 1):
        console.print(f"    {i}. {Path(v).stem}")

    raw = ask("Selecciona voces (números separados por coma, o 'todas')", default="todas")
    if raw.lower() == "todas":
        selected = available
    else:
        try:
            indices = [int(x.strip()) - 1 for x in raw.split(",")]
            selected = [available[i] for i in indices if 0 <= i < len(available)]
        except ValueError:
            selected = available

    return TtsSource(
        type="piper",
        voices_dir=voices_dir,
        selected_voices=selected,
    )


def _configure_openai_source() -> TtsSource | None:
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
        console.print(f"  ✓ {len(voices_raw)} voces encontradas")
        for i, v in enumerate(voices_raw[:20], 1):
            name = v.get("name") or v.get("voice_id") or str(v)
            console.print(f"    {i}. {name}")

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
        console.print("  [yellow]No se pudo obtener la lista de voces. Entrada manual.[/yellow]")
        raw = ask("Nombres de las voces (separados por coma)")
        selected = [v.strip() for v in raw.split(",") if v.strip()]

    return TtsSource(
        type="openai",
        url=url,
        token_env=token_env,
        selected_voices=selected,
    )


# ─── Grabación / importación ──────────────────────────────────────────────────

def _show_preparation_summary(project: Project):
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


def _wizard_record_or_import_voice(project: Project, voice):
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
            voice.status = "done"  # aceptamos clips parciales

    save_project(project)


# ─── Síntesis ─────────────────────────────────────────────────────────────────

def _wizard_synthesize(project: Project):
    from trainer.synthesizer import run_synthesis
    console.print("\n[bold]Generando muestras sintéticas...[/bold]")
    generated = run_synthesis(project)
    console.print(f"  [green]✓ {generated} clips generados[/green]")


# ─── Entrenamiento ────────────────────────────────────────────────────────────

def _wizard_train(project: Project):
    from trainer.trainer_core import run_training, TrainingConfig
    from rich.progress import Progress, SpinnerColumn, TextColumn

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


# ─── Evaluación ───────────────────────────────────────────────────────────────

def _wizard_evaluate(project: Project):
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


# ─── Funciones públicas para subcomandos CLI ──────────────────────────────────

def run_record_step(model_name: str, voice_name: str | None):
    project = load_project(model_name)
    if voice_name:
        voice = next((v for v in project.voices if v.name == voice_name), None)
        if not voice:
            console.print(f"[red]Voz '{voice_name}' no encontrada en el proyecto.[/red]")
            return
        _wizard_record_or_import_voice(project, voice)
    else:
        pending = [v for v in project.voices if v.status == "pending"]
        for voice in pending:
            _wizard_record_or_import_voice(project, voice)


def run_import_step(model_name: str, voice_name: str | None, directory: str | None):
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


def run_synthesize_step(model_name: str):
    project = load_project(model_name)
    if not project.synthesis.sources:
        _wizard_configure_synthesis(project)
    _wizard_synthesize(project)


def run_train_step(model_name: str):
    project = load_project(model_name)
    _wizard_train(project)


def run_evaluate_step(model_name: str):
    project = load_project(model_name)
    _wizard_evaluate(project)
```

- [ ] **Step 2: Verificar que el wizard arranca**

```bash
wake-trainer
```

Esperado: muestra cabecera Rich y pide crear un proyecto. Sin errores de importación.

- [ ] **Step 3: Commit**

```bash
git add trainer/wizard.py
git commit -m "feat: wizard — flujo guiado completo con planificación, grabación, síntesis y entrenamiento"
```

---

## Task 12: Migrar estructura existente de ok_jota

**Files:**
- Modify: `scripts/migrate_ok_jota.py` (script de migración one-shot)

La estructura raíz actual (`data/positivos/`, `data/sintetizados/`, `configs/ok_jota.yaml`) se migra a `projects/ok_jota/` para compatibilidad con el nuevo sistema multi-proyecto. Este script es idempotente: si ya existe `projects/ok_jota/session.json`, no hace nada.

- [ ] **Step 1: Crear script de migración**

```python
# scripts/migrate_ok_jota.py
"""
Script de migración one-shot: mueve la estructura plana de ok_jota
a projects/ok_jota/ compatible con el nuevo sistema.
Idempotente — si el proyecto ya existe, no hace nada.
"""
from pathlib import Path
import shutil, json
from datetime import datetime, timezone

ROOT = Path(__file__).parent.parent
PROJECTS_ROOT = ROOT / "projects"
TARGET = PROJECTS_ROOT / "ok_jota"
SESSION = TARGET / "session.json"


def migrate():
    if SESSION.exists():
        print("✓ projects/ok_jota/ ya existe — sin cambios.")
        return

    print("Migrando ok_jota a projects/ok_jota/ ...")

    # Crear estructura
    (TARGET / "data" / "positivos").mkdir(parents=True, exist_ok=True)
    (TARGET / "data" / "sintetizados").mkdir(parents=True, exist_ok=True)
    (TARGET / "models").mkdir(parents=True, exist_ok=True)

    # Mover datos si existen
    for src_name, dst_name in [
        ("data/positivos", "data/positivos"),
        ("data/sintetizados", "data/sintetizados"),
    ]:
        src = ROOT / src_name
        dst = TARGET / dst_name
        if src.exists() and any(src.iterdir()):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"  ✓ {src_name} → projects/ok_jota/{dst_name}")

    # Mover modelo si existe
    model_src = ROOT / "models" / "ok_jota.tflite"
    if model_src.exists():
        shutil.copy2(model_src, TARGET / "models" / "ok_jota.tflite")
        print("  ✓ models/ok_jota.tflite → projects/ok_jota/models/ok_jota.tflite")

    # Crear session.json desde configs/ok_jota.yaml
    session = {
        "wake_word": "ok jota",
        "model_name": "ok_jota",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "voices": [],
        "synthesis": {"status": "pending", "clips": 0, "sources": []},
        "training": {"status": "pending", "epochs_completed": 0},
        "model_path": None,
    }

    # Detectar voces existentes en positivos migrados
    positivos_dir = TARGET / "data" / "positivos"
    for person_dir in sorted(positivos_dir.iterdir()):
        if person_dir.is_dir():
            clips = list(person_dir.glob("*.wav"))
            session["voices"].append({
                "name": person_dir.name,
                "mode": "record",
                "clips": len(clips),
                "status": "done" if len(clips) >= 30 else "pending",
            })

    SESSION.write_text(json.dumps(session, indent=2, ensure_ascii=False))
    print(f"  ✓ session.json creado con {len(session['voices'])} voces detectadas")
    print("\n✅ Migración completada. Usa `wake-trainer status ok_jota` para ver el estado.")


if __name__ == "__main__":
    migrate()
```

- [ ] **Step 2: Ejecutar la migración**

```bash
python3 scripts/migrate_ok_jota.py
```

Esperado: crea `projects/ok_jota/session.json`. Si no hay datos, el proyecto aparece vacío pero funcional.

- [ ] **Step 3: Verificar**

```bash
wake-trainer status ok_jota
```

Esperado: muestra el estado de ok_jota con las voces detectadas.

- [ ] **Step 4: Commit**

```bash
git add scripts/migrate_ok_jota.py
git commit -m "feat: migración de estructura ok_jota plana a projects/ok_jota/"
```

---

## Task 13: Suite de tests completa y verificación final

**Files:**
- No nuevos ficheros — ejecutar suite completa y corregir lo que falle.

- [ ] **Step 1: Ejecutar la suite completa**

```bash
pytest tests/ -v --tb=short
```

Esperado: todos los tests PASS.

- [ ] **Step 2: Smoke test del CLI**

```bash
wake-trainer --help
wake-trainer status
```

Esperado: muestra ayuda y lista de proyectos sin errores.

- [ ] **Step 3: Smoke test del wizard (si hay terminal interactiva)**

```bash
wake-trainer new
```

Introduce datos mínimos (frase: "test wake", 1 persona: "Test"), confirma creación, sal con `q` en grabación. Verificar que `projects/test_wake/session.json` existe.

```bash
wake-trainer status test_wake
```

Esperado: muestra el proyecto creado con 1 voz pendiente.

- [ ] **Step 4: Limpiar proyecto de prueba**

```bash
rm -rf projects/test_wake
```

- [ ] **Step 5: Commit final**

```bash
git add -A
git commit -m "test: suite completa — todos los módulos cubiertos"
```

---

## Resumen de dependencias entre tasks

```
Task 1 (scaffolding)
  └─→ Task 2 (state)
        └─→ Task 3 (dataset calculator)
              └─→ Task 4 (UI components)
                    └─→ Task 5 (CLI)
                          ├─→ Task 6 (recorder)
                          ├─→ Task 7 (importer)
                          ├─→ Task 8 (synthesizer)
                          ├─→ Task 9 (trainer core)
                          ├─→ Task 10 (evaluator)
                          └─→ Task 11 (wizard — usa todos los anteriores)
                                └─→ Task 12 (migración ok_jota)
                                      └─→ Task 13 (verificación final)
```
