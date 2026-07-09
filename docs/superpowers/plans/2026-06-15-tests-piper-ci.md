# Tests de workflows, descarga de modelos Piper y CI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Añadir tests para los módulos de workflows, un comando `wake-trainer providers piper-voices` para descargar modelos Piper desde HuggingFace, y un workflow de GitHub Actions que ejecute pytest en cada push.

**Architecture:** Los tests de workflows usan monkeypatch de las dependencias IO (ask, save_project, recorder, importer). El descargador de Piper vive en `trainer/piper_downloader.py` con funciones puras (fetch + download), el flujo interactivo en el comando CLI. La CI es un workflow YAML mínimo que solo instala dependencias de dev (sin openwakeword[train]).

**Tech Stack:** Python 3.11+, pytest, monkeypatch, unittest.mock.patch/AsyncMock, httpx, typer, GitHub Actions

---

## Estructura de ficheros

### Nuevos
- `trainer/piper_downloader.py` — `fetch_voices_index(lang_filter)` y `download_voice(file_paths, dest_dir)` sin lógica interactiva
- `tests/test_workflows_recording.py` — 4 tests para `record_or_import_voice`
- `tests/test_workflows_training.py` — 4 tests para `train_project` y `evaluate_project`
- `tests/test_workflows_synthesis.py` — 3 tests para `_provider_to_tts_source`
- `tests/test_piper_downloader.py` — 3 tests para fetch y download
- `.github/workflows/tests.yml` — CI simple

### Modificados
- `trainer/cli.py` — añadir `@providers_app.command("piper-voices")`

---

## Task 1: Tests de trainer/workflows/recording.py

**Files:**
- Create: `tests/test_workflows_recording.py`

- [ ] **Step 1: Escribir los tests**

```python
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
```

- [ ] **Step 2: Verificar que los tests fallan**

```bash
cd /Users/alfonsogarre/Workspace/jota-wake-trainer && python3 -m pytest tests/test_workflows_recording.py -v
```

Esperado: 4 FAILED con `ModuleNotFoundError` (el fichero no existe aún) → en realidad los tests pasarán de inmediato porque el módulo ya existe. Si pasan todos, saltar al Step 4.

- [ ] **Step 3: Verificar suite completa sin regresiones**

```bash
cd /Users/alfonsogarre/Workspace/jota-wake-trainer && python3 -m pytest tests/ -q --tb=short
```

Esperado: 65 PASSED (61 previos + 4 nuevos).

- [ ] **Step 4: Commit**

```bash
cd /Users/alfonsogarre/Workspace/jota-wake-trainer && git add tests/test_workflows_recording.py && git commit -m "test: añadir tests para workflows/recording.py"
```

---

## Task 2: Tests de trainer/workflows/training.py

**Files:**
- Create: `tests/test_workflows_training.py`

- [ ] **Step 1: Escribir los tests**

```python
# tests/test_workflows_training.py
from __future__ import annotations
from pathlib import Path
import pytest
import trainer.trainer_core
import trainer.workflows.training as train_mod
from trainer.state import Project, Voice, SynthesisState, TrainingState


def _make_project(model_path: str | None = None):
    return Project(
        wake_word="ok jota",
        model_name="ok_jota",
        created_at="2026-01-01T00:00:00+00:00",
        voices=[Voice(name="Alfonso")],
        synthesis=SynthesisState(),
        training=TrainingState(),
        model_path=model_path,
    )


def test_train_project_aborts_when_n(monkeypatch):
    monkeypatch.setattr(train_mod, "calculate_dataset", lambda p: {"total": 500})
    monkeypatch.setattr(train_mod, "ask_choice", lambda *a, **kw: "n")
    monkeypatch.setattr(train_mod, "save_project", lambda p: None)

    project = _make_project()
    train_mod.train_project(project)

    assert project.training.status == "pending"


def test_train_project_sets_done_on_success(monkeypatch):
    monkeypatch.setattr(train_mod, "calculate_dataset", lambda p: {"total": 500})
    monkeypatch.setattr(train_mod, "ask_choice", lambda *a, **kw: "s")
    monkeypatch.setattr(train_mod, "save_project", lambda p: None)
    monkeypatch.setattr(
        trainer.trainer_core, "run_training",
        lambda path, cfg: Path("models/ok_jota.tflite"),
    )

    project = _make_project()
    train_mod.train_project(project)

    assert project.training.status == "done"
    assert project.model_path == "models/ok_jota.tflite"


def test_train_project_sets_pending_on_failure(monkeypatch):
    monkeypatch.setattr(train_mod, "calculate_dataset", lambda p: {"total": 500})
    monkeypatch.setattr(train_mod, "ask_choice", lambda *a, **kw: "s")
    monkeypatch.setattr(train_mod, "save_project", lambda p: None)
    monkeypatch.setattr(
        trainer.trainer_core, "run_training",
        lambda path, cfg: (_ for _ in ()).throw(RuntimeError("sin GPU")),
    )

    project = _make_project()
    train_mod.train_project(project)

    assert project.training.status == "pending"


def test_evaluate_project_skips_when_no_model_path(monkeypatch):
    monkeypatch.setattr(train_mod, "save_project", lambda p: None)

    project = _make_project(model_path=None)
    train_mod.evaluate_project(project)  # no debe lanzar excepción
```

- [ ] **Step 2: Verificar que los tests pasan**

```bash
cd /Users/alfonsogarre/Workspace/jota-wake-trainer && python3 -m pytest tests/test_workflows_training.py -v
```

Esperado: 4 PASSED.

- [ ] **Step 3: Suite completa sin regresiones**

```bash
cd /Users/alfonsogarre/Workspace/jota-wake-trainer && python3 -m pytest tests/ -q --tb=short
```

Esperado: 69 PASSED.

- [ ] **Step 4: Commit**

```bash
cd /Users/alfonsogarre/Workspace/jota-wake-trainer && git add tests/test_workflows_training.py && git commit -m "test: añadir tests para workflows/training.py"
```

---

## Task 3: Tests de trainer/workflows/synthesis.py

**Files:**
- Create: `tests/test_workflows_synthesis.py`

- [ ] **Step 1: Escribir los tests**

```python
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
```

- [ ] **Step 2: Verificar que los tests pasan**

```bash
cd /Users/alfonsogarre/Workspace/jota-wake-trainer && python3 -m pytest tests/test_workflows_synthesis.py -v
```

Esperado: 3 PASSED.

- [ ] **Step 3: Suite completa sin regresiones**

```bash
cd /Users/alfonsogarre/Workspace/jota-wake-trainer && python3 -m pytest tests/ -q --tb=short
```

Esperado: 72 PASSED.

- [ ] **Step 4: Commit**

```bash
cd /Users/alfonsogarre/Workspace/jota-wake-trainer && git add tests/test_workflows_synthesis.py && git commit -m "test: añadir tests para workflows/synthesis.py"
```

---

## Task 4: trainer/piper_downloader.py + tests

**Files:**
- Create: `trainer/piper_downloader.py`
- Create: `tests/test_piper_downloader.py`

- [ ] **Step 1: Escribir los tests que fallan**

```python
# tests/test_piper_downloader.py
from __future__ import annotations
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import trainer.piper_downloader as dl_mod


FAKE_VOICES = {
    "es_ES/davefx/medium": {
        "name": "davefx",
        "language": {"code": "es_ES", "name_english": "Spanish (Spain)"},
        "quality": "medium",
        "files": {
            "es_ES/davefx/medium/es_ES-davefx-medium.onnx": {"size_bytes": 61000000},
            "es_ES/davefx/medium/es_ES-davefx-medium.onnx.json": {"size_bytes": 4000},
        },
    },
    "en_US/lessac/medium": {
        "name": "lessac",
        "language": {"code": "en_US", "name_english": "English (US)"},
        "quality": "medium",
        "files": {
            "en_US/lessac/medium/en_US-lessac-medium.onnx": {"size_bytes": 63000000},
            "en_US/lessac/medium/en_US-lessac-medium.onnx.json": {"size_bytes": 4500},
        },
    },
}


def test_fetch_voices_index_filters_by_lang():
    mock_resp = MagicMock()
    mock_resp.json.return_value = FAKE_VOICES

    with patch("httpx.get", return_value=mock_resp):
        result = dl_mod.fetch_voices_index(lang_filter="es")

    assert "es_ES/davefx/medium" in result
    assert "en_US/lessac/medium" not in result


def test_fetch_voices_index_no_filter_returns_all():
    mock_resp = MagicMock()
    mock_resp.json.return_value = FAKE_VOICES

    with patch("httpx.get", return_value=mock_resp):
        result = dl_mod.fetch_voices_index(lang_filter=None)

    assert len(result) == 2


def test_download_voice_creates_files(tmp_path):
    file_paths = list(FAKE_VOICES["es_ES/davefx/medium"]["files"].keys())

    fake_content = b"fake-model-bytes"
    mock_response = MagicMock()
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)
    mock_response.iter_bytes.return_value = [fake_content]

    with patch("httpx.stream", return_value=mock_response):
        downloaded = dl_mod.download_voice(file_paths, tmp_path)

    assert len(downloaded) == 2
    for p in downloaded:
        assert p.exists()
        assert p.read_bytes() == fake_content
```

- [ ] **Step 2: Verificar que los tests fallan**

```bash
cd /Users/alfonsogarre/Workspace/jota-wake-trainer && python3 -m pytest tests/test_piper_downloader.py -v
```

Esperado: 3 FAILED con `ModuleNotFoundError: No module named 'trainer.piper_downloader'`.

- [ ] **Step 3: Implementar `trainer/piper_downloader.py`**

```python
# trainer/piper_downloader.py
from __future__ import annotations
from pathlib import Path
import httpx

VOICES_INDEX_URL = "https://huggingface.co/rhasspy/piper-voices/raw/main/voices.json"
HF_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"


def fetch_voices_index(lang_filter: str | None = None) -> dict[str, dict]:
    resp = httpx.get(VOICES_INDEX_URL, timeout=15, follow_redirects=True)
    resp.raise_for_status()
    voices = resp.json()
    if lang_filter:
        voices = {
            k: v for k, v in voices.items()
            if v["language"]["code"].startswith(lang_filter)
        }
    return voices


def download_voice(file_paths: list[str], dest_dir: Path) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for file_path in file_paths:
        filename = Path(file_path).name
        url = HF_BASE_URL + file_path
        dest = dest_dir / filename
        with httpx.stream("GET", url, timeout=120, follow_redirects=True) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_bytes(chunk_size=8192):
                    f.write(chunk)
        downloaded.append(dest)
    return downloaded
```

- [ ] **Step 4: Verificar que los 3 tests pasan**

```bash
cd /Users/alfonsogarre/Workspace/jota-wake-trainer && python3 -m pytest tests/test_piper_downloader.py -v
```

Esperado: 3 PASSED.

- [ ] **Step 5: Suite completa sin regresiones**

```bash
cd /Users/alfonsogarre/Workspace/jota-wake-trainer && python3 -m pytest tests/ -q --tb=short
```

Esperado: 75 PASSED.

- [ ] **Step 6: Commit**

```bash
cd /Users/alfonsogarre/Workspace/jota-wake-trainer && git add trainer/piper_downloader.py tests/test_piper_downloader.py && git commit -m "feat: añadir piper_downloader con fetch_voices_index y download_voice"
```

---

## Task 5: Comando CLI `wake-trainer providers piper-voices`

**Files:**
- Modify: `trainer/cli.py`

El comando lista los modelos disponibles filtrados por idioma y deja al usuario seleccionar cuáles descargar.

- [ ] **Step 1: Leer el estado actual de cli.py**

Leer `trainer/cli.py` para localizar el bloque de `providers_app` donde insertar el nuevo comando (después de `providers_remove`).

- [ ] **Step 2: Añadir el comando**

Insertar justo después de la función `providers_remove` y antes del callback `@app.callback`:

```python
@providers_app.command("piper-voices")
def providers_piper_voices(
    lang: str = typer.Option("es", "--lang", "-l", help="Filtro de idioma (ej: es, en, fr)."),
    dest: str = typer.Option("piper/voices", "--dest", "-d", help="Directorio de destino."),
):
    """Lista y descarga modelos de voz Piper desde HuggingFace."""
    from trainer.piper_downloader import fetch_voices_index, download_voice
    from trainer.ui.prompts import ask
    from rich.table import Table
    from rich import box as rich_box
    from pathlib import Path

    console.print(f"  Obteniendo índice de voces Piper (idioma: {lang})...")
    try:
        voices = fetch_voices_index(lang_filter=lang)
    except Exception as exc:
        console.print(f"[red]Error al obtener el índice: {exc}[/red]")
        raise typer.Exit(1)

    if not voices:
        console.print(f"[yellow]No se encontraron voces para el idioma '{lang}'.[/yellow]")
        raise typer.Exit(0)

    items = list(voices.items())
    table = Table(box=rich_box.SIMPLE_HEAD)
    table.add_column("#", justify="right")
    table.add_column("Clave")
    table.add_column("Calidad")
    table.add_column("Tamaño aprox.")
    for i, (key, info) in enumerate(items, 1):
        total_bytes = sum(f.get("size_bytes", 0) for f in info["files"].values())
        size_mb = f"{total_bytes / 1_000_000:.0f} MB"
        table.add_row(str(i), key, info.get("quality", "?"), size_mb)
    console.print(table)

    raw = ask("Selecciona voces a descargar (números separados por coma, o 'todas')", default="todas")
    if raw.strip().lower() == "todas":
        selected_items = items
    else:
        try:
            indices = [int(x.strip()) - 1 for x in raw.split(",") if x.strip()]
            selected_items = [items[i] for i in indices if 0 <= i < len(items)]
        except ValueError:
            console.print("[red]Selección inválida.[/red]")
            raise typer.Exit(1)

    dest_path = Path(dest)
    for key, info in selected_items:
        file_paths = list(info["files"].keys())
        console.print(f"  Descargando [bold]{key}[/bold]...")
        try:
            downloaded = download_voice(file_paths, dest_path)
            for p in downloaded:
                console.print(f"    ✓ {p.name}")
        except Exception as exc:
            console.print(f"  [red]Error descargando {key}: {exc}[/red]")

    console.print(f"\n[green]Modelos guardados en {dest_path.resolve()}[/green]")
```

- [ ] **Step 3: Verificar suite completa sin regresiones**

```bash
cd /Users/alfonsogarre/Workspace/jota-wake-trainer && python3 -m pytest tests/ -q --tb=short
```

Esperado: 75 PASSED.

- [ ] **Step 4: Verificar que el comando aparece en el help**

```bash
cd /Users/alfonsogarre/Workspace/jota-wake-trainer && python3 -m trainer.cli providers --help
```

Esperado: `piper-voices` aparece en la lista de comandos.

- [ ] **Step 5: Commit**

```bash
cd /Users/alfonsogarre/Workspace/jota-wake-trainer && git add trainer/cli.py && git commit -m "feat: comando providers piper-voices para descargar modelos Piper"
```

---

## Task 6: GitHub Actions CI

**Files:**
- Create: `.github/workflows/tests.yml`

- [ ] **Step 1: Crear el directorio y el fichero**

```yaml
# .github/workflows/tests.yml
name: Tests

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Instalar dependencias
        run: pip install -e ".[dev]"

      - name: Ejecutar tests
        run: pytest tests/ -q
```

Nota: solo instala `.[dev]` (pytest + httpx), NO `.[train]` (openwakeword) porque tarda varios minutos. Los tests que dependen de openwakeword ya lo mockean.

- [ ] **Step 2: Verificar que el fichero es YAML válido**

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/tests.yml'))" && echo "YAML válido"
```

Esperado: `YAML válido`.

- [ ] **Step 3: Commit y push**

```bash
cd /Users/alfonsogarre/Workspace/jota-wake-trainer && git add .github/workflows/tests.yml && git commit -m "ci: GitHub Actions — ejecutar pytest en cada push" && git push origin main
```

Esperado: el workflow aparece en la pestaña Actions del repo de GitHub al cabo de unos segundos.

---

## Resultado final

| Fichero | Cambio |
|---------|--------|
| `tests/test_workflows_recording.py` | Nuevo — 4 tests |
| `tests/test_workflows_training.py` | Nuevo — 4 tests |
| `tests/test_workflows_synthesis.py` | Nuevo — 3 tests |
| `trainer/piper_downloader.py` | Nuevo |
| `tests/test_piper_downloader.py` | Nuevo — 3 tests |
| `trainer/cli.py` | `providers piper-voices` añadido |
| `.github/workflows/tests.yml` | Nuevo |

**Tests totales:** 75 (61 previos + 14 nuevos).
