# Interactive TUI con questionary — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reemplazar todos los prompts de texto del CLI (console.input, selección por número) con widgets interactivos de questionary: flechas, checkboxes y fuzzy search. El usuario nunca escribe números ni letras para seleccionar opciones.

**Architecture:** Las cuatro funciones públicas de `trainer/ui/prompts.py` mantienen exactamente la misma firma — cero cambios en callers. `trainer/ui/voice_selection.py` pasa a `questionary.checkbox` con `Choice(title=..., value=...)`. `trainer/wizard.py` usa `questionary.select` con `Choice`/`Separator` para el menú de proyectos. `trainer/workflows/synthesis.py` usa `questionary.checkbox` para la selección de providers.

**Tech Stack:** questionary>=2.0 (Choice, Separator, text, select, checkbox), Rich (paneles sin cambios), pytest + monkeypatch

---

## Mapa de archivos

| Archivo | Acción |
|---------|--------|
| `pyproject.toml` | Modificar — añadir `questionary>=2.0` a `dependencies` |
| `trainer/ui/prompts.py` | Reescritura completa |
| `trainer/ui/voice_selection.py` | Reescritura completa |
| `trainer/wizard.py` | Modificar — `run_wizard` reemplaza `console.input` por `questionary.select` |
| `trainer/workflows/synthesis.py` | Modificar — `configure_synthesis` reemplaza loop numérico por `questionary.checkbox` |
| `tests/test_ui_prompts.py` | Crear nuevo |
| `tests/test_ui_voice_selection.py` | Reemplazar tests |
| `tests/test_wizard_select.py` | Crear nuevo |

---

### Task 1: Añadir dependencia questionary

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Añadir questionary a dependencies**

Editar `pyproject.toml`, sección `[project]`, campo `dependencies`:

```toml
dependencies = [
    "typer>=0.12",
    "rich>=13",
    "questionary>=2.0",
    "sounddevice>=0.4.7",
    "soundfile>=0.12",
    "numpy>=1.24",
    "httpx>=0.27",
]
```

- [ ] **Step 2: Instalar y verificar**

```bash
pip install -e ".[dev]"
python -c "import questionary; print(questionary.__version__)"
```

Expected: versión ≥2.0 impresa, sin errores.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat: añadir questionary>=2.0 como dependencia runtime"
```

---

### Task 2: Reescribir trainer/ui/prompts.py

**Files:**
- Create: `tests/test_ui_prompts.py`
- Modify: `trainer/ui/prompts.py`

- [ ] **Step 1: Escribir tests que fallan**

Crear `tests/test_ui_prompts.py`:

```python
# tests/test_ui_prompts.py
from __future__ import annotations
import pytest
import questionary
from trainer.ui import prompts as pm


class MockQ:
    def __init__(self, value):
        self._v = value

    def ask(self):
        return self._v


def test_ask_devuelve_valor(monkeypatch):
    monkeypatch.setattr(questionary, "text", lambda *a, **kw: MockQ("hola"))
    assert pm.ask("¿Nombre?") == "hola"


def test_ask_devuelve_default_si_none(monkeypatch):
    monkeypatch.setattr(questionary, "text", lambda *a, **kw: MockQ(None))
    assert pm.ask("¿Nombre?", default="mundo") == "mundo"


def test_ask_devuelve_cadena_vacia_sin_default(monkeypatch):
    monkeypatch.setattr(questionary, "text", lambda *a, **kw: MockQ(None))
    assert pm.ask("¿Nombre?") == ""


def test_ask_choice_devuelve_seleccion(monkeypatch):
    monkeypatch.setattr(questionary, "select", lambda *a, **kw: MockQ("s"))
    assert pm.ask_choice("¿Crear?", ["s", "n"], default="s") == "s"


def test_ask_choice_devuelve_default_si_none(monkeypatch):
    monkeypatch.setattr(questionary, "select", lambda *a, **kw: MockQ(None))
    assert pm.ask_choice("¿Crear?", ["s", "n"], default="s") == "s"


def test_ask_choice_devuelve_primero_sin_default(monkeypatch):
    monkeypatch.setattr(questionary, "select", lambda *a, **kw: MockQ(None))
    assert pm.ask_choice("¿Tipo?", ["piper", "openai"]) == "piper"


def test_ask_int_devuelve_valor(monkeypatch):
    monkeypatch.setattr(questionary, "text", lambda *a, **kw: MockQ("3"))
    assert pm.ask_int("¿Cuántos?", minimum=1) == 3


def test_ask_int_devuelve_default_si_none(monkeypatch):
    monkeypatch.setattr(questionary, "text", lambda *a, **kw: MockQ(None))
    assert pm.ask_int("¿Cuántos?", default=2, minimum=1) == 2


def test_ask_int_devuelve_minimum_si_none_sin_default(monkeypatch):
    monkeypatch.setattr(questionary, "text", lambda *a, **kw: MockQ(None))
    assert pm.ask_int("¿Cuántos?", minimum=5) == 5
```

- [ ] **Step 2: Verificar que fallan**

```bash
pytest tests/test_ui_prompts.py -v
```

Expected: 9 fallos con `AttributeError` (questionary no está en módulo original).

- [ ] **Step 3: Reescribir trainer/ui/prompts.py**

```python
# trainer/ui/prompts.py
from __future__ import annotations
import questionary
from rich.console import Console
from rich.panel import Panel
from rich import box

console = Console()


def explain(text: str) -> None:
    console.print(Panel(text, box=box.SIMPLE, style="dim"))


def ask(prompt: str, default: str | None = None) -> str:
    result = questionary.text(prompt, default=default or "").ask()
    if result is None:
        return default or ""
    return result


def ask_choice(prompt: str, options: list[str], default: str | None = None) -> str:
    result = questionary.select(prompt, choices=options, default=default).ask()
    if result is None:
        return default or options[0]
    return result


def ask_int(prompt: str, default: int | None = None, minimum: int = 1) -> int:
    def _validate(val: str) -> bool | str:
        try:
            if int(val) >= minimum:
                return True
            return f"Debe ser al menos {minimum}"
        except ValueError:
            return "Introduce un número entero"

    default_str = str(default) if default is not None else str(minimum)
    result = questionary.text(prompt, default=default_str, validate=_validate).ask()
    if result is None:
        return default if default is not None else minimum
    try:
        return int(result)
    except ValueError:
        return default if default is not None else minimum
```

- [ ] **Step 4: Verificar que pasan**

```bash
pytest tests/test_ui_prompts.py -v
```

Expected: 9 PASSED.

- [ ] **Step 5: Suite completa**

```bash
pytest tests/ -q
```

Expected: sin nuevos fallos (los tests de workflows mockean `ask`/`ask_choice` por nombre de módulo — no cambian).

- [ ] **Step 6: Commit**

```bash
git add trainer/ui/prompts.py tests/test_ui_prompts.py
git commit -m "feat: reescribir prompts.py con questionary (select, text, checkbox)"
```

---

### Task 3: Reescribir trainer/ui/voice_selection.py

**Files:**
- Modify: `tests/test_ui_voice_selection.py`
- Modify: `trainer/ui/voice_selection.py`

- [ ] **Step 1: Reemplazar tests**

Sobrescribir `tests/test_ui_voice_selection.py` con:

```python
# tests/test_ui_voice_selection.py
from __future__ import annotations
import pytest
import questionary
from trainer.ui import voice_selection as vs_mod

VOICES_RAW = [
    {"name": "Alice", "voice_id": "alice_id"},
    {"name": "Bob",   "voice_id": "bob_id"},
    {"name": "Carol", "voice_id": "carol_id"},
]
PIPER_VOICES = ["piper/voices/es_ES.onnx", "piper/voices/en_US.onnx"]


class MockQ:
    def __init__(self, value):
        self._v = value

    def ask(self):
        return self._v


def test_select_openai_todas(monkeypatch):
    monkeypatch.setattr(questionary, "checkbox", lambda *a, **kw: MockQ(["alice_id", "bob_id", "carol_id"]))
    assert vs_mod.select_openai_voices(VOICES_RAW) == ["alice_id", "bob_id", "carol_id"]


def test_select_openai_subset(monkeypatch):
    monkeypatch.setattr(questionary, "checkbox", lambda *a, **kw: MockQ(["alice_id", "carol_id"]))
    assert vs_mod.select_openai_voices(VOICES_RAW) == ["alice_id", "carol_id"]


def test_select_openai_ctrl_c_devuelve_vacio(monkeypatch):
    monkeypatch.setattr(questionary, "checkbox", lambda *a, **kw: MockQ(None))
    assert vs_mod.select_openai_voices(VOICES_RAW) == []


def test_select_piper_subset(monkeypatch):
    monkeypatch.setattr(questionary, "checkbox", lambda *a, **kw: MockQ(["piper/voices/es_ES.onnx"]))
    assert vs_mod.select_piper_voices(PIPER_VOICES) == ["piper/voices/es_ES.onnx"]


def test_select_piper_ctrl_c_devuelve_vacio(monkeypatch):
    monkeypatch.setattr(questionary, "checkbox", lambda *a, **kw: MockQ(None))
    assert vs_mod.select_piper_voices(PIPER_VOICES) == []
```

- [ ] **Step 2: Verificar que fallan**

```bash
pytest tests/test_ui_voice_selection.py -v
```

Expected: fallos porque el módulo actual usa `ask`, no `questionary.checkbox`.

- [ ] **Step 3: Reescribir trainer/ui/voice_selection.py**

```python
# trainer/ui/voice_selection.py
from __future__ import annotations
from pathlib import Path
import questionary
from questionary import Choice


def select_openai_voices(voices_raw: list[dict]) -> list[str]:
    def get_id(v: dict) -> str:
        return v.get("voice_id") or v.get("id") or v.get("name") or str(v)

    def get_label(v: dict) -> str:
        return v.get("name") or get_id(v)

    choices = [Choice(title=get_label(v), value=get_id(v)) for v in voices_raw]
    result = questionary.checkbox(
        "Selecciona voces",
        choices=choices,
        instruction="(↑↓ navegar · espacio marcar · escribir para filtrar)",
    ).ask()
    return result if result is not None else []


def select_piper_voices(available: list[str]) -> list[str]:
    choices = [Choice(title=Path(v).stem, value=v) for v in available]
    result = questionary.checkbox(
        "Selecciona modelos Piper",
        choices=choices,
        instruction="(↑↓ navegar · espacio marcar)",
    ).ask()
    return result if result is not None else []
```

- [ ] **Step 4: Verificar que pasan**

```bash
pytest tests/test_ui_voice_selection.py -v
```

Expected: 5 PASSED.

- [ ] **Step 5: Suite completa**

```bash
pytest tests/ -q
```

Expected: sin nuevos fallos.

- [ ] **Step 6: Commit**

```bash
git add trainer/ui/voice_selection.py tests/test_ui_voice_selection.py
git commit -m "feat: voice_selection usa questionary.checkbox con Choice(title, value)"
```

---

### Task 4: Actualizar wizard.py — menú de proyectos

**Files:**
- Modify: `trainer/wizard.py`
- Create: `tests/test_wizard_select.py`

- [ ] **Step 1: Escribir test que falla**

Crear `tests/test_wizard_select.py`:

```python
# tests/test_wizard_select.py
from __future__ import annotations
import pytest
import questionary
from unittest.mock import MagicMock
import trainer.wizard as wiz_mod


class MockQ:
    def __init__(self, value):
        self._v = value

    def ask(self):
        return self._v


def _fake_project(name="test"):
    p = MagicMock()
    p.model_name = name
    p.model_path = None
    p.training.status = "pending"
    p.synthesis.status = "pending"
    p.voices = []
    return p


def test_run_wizard_elige_salir(monkeypatch):
    project = _fake_project()
    monkeypatch.setattr(wiz_mod, "list_projects", lambda: [project])
    monkeypatch.setattr(wiz_mod, "print_header", lambda: None)
    monkeypatch.setattr(wiz_mod, "print_project_list", lambda _: None)
    monkeypatch.setattr(questionary, "select", lambda *a, **kw: MockQ("__quit__"))

    wiz_mod.run_wizard()  # no debe lanzar excepción ni llamar a _wizard_continue


def test_run_wizard_elige_proyecto(monkeypatch):
    project = _fake_project()
    continued = []

    monkeypatch.setattr(wiz_mod, "list_projects", lambda: [project])
    monkeypatch.setattr(wiz_mod, "print_header", lambda: None)
    monkeypatch.setattr(wiz_mod, "print_project_list", lambda _: None)
    monkeypatch.setattr(wiz_mod, "_wizard_continue", lambda p: continued.append(p))
    monkeypatch.setattr(questionary, "select", lambda *a, **kw: MockQ(project))

    wiz_mod.run_wizard()
    assert continued == [project]
```

- [ ] **Step 2: Verificar que fallan**

```bash
pytest tests/test_wizard_select.py -v
```

Expected: fallos porque `run_wizard` usa `console.input`, no `questionary.select`.

- [ ] **Step 3: Reemplazar run_wizard en trainer/wizard.py**

Sustituir la función `run_wizard` completa:

```python
def run_wizard():
    print_header()
    projects = list_projects()

    if not projects:
        console.print("\n[dim]No hay proyectos. Creando uno nuevo...[/dim]\n")
        project = _wizard_new_project()
        if project:
            _wizard_continue(project)
        return

    import questionary
    from questionary import Choice, Separator

    def _status_label(p) -> str:
        if p.model_path:
            return "modelo listo"
        if p.training.status == "done":
            return "entrenado"
        if p.synthesis.status == "done":
            return "listo para entrenar"
        if any(v.status == "done" for v in p.voices):
            return "grabando"
        return "en curso"

    choices = [Choice(title=f"{p.model_name} — {_status_label(p)}", value=p) for p in projects]
    choices.append(Separator())
    choices.append(Choice(title="+ Nuevo proyecto", value="__new__"))
    choices.append(Choice(title="✕ Salir", value="__quit__"))

    result = questionary.select("Elige un proyecto", choices=choices).ask()

    if result is None or result == "__quit__":
        return
    if result == "__new__":
        project = _wizard_new_project()
        if project:
            _wizard_continue(project)
        return
    _wizard_continue(result)
```

- [ ] **Step 4: Verificar que pasan**

```bash
pytest tests/test_wizard_select.py -v
```

Expected: 2 PASSED.

- [ ] **Step 5: Suite completa**

```bash
pytest tests/ -q
```

Expected: sin nuevos fallos.

- [ ] **Step 6: Commit**

```bash
git add trainer/wizard.py tests/test_wizard_select.py
git commit -m "feat: wizard usa questionary.select con flechas para elegir proyecto"
```

---

### Task 5: Actualizar synthesis.py — selección de providers con checkbox

**Files:**
- Modify: `trainer/workflows/synthesis.py`
- Modify: `tests/test_workflows_synthesis.py`

- [ ] **Step 1: Añadir test que falla**

Añadir al final de `tests/test_workflows_synthesis.py`:

```python
def test_configure_synthesis_sin_seleccion_no_añade_fuentes(monkeypatch):
    import questionary
    from trainer.workflows.synthesis import configure_synthesis
    from trainer.state import Project, SynthesisConfig
    from trainer.providers import ProviderConfig

    fake_provider = ProviderConfig(name="elevenlabs", type="openai", voices=["alice"])

    monkeypatch.setattr(syn_mod, "load_providers", lambda: [fake_provider])

    class MockQ:
        def ask(self):
            return []  # sin selección → omitir síntesis

    monkeypatch.setattr(questionary, "checkbox", lambda *a, **kw: MockQ())

    project = MagicMock()
    project.synthesis.sources = []
    configure_synthesis(project)

    assert project.synthesis.sources == []
```

También añadir el import de `MagicMock` al principio del archivo si no está:

```python
from unittest.mock import AsyncMock, MagicMock, patch
```

- [ ] **Step 2: Verificar que falla**

```bash
pytest tests/test_workflows_synthesis.py::test_configure_synthesis_sin_seleccion_no_añade_fuentes -v
```

Expected: FAIL — `configure_synthesis` usa `ask`, no `questionary.checkbox`.

- [ ] **Step 3: Reemplazar la función configure_synthesis en trainer/workflows/synthesis.py**

Localizar la función `configure_synthesis` (línea ~159) y sustituirla completamente:

```python
def configure_synthesis(project: Project) -> None:
    import questionary
    from questionary import Choice, Separator
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

    if not providers:
        return

    ADD_NEW = "__add_new__"

    while True:
        choices = []
        for p in providers:
            voices_info = f"{len(p.voices)} voces" if p.voices else "voces auto"
            choices.append(Choice(title=f"{p.name} ({p.type}) — {voices_info}", value=p.name))
        choices.append(Separator())
        choices.append(Choice(title="+ Añadir nuevo provider", value=ADD_NEW))

        selected = questionary.checkbox(
            "Providers TTS para este proyecto",
            choices=choices,
            instruction="espacio marcar · ↵ confirmar · sin selección = omitir síntesis",
        ).ask()

        if selected is None:
            return

        if ADD_NEW in selected:
            add_provider_interactive()
            providers = load_providers()
            continue

        if not selected:
            return

        provider_map = {p.name: p for p in providers}
        for name in selected:
            p = provider_map.get(name)
            if p:
                source = _provider_to_tts_source(p)
                if source and source.selected_voices:
                    project.synthesis.sources.append(source)
                    save_project(project)
        break
```

- [ ] **Step 4: Verificar que pasan**

```bash
pytest tests/test_workflows_synthesis.py -v
```

Expected: 4 PASSED (los 3 anteriores + el nuevo).

- [ ] **Step 5: Suite completa**

```bash
pytest tests/ -q
```

Expected: todos los tests pasan.

- [ ] **Step 6: Commit**

```bash
git add trainer/workflows/synthesis.py tests/test_workflows_synthesis.py
git commit -m "feat: synthesis usa questionary.checkbox para seleccionar providers TTS"
```

---

## Verificación final

```bash
pytest tests/ -v
```

Expected: todos los tests pasan.

```bash
wake-trainer
```

Verificar manualmente:
- Menú inicial: flechas arriba/abajo para elegir proyecto
- Preguntas de texto: cursor activo en campo, Enter confirma
- Elecciones s/n: flechas para seleccionar
- Selección de voces TTS: checkbox con navegación y filtrado por texto
- Selección de providers: checkbox, sin marca = omitir síntesis

```bash
git log --oneline -6
```

Expected: 5 commits de este plan más el commit inicial de la dependencia.
