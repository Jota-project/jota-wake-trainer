# UI Interactiva con questionary — Diseño

**Fecha:** 2026-06-16
**Estado:** Aprobado

---

## Objetivo

Reemplazar todos los prompts de texto del CLI (`console.input`, selección por número) por widgets interactivos de terminal usando `questionary`: navegación con flechas, checkboxes y fuzzy search. El usuario nunca tiene que escribir números ni letras para seleccionar opciones.

---

## Decisiones de diseño

| Pregunta | Decisión |
|----------|----------|
| Librería | `questionary>=2.0` |
| Selección única (s/n, grabar/importar, proyecto) | `questionary.select` — flechas siempre |
| Selección múltiple (voces, providers) | `questionary.checkbox` |
| Listas largas (24+ voces ElevenLabs) | `questionary.checkbox` con instrucción de filtrado por texto |
| Campos de texto libre (nombres, rutas) | `questionary.text` con validación inline |
| Paneles de contexto (`explain()`) | Sin cambios — Rich sigue dibujando paneles, questionary aparece justo debajo |

---

## Archivos afectados

### `pyproject.toml`
Añadir `questionary>=2.0` a `dependencies` (runtime, no solo dev).

### `trainer/ui/prompts.py` — reescritura completa

Las cuatro funciones públicas mantienen exactamente la misma firma externa. Ningún caller necesita cambios.

```python
def ask(prompt: str, default: str | None = None) -> str:
    # questionary.text(prompt, default=default).ask()
    # Si el usuario pulsa Ctrl+C → devuelve default o ""

def ask_choice(prompt: str, options: list[str], default: str | None = None) -> str:
    # questionary.select(prompt, choices=options, default=default).ask()

def ask_int(prompt: str, default: int | None = None, minimum: int = 1) -> int:
    # questionary.text(prompt, validate=IntValidator(minimum)).ask()

def explain(text: str) -> None:
    # Sin cambios — Rich Panel con box.SIMPLE y style="dim"
```

**Manejo de Ctrl+C:** Si `questionary.ask()` devuelve `None` (el usuario interrumpió), las funciones deben manejar eso con gracia: `ask` devuelve `default or ""`, `ask_choice` devuelve `default or options[0]`, `ask_int` devuelve `default or minimum`.

### `trainer/ui/voice_selection.py` — checkbox con fuzzy search

```python
def select_openai_voices(voices_raw: list[dict]) -> list[str]:
    # choices = [Choice(title=name, value=voice_id) for ...]
    # questionary.checkbox(
    #     "Selecciona voces",
    #     choices=choices,
    #     instruction="(↑↓ navegar · espacio marcar · escribir para filtrar)"
    # ).ask()
    # Devuelve lista de voice_ids (ya era así desde el fix anterior)

def select_piper_voices(available: list[str]) -> list[str]:
    # choices = [Choice(title=Path(v).stem, value=v) for v in available]
    # questionary.checkbox("Selecciona modelos Piper", choices=choices).ask()
```

### `trainer/wizard.py` — selección de proyecto

El `console.input("Elige un proyecto...")` actual se reemplaza por `questionary.select` con entradas dinámicas:

```
❯ ok_jota — en curso
  hey_asistente — listo
  ──────────────────────
  + Nuevo proyecto
  ✕ Salir
```

### `trainer/workflows/synthesis.py` — selección de providers

El prompt numérico de providers pasa a `questionary.checkbox`:

```
? Providers TTS para este proyecto
❯ ● elevenlabs (openai)
  ○ jspeaker (openai)
  ○ + Añadir nuevo provider
─────────────────────────
espacio marcar · ↵ confirmar · sin selección = omitir síntesis
```

Si el usuario no selecciona ningún provider (marca cero opciones y confirma), se omite la síntesis — equivalente al antiguo `0. Omitir síntesis TTS`.

### `.gitignore`
Añadir `.superpowers/` (archivos de sesiones de brainstorming).

---

## Tests

Los tests existentes de `prompts.py` y `voice_selection.py` usan `monkeypatch` sobre los nombres locales del módulo. Al reescribir las funciones con questionary, los tests deben mockear `questionary.text`, `questionary.select` y `questionary.checkbox` en lugar de `console.input` y `ask`.

Los tests de `test_workflows_recording.py`, `test_workflows_training.py`, `test_workflows_synthesis.py` ya mockean `ask` y `ask_choice` por nombre de módulo — esos no cambian.

---

## Fuera de alcance

- La grabación de audio (`recorder.py`) tiene su propio flujo interactivo (`↵ continuar / r repetir / q pausar`) — se puede mejorar después pero no es parte de este cambio.
- No se cambia el estilo visual de Rich (colores, paneles, tablas) — solo la capa de input.
- No se introduce Textual ni ningún otro framework de TUI completo.
