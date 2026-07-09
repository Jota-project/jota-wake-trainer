# trainer/ui/voice_selection.py
from __future__ import annotations
from pathlib import Path
from rich.console import Console
from trainer.ui.prompts import ask

console = Console()


def select_openai_voices(voices_raw: list[dict]) -> list[str]:
    def get_id(v: dict) -> str:
        return v.get("voice_id") or v.get("id") or v.get("name") or str(v)

    def get_label(v: dict) -> str:
        return v.get("name") or get_id(v)

    console.print(f"  ✓ {len(voices_raw)} voces encontradas")
    for i, v in enumerate(voices_raw[:20], 1):
        console.print(f"    {i}. {get_label(v)}")
    raw = ask("Selecciona voces (números separados por coma, o 'todas')", default="todas")
    if raw.lower() == "todas":
        return [get_id(v) for v in voices_raw]
    try:
        indices = [int(x.strip()) - 1 for x in raw.split(",")]
        return [get_id(voices_raw[i]) for i in indices if 0 <= i < len(voices_raw)]
    except ValueError:
        return [get_id(v) for v in voices_raw]


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
