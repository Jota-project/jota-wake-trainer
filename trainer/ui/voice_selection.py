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
