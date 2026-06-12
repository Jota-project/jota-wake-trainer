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
