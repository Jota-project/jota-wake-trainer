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
