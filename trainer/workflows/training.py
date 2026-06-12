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
