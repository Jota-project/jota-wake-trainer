# trainer/workflows/training.py
from __future__ import annotations
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markup import escape
from rich import box

from trainer.state import Project, save_project, load_project, calculate_dataset
from trainer.ui.prompts import ask_choice

console = Console()


def train_project(project: Project, negative_mode: str = "quick") -> None:
    from trainer.trainer_core import run_training, TrainingConfig

    console.print(Panel("[bold]Entrenamiento[/bold]", box=box.ROUNDED))
    stats = calculate_dataset(project)
    console.print(f"  Dataset estimado: [bold]{stats['total']}[/bold] muestras")
    if negative_mode == "quick":
        console.print(
            "  [dim]Negativos: modo rápido (~200 MB, reutiliza el set de validación). "
            "Usa --full para un modelo más robusto (~17 GB).[/dim]"
        )
    else:
        console.print("  [dim]Negativos: modo full (~17 GB, ACAV100M — puede tardar horas en descargar).[/dim]")

    confirm = ask_choice("¿Iniciar entrenamiento?", ["s", "n"], default="s")
    if confirm != "s":
        return

    project.training.status = "in_progress"
    save_project(project)

    cfg = TrainingConfig(
        model_name=project.model_name,
        output_dir=project.models_path,
        negative_mode=negative_mode,
    )

    try:
        tflite_path = run_training(project, cfg)
        project.training.status = "done"
        project.model_path = str(tflite_path)
        save_project(project)
        console.print(f"\n  [green]✓ Modelo guardado en {tflite_path}[/green]")
    except Exception as exc:
        project.training.status = "pending"
        save_project(project)
        console.print(f"\n  [red]✗ Entrenamiento fallido: {escape(str(exc))}[/red]")


def evaluate_project(project: Project) -> None:
    from trainer.evaluator import evaluate_model
    from rich.table import Table
    if not project.model_path:
        console.print("[yellow]No hay modelo entrenado para evaluar.[/yellow]")
        return

    console.print(Panel("[bold]Evaluación del modelo[/bold]", box=box.ROUNDED))
    result = evaluate_model(
        model_path=Path(project.model_path),
        positivos_path=project.positivos_path,
        negativos_path=project.negativos_path,
    )

    perf = result.performance
    runtime_note = ""
    if perf and Path(project.model_path).suffix == ".tflite" and perf.runtime_used == "onnx":
        runtime_note = (
            "  [yellow]⚠️  Se pidió TFLite pero no se encontró 'tflite_runtime' instalado — "
            "esta evaluación se hizo con ONNX (mismo modelo, mismos pesos, solo distinto "
            "motor de inferencia: el resultado es igual de válido). Si tu despliegue real "
            "SÍ usa tflite_runtime, instálalo aquí también con 'pip install tflite-runtime' "
            "para evaluar exactamente con el mismo runtime que producción.[/yellow]\n"
        )

    console.print(f"""
  Modelo:            [bold]{project.model_path}[/bold]
  Runtime usado:     [bold]{perf.runtime_used if perf else '?'}[/bold]
  Precisión:         [bold]{result.precision*100:.1f}%[/bold]  (de lo que detecta, % que es de verdad la wake word)
  Recall/Sensib.:    [bold]{result.recall*100:.1f}%[/bold]  (de los positivos reales, % que detecta)
  Falsos positivos:  [bold]{result.false_positives}[/bold] ({escape(result.false_positive_note)})
  Threshold usado:   [bold]{result.threshold}[/bold]
    """)
    if runtime_note:
        console.print(runtime_note)

    if perf:
        console.print(
            f"  [dim]Rendimiento: {perf.total_audio_seconds}s de audio evaluados en "
            f"{perf.total_processing_seconds}s de cómputo "
            f"(~{perf.real_time_factor}x más rápido que tiempo real, "
            f"~{perf.avg_ms_per_clip} ms/clip de media).[/dim]\n"
        )

    if result.threshold_sweep:
        table = Table(title="Sensibilidad por threshold", show_lines=False)
        table.add_column("Threshold", justify="right")
        table.add_column("Recall (sensib.)", justify="right")
        table.add_column("Precisión", justify="right")
        table.add_column("Falsos +", justify="right")
        table.add_column("Accuracy", justify="right")
        for stats in result.threshold_sweep:
            marker = " ← usado" if abs(stats.threshold - result.threshold) < 1e-9 else ""
            table.add_row(
                f"{stats.threshold}{marker}",
                f"{stats.recall*100:.0f}%",
                f"{stats.precision*100:.0f}%",
                str(stats.false_positives),
                f"{stats.accuracy*100:.0f}%",
            )
        console.print(table)
        console.print(
            "  [dim]Threshold más bajo = más sensible (detecta más, pero más falsos positivos). "
            "Más alto = más estricto (menos falsos positivos, pero puede no detectarte a ti).[/dim]\n"
        )

    if result.passed():
        console.print("  [green]✅ El modelo supera los umbrales mínimos.[/green]")
    else:
        console.print("  [yellow]⚠️  Considera grabar más muestras y reentrenar.[/yellow]")


def run_train_step(model_name: str, full: bool = False) -> None:
    project = load_project(model_name)
    train_project(project, negative_mode="full" if full else "quick")


def run_evaluate_step(model_name: str) -> None:
    project = load_project(model_name)
    evaluate_project(project)
