# trainer/wizard.py
from __future__ import annotations
from rich.console import Console
from rich.panel import Panel
from rich import box

from trainer.state import (
    Project, create_project, list_projects, calculate_dataset,
)
from trainer.ui.panels import (
    print_header, print_project_list, print_project_status, print_dataset_summary,
)
from trainer.ui.prompts import explain, ask, ask_choice, ask_int
from trainer.workflows.recording import show_preparation_summary, record_or_import_voice
from trainer.workflows.synthesis import configure_synthesis, synthesize_project
from trainer.workflows.training import train_project, evaluate_project
from trainer.workflows import (
    run_record_step, run_import_step, run_synthesize_step,
    run_train_step, run_evaluate_step,
)

console = Console()


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
    choice = console.input(
        "  Elige un proyecto [1-{n}], [n] nuevo, [q] salir: ".format(n=len(projects))
    ).strip().lower()

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
    print_project_status(project)
    console.print()

    if not project.voices:
        explain(
            "Este proyecto no tiene personas configuradas para grabar.\n"
            "Cada persona graba [bold]30 clips[/bold] en 10 condiciones distintas.\n"
            "Cuantas más personas graben, mejor reconocerá el modelo sus voces."
        )
        add = ask_choice("¿Añadir personas para grabar?", ["s", "n"], default="s")
        if add == "s":
            n = ask_int("¿Cuántas personas van a grabar?", minimum=1)
            from trainer.state import Voice, save_project
            for i in range(1, n + 1):
                name = ask(f"  Nombre de la persona {i}")
                project.voices.append(Voice(name=name or f"Persona {i}"))
            save_project(project)

    pending_voices = [v for v in project.voices if v.status == "pending"]
    if pending_voices:
        show_preparation_summary(project)
        for voice in pending_voices:
            record_or_import_voice(project, voice)

    if project.synthesis.status == "pending" and project.synthesis.sources:
        synthesize_project(project)

    if project.synthesis.status != "done":
        configure_synthesis(project)
        synthesize_project(project)

    if project.training.status == "pending":
        train_project(project)

    if project.training.status == "done" and not project.model_path:
        evaluate_project(project)


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
        configure_synthesis(project)

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


def _wizard_add_provider() -> None:
    from trainer.workflows.synthesis import add_provider_interactive
    add_provider_interactive()
