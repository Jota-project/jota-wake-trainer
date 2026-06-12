from __future__ import annotations
import typer
from rich.console import Console
from trainer.state import list_projects, load_project
from trainer.ui.panels import print_header, print_project_list, print_project_status
from trainer.ui.tables import print_providers_table

app = typer.Typer(
    name="wake-trainer",
    help="JWake Trainer — entrena wake words personalizadas con openWakeWord.",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


# ─── Subgrupo providers ────────────────────────────────────────────────────────

providers_app = typer.Typer(
    name="providers",
    help="Gestiona providers TTS globales preconfigurados.",
    add_completion=False,
)
app.add_typer(providers_app)


@providers_app.command("list")
def providers_list():
    """Lista todos los providers TTS configurados."""
    from trainer.providers import load_providers
    print_providers_table(load_providers())


@providers_app.command("add")
def providers_add(
    name: str = typer.Option(None, "--name", "-n", help="Nombre del provider."),
    type_: str = typer.Option(None, "--type", "-t", metavar="TYPE", help="Tipo: piper u openai."),
    url: str = typer.Option(None, "--url", help="URL del endpoint (openai)."),
    token_env: str = typer.Option(None, "--token-env", help="Variable de entorno del token."),
    voices: list[str] = typer.Option([], "--voice", help="Voz (repetible)."),
    speeds: list[float] = typer.Option([], "--speed", help="Velocidad (repetible)."),
):
    """Añade o actualiza un provider TTS global. Sin flags: lanza wizard interactivo."""
    if not name or not type_:
        from trainer.workflows.synthesis import add_provider_interactive
        add_provider_interactive()
        return

    if type_ not in ("piper", "openai"):
        console.print(f"[red]Tipo inválido: '{type_}'. Usa 'piper' u 'openai'.[/red]")
        raise typer.Exit(1)

    from trainer.providers import ProviderConfig, add_or_update_provider
    provider = ProviderConfig(
        name=name,
        type=type_,
        url=url,
        token_env=token_env,
        voices=voices,
        speeds=speeds if speeds else [0.8, 0.9, 1.0, 1.1, 1.2],
    )
    add_or_update_provider(provider)
    console.print(f"  [green]✅ Provider '{name}' guardado.[/green]")


@providers_app.command("remove")
def providers_remove(
    name: str = typer.Argument(..., help="Nombre del provider a eliminar."),
):
    """Elimina un provider TTS global."""
    from trainer.providers import remove_provider
    if not remove_provider(name):
        console.print(f"[red]Provider '{name}' no encontrado.[/red]")
        raise typer.Exit(1)
    console.print(f"  [green]✅ Provider '{name}' eliminado.[/green]")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Sin subcomando: lanza el wizard interactivo."""
    if ctx.invoked_subcommand is None:
        from trainer.wizard import run_wizard
        run_wizard()


@app.command()
def status(model_name: str = typer.Argument(None, help="Nombre del modelo. Si se omite, muestra todos.")):
    """Muestra el estado de uno o todos los proyectos."""
    print_header()
    if model_name:
        try:
            project = load_project(model_name)
            print_project_status(project)
        except FileNotFoundError:
            console.print(f"[red]Proyecto '{model_name}' no encontrado.[/red]")
            raise typer.Exit(1)
    else:
        projects = list_projects()
        print_project_list(projects)


@app.command()
def record(
    model_name: str = typer.Argument(..., help="Nombre del modelo."),
    voice: str = typer.Option(None, "--voice", "-v", help="Nombre de la voz a grabar."),
):
    """Graba muestras de voz para un proyecto existente."""
    from trainer.wizard import run_record_step
    run_record_step(model_name, voice)


@app.command("import")
def import_clips(
    model_name: str = typer.Argument(..., help="Nombre del modelo."),
    voice: str = typer.Option(None, "--voice", "-v", help="Nombre de la voz."),
    directory: str = typer.Option(None, "--dir", "-d", help="Carpeta con los WAVs."),
):
    """Importa clips WAV externos para una voz."""
    from trainer.wizard import run_import_step
    run_import_step(model_name, voice, directory)


@app.command()
def synthesize(model_name: str = typer.Argument(..., help="Nombre del modelo.")):
    """Genera muestras sintéticas con las fuentes TTS configuradas."""
    from trainer.wizard import run_synthesize_step
    run_synthesize_step(model_name)


@app.command()
def train(model_name: str = typer.Argument(..., help="Nombre del modelo.")):
    """Entrena el modelo con las muestras disponibles."""
    from trainer.wizard import run_train_step
    run_train_step(model_name)


@app.command()
def evaluate(model_name: str = typer.Argument(..., help="Nombre del modelo.")):
    """Evalúa el modelo entrenado."""
    from trainer.wizard import run_evaluate_step
    run_evaluate_step(model_name)
