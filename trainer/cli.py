from __future__ import annotations
import functools
import typer
from rich.console import Console
from rich.markup import escape
from trainer.state import list_projects, load_project, ProjectNotFoundError
from trainer.ui.panels import print_header, print_project_list, print_project_status
from trainer.ui.tables import print_providers_table

app = typer.Typer(
    name="wake-trainer",
    help="JWake Trainer — entrena wake words personalizadas con openWakeWord.",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


def _friendly_errors(func):
    """
    Decorador para comandos que operan sobre un proyecto existente (recibido
    como argumento `model_name`). Convierte errores previsibles y ya
    diagnosticados (proyecto no encontrado, típicamente por pasar una ruta
    en vez del nombre) en un mensaje corto y accionable en vez de un
    traceback completo — el error en sí no es un bug del programa, así que
    no debería sonar ni parecer uno.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ProjectNotFoundError as exc:
            console.print(f"[red]{escape(str(exc))}[/red]")
            raise typer.Exit(1)
    return wrapper


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
    type_: str = typer.Option(None, "--type", "-t", metavar="TYPE", help="Tipo: piper, openai o google."),
    url: str = typer.Option(None, "--url", help="URL del endpoint (openai)."),
    token_env: str = typer.Option(None, "--token-env", help="Variable de entorno del token."),
    voices: list[str] = typer.Option([], "--voice", help="Voz (repetible)."),
    speeds: list[float] = typer.Option([], "--speed", help="Velocidad (repetible)."),
    all_voices: bool = typer.Option(
        False, "--all-voices",
        help="Descubre y añade automáticamente TODAS las voces disponibles (solo --type google por ahora), en vez de listarlas una a una con --voice.",
    ),
    lang: str = typer.Option("es", "--lang", "-l", help="Idioma a usar con --all-voices (ej: es, es-ES, es-US)."),
):
    """Añade o actualiza un provider TTS global. Sin flags: lanza wizard interactivo."""
    if not name or not type_:
        from trainer.workflows.synthesis import add_provider_interactive
        add_provider_interactive()
        return

    if type_ not in ("piper", "openai", "google"):
        console.print(f"[red]Tipo inválido: '{type_}'. Usa 'piper', 'openai' o 'google'.[/red]")
        raise typer.Exit(1)

    if all_voices:
        if type_ != "google":
            console.print("[red]--all-voices solo está soportado para --type google por ahora.[/red]")
            raise typer.Exit(1)

        import asyncio
        import os
        from trainer.synthesizer import list_voices_google

        env_name = token_env or "GOOGLE_TTS_API_KEY"
        token = os.environ.get(env_name, "")
        if not token:
            console.print(
                f"[red]No se encontró la variable de entorno '{env_name}' con la API key de Google Cloud TTS.[/red]"
            )
            raise typer.Exit(1)

        console.print(f"  Consultando voces de Google Cloud TTS (idioma: {lang})...")
        discovered = asyncio.run(list_voices_google(token, language_code=lang))
        if not discovered:
            console.print(f"[red]No se encontraron voces para '{lang}' — no se guarda el provider.[/red]")
            raise typer.Exit(1)
        voices = sorted({v.get("name") for v in discovered if v.get("name")})
        console.print(f"  Encontradas {len(voices)} voces — se añadirán todas.")

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
    n_speeds = len(provider.speeds)
    console.print(
        f"  [green]✅ Provider '{name}' guardado con {len(voices)} voces × {n_speeds} velocidades "
        f"= {len(voices) * n_speeds} clips por tanda de síntesis.[/green]"
    )


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


@providers_app.command("piper-voices")
def providers_piper_voices(
    lang: str = typer.Option("es", "--lang", "-l", help="Filtro de idioma (ej: es, en, fr)."),
    dest: str = typer.Option("piper/voices", "--dest", "-d", help="Directorio de destino."),
):
    """Lista y descarga modelos de voz Piper desde HuggingFace."""
    from trainer.piper_downloader import fetch_voices_index, download_voice
    from trainer.ui.prompts import ask
    from rich.table import Table
    from rich import box as rich_box
    from pathlib import Path

    console.print(f"  Obteniendo índice de voces Piper (idioma: {lang})...")
    try:
        voices = fetch_voices_index(lang_filter=lang)
    except Exception as exc:
        console.print(f"[red]Error al obtener el índice: {escape(str(exc))}[/red]")
        raise typer.Exit(1)

    if not voices:
        console.print(f"[yellow]No se encontraron voces para el idioma '{lang}'.[/yellow]")
        raise typer.Exit(0)

    items = list(voices.items())
    table = Table(box=rich_box.SIMPLE_HEAD)
    table.add_column("#", justify="right")
    table.add_column("Clave")
    table.add_column("Calidad")
    table.add_column("Tamaño aprox.")
    for i, (key, info) in enumerate(items, 1):
        total_bytes = sum(f.get("size_bytes", 0) for f in info["files"].values())
        size_mb = f"{total_bytes / 1_000_000:.0f} MB"
        table.add_row(str(i), key, info.get("quality", "?"), size_mb)
    console.print(table)

    raw = ask("Selecciona voces a descargar (números separados por coma, o 'todas')", default="todas")
    if raw.strip().lower() == "todas":
        selected_items = items
    else:
        try:
            indices = [int(x.strip()) - 1 for x in raw.split(",") if x.strip()]
            selected_items = [items[i] for i in indices if 0 <= i < len(items)]
        except ValueError:
            console.print("[red]Selección inválida.[/red]")
            raise typer.Exit(1)

    dest_path = Path(dest)
    for key, info in selected_items:
        file_paths = list(info["files"].keys())
        console.print(f"  Descargando [bold]{key}[/bold]...")
        try:
            downloaded = download_voice(file_paths, dest_path)
            for p in downloaded:
                console.print(f"    ✓ {p.name}")
        except Exception as exc:
            console.print(f"  [red]Error descargando {key}: {escape(str(exc))}[/red]")

    console.print(f"\n[green]Modelos guardados en {dest_path.resolve()}[/green]")


@providers_app.command("google-voices")
def providers_google_voices(
    token_env: str = typer.Option(
        "GOOGLE_TTS_API_KEY", "--token-env",
        help="Variable de entorno con la API key de Google Cloud TTS.",
    ),
    lang: str = typer.Option("es", "--lang", "-l", help="Filtro de idioma (ej: es, es-ES, es-US)."),
):
    """
    Lista las voces de Google Cloud TTS disponibles para un idioma.

    No descarga nada (a diferencia de 'piper-voices') — Google Cloud TTS es
    una API, no hay modelos que bajar. Esto solo consulta el catálogo real
    de voces vía GET /v1/voices en vez de mantener una lista fija a mano,
    que se quedaría desactualizada según Google añade voces nuevas
    (Neural2, Studio, Chirp3...). Copia los nombres que quieras usar y
    añádelos con 'wake-trainer providers add --type google --voice NOMBRE'.
    """
    import asyncio
    import os
    from trainer.synthesizer import list_voices_google
    from rich.table import Table
    from rich import box as rich_box

    token = os.environ.get(token_env, "")
    if not token:
        console.print(
            f"[red]No se encontró la variable de entorno '{token_env}' con la API key de Google Cloud TTS.[/red]"
        )
        raise typer.Exit(1)

    console.print(f"  Consultando voces de Google Cloud TTS (idioma: {lang})...")
    voices = asyncio.run(list_voices_google(token, language_code=lang))
    if not voices:
        console.print(
            f"[yellow]No se encontraron voces para '{lang}' (o la API key/permiso no es válido).[/yellow]"
        )
        raise typer.Exit(0)

    table = Table(box=rich_box.SIMPLE_HEAD)
    table.add_column("Nombre")
    table.add_column("Género")
    table.add_column("Sample rate nativo")
    for v in sorted(voices, key=lambda v: v.get("name", "")):
        names = v.get("name", "?")
        gender = v.get("ssmlGender", "?")
        sr = v.get("naturalSampleRateHertz", "?")
        table.add_row(names, gender, str(sr))
    console.print(table)
    console.print(
        "\n  [dim]Añade las que quieras con: wake-trainer providers add --name google --type google "
        f"--token-env {token_env} --voice NOMBRE1 --voice NOMBRE2 ...[/dim]"
    )


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Sin subcomando: lanza el wizard interactivo."""
    if ctx.invoked_subcommand is None:
        from trainer.wizard import run_wizard
        run_wizard()


@app.command()
@_friendly_errors
def status(model_name: str = typer.Argument(None, help="Nombre del modelo. Si se omite, muestra todos.")):
    """Muestra el estado de uno o todos los proyectos."""
    print_header()
    if model_name:
        project = load_project(model_name)
        print_project_status(project)
    else:
        projects = list_projects()
        print_project_list(projects)


@app.command()
@_friendly_errors
def record(
    model_name: str = typer.Argument(..., help="Nombre del modelo."),
    voice: str = typer.Option(None, "--voice", "-v", help="Nombre de la voz a grabar."),
):
    """Graba muestras de voz para un proyecto existente."""
    from trainer.wizard import run_record_step
    run_record_step(model_name, voice)


@app.command("import")
@_friendly_errors
def import_clips(
    model_name: str = typer.Argument(..., help="Nombre del modelo."),
    voice: str = typer.Option(None, "--voice", "-v", help="Nombre de la voz."),
    directory: str = typer.Option(None, "--dir", "-d", help="Carpeta con los WAVs."),
):
    """Importa clips WAV externos para una voz."""
    from trainer.wizard import run_import_step
    run_import_step(model_name, voice, directory)


@app.command()
@_friendly_errors
def synthesize(
    model_name: str = typer.Argument(..., help="Nombre del modelo."),
    add_provider: bool = typer.Option(
        False, "--add-provider",
        help="Añade un provider TTS nuevo a este proyecto antes de sintetizar, "
             "aunque ya tenga alguno configurado (p.ej. añadir Piper si el "
             "provider de pago se quedó sin cuota).",
    ),
):
    """Genera muestras sintéticas con las fuentes TTS configuradas."""
    from trainer.wizard import run_synthesize_step
    run_synthesize_step(model_name, add_provider=add_provider)


@app.command("synthesize-negatives")
@_friendly_errors
def synthesize_negatives(
    model_name: str = typer.Argument(..., help="Nombre del modelo."),
    count: int = typer.Option(
        120, "--count", "-c",
        help="Número de frases negativas a sintetizar (variaciones fonéticas cercanas "
             "a la wake word + frases genéricas). Ya sintetizadas se saltan (catálogo "
             "propio en data/negativos/.negative_catalog.json); solo se generan las que faltan.",
    ),
):
    """
    Genera clips negativos 'duros' sintetizados con las mismas fuentes TTS
    ya configuradas para el proyecto (variaciones cercanas a la wake word +
    frases genéricas de uso habitual). Independiente de 'train': permite
    revisar cuántos se generan y con qué fuentes antes de entrenar.
    """
    from trainer.state import load_project
    from trainer.trainer_core import _synthesize_negative_clips

    project = load_project(model_name)
    out_dir = project.negativos_path
    before = len(list(out_dir.glob("*.wav"))) if out_dir.exists() else 0

    if not any(s.selected_voices for s in project.synthesis.sources):
        console.print(
            "[red]No hay ninguna fuente TTS configurada en este proyecto "
            "(ejecuta antes 'wake-trainer synthesize --add-provider').[/red]"
        )
        raise typer.Exit(1)

    console.print(f"[bold]Sintetizando negativos para '{model_name}'[/bold] (objetivo: {count} frases)...")
    clips = _synthesize_negative_clips(project, out_dir, n_phrases=count)
    after = len(clips)
    nuevos = after - before
    console.print(f"  [green]✓ {nuevos} clip(s) nuevo(s) generado(s)[/green] — total en {out_dir}: {after}")


@app.command()
@_friendly_errors
def train(
    model_name: str = typer.Argument(..., help="Nombre del modelo."),
    full: bool = typer.Option(
        False, "--full",
        help="Descarga el dataset ACAV100M completo (~17 GB) como negativos de entrenamiento "
             "en vez del modo rápido (~200 MB). Más robusto frente a falsos positivos, pero mucho más lento.",
    ),
):
    """Entrena el modelo con las muestras disponibles."""
    from trainer.wizard import run_train_step
    run_train_step(model_name, full=full)


@app.command()
@_friendly_errors
def evaluate(model_name: str = typer.Argument(..., help="Nombre del modelo.")):
    """Evalúa el modelo entrenado."""
    from trainer.wizard import run_evaluate_step
    run_evaluate_step(model_name)


@app.command()
@_friendly_errors
def convert(model_name: str = typer.Argument(..., help="Nombre del modelo.")):
    """
    Convierte un .onnx ya entrenado a .tflite, sin repetir el entrenamiento.

    Útil cuando 'train' terminó de entrenar (el .onnx ya quedó guardado) pero
    la conversión a TFLite via onnx2tf se quedó colgada/tardó demasiado y
    hubo que matar el proceso — aquí se reintenta solo ese último paso.
    """
    from trainer.state import load_project
    from trainer.tflite_export import convert_onnx_to_tflite, TFLiteExportError

    project = load_project(model_name)
    onnx_path = project.models_path / f"{model_name}.onnx"
    tflite_path = project.models_path / f"{model_name}.tflite"

    if not onnx_path.exists():
        console.print(f"[red]No existe {onnx_path} — entrena primero con 'wake-trainer train {model_name}'.[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Convirtiendo {onnx_path.name} a TFLite...[/bold]")
    try:
        convert_onnx_to_tflite(onnx_path, tflite_path)
        console.print(f"  [green]✓ Guardado en {tflite_path}[/green]")
    except TFLiteExportError as exc:
        console.print(f"  [red]{escape(str(exc))}[/red]")
        raise typer.Exit(1)
