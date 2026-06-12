# trainer/wizard.py
from __future__ import annotations
import os
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich import box

from trainer.state import (
    Project, TtsSource, create_project, load_project, save_project,
    list_projects, calculate_dataset,
)
from trainer.ui.panels import (
    print_header, print_project_list, print_project_status, print_dataset_summary,
)
from trainer.ui.prompts import explain, ask, ask_choice, ask_int

console = Console()


# ─── Entry point ──────────────────────────────────────────────────────────────

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
    choice = console.input("  Elige un proyecto [1-{n}], [n] nuevo, [q] salir: ".format(n=len(projects))).strip().lower()

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
    """Continúa desde donde se dejó el proyecto."""
    print_project_status(project)
    console.print()

    pending_voices = [v for v in project.voices if v.status == "pending"]
    if pending_voices:
        _show_preparation_summary(project)
        for voice in pending_voices:
            _wizard_record_or_import_voice(project, voice)

    if project.synthesis.status == "pending" and project.synthesis.sources:
        _wizard_synthesize(project)

    if project.synthesis.status != "done":
        _wizard_configure_synthesis(project)
        _wizard_synthesize(project)

    if project.training.status == "pending":
        _wizard_train(project)

    if project.training.status == "done" and not project.model_path:
        _wizard_evaluate(project)


# ─── Nueva wake word ──────────────────────────────────────────────────────────

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
        _wizard_configure_synthesis(project)

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


# ─── Configuración de síntesis ────────────────────────────────────────────────

def _wizard_configure_synthesis(project: Project):
    from trainer.providers import load_providers

    providers = load_providers()

    if not providers:
        add_now = ask_choice(
            "No hay providers TTS configurados. ¿Configurar uno ahora?",
            ["s", "n"],
            default="s",
        )
        if add_now == "s":
            _wizard_add_provider()
            providers = load_providers()

    if providers:
        while True:
            console.print("\n  [bold]Providers TTS disponibles:[/bold]")
            for i, p in enumerate(providers, 1):
                voices_info = f"{len(p.voices)} voces" if p.voices else "voces auto"
                console.print(f"    {i}. [bold]{p.name}[/bold] ({p.type}) — {voices_info}")
            console.print(f"    {len(providers) + 1}. Añadir nuevo provider ahora")
            console.print(f"    0. Omitir síntesis TTS")

            raw = ask("Selecciona provider(s) para este proyecto (números separados por coma)")
            if raw.strip() == "0":
                return

            try:
                indices = [int(x.strip()) for x in raw.split(",") if x.strip()]
            except ValueError:
                console.print("  [red]Entrada inválida.[/red]")
                continue

            for idx in indices:
                if idx == len(providers) + 1:
                    _wizard_add_provider()
                    providers = load_providers()
                elif 1 <= idx <= len(providers):
                    source = _provider_to_tts_source(providers[idx - 1], project)
                    if source and source.selected_voices:
                        project.synthesis.sources.append(source)
                        save_project(project)

            more = ask_choice("¿Añadir otro provider al proyecto?", ["s", "n"], default="n")
            if more != "s":
                break
    else:
        # Flujo manual de respaldo — sin providers configurados y usuario declinó
        while True:
            source_type = ask_choice(
                "Tipo de fuente TTS",
                ["piper", "openai", "ninguna"],
                default="piper",
            )
            if source_type == "ninguna":
                break
            if source_type == "piper":
                source = _configure_piper_source()
            else:
                source = _configure_openai_source()
            if source and source.selected_voices:
                project.synthesis.sources.append(source)
                save_project(project)
            more = ask_choice("¿Añadir otra fuente TTS?", ["s", "n"], default="n")
            if more != "s":
                break


def _configure_piper_source() -> TtsSource | None:
    from trainer.synthesizer import list_voices_piper
    voices_dir = ask("Directorio de voces Piper", default="piper/voices")
    available = list_voices_piper(voices_dir)
    if not available:
        console.print(f"  [red]No se encontraron modelos .onnx en {voices_dir}[/red]")
        return None

    console.print("  Voces disponibles:")
    for i, v in enumerate(available, 1):
        console.print(f"    {i}. {Path(v).stem}")

    raw = ask("Selecciona voces (números separados por coma, o 'todas')", default="todas")
    if raw.lower() == "todas":
        selected = available
    else:
        try:
            indices = [int(x.strip()) - 1 for x in raw.split(",")]
            selected = [available[i] for i in indices if 0 <= i < len(available)]
        except ValueError:
            selected = available

    return TtsSource(
        type="piper",
        voices_dir=voices_dir,
        selected_voices=selected,
    )


def _configure_openai_source() -> TtsSource | None:
    import asyncio
    from trainer.synthesizer import list_voices_openai

    url = ask("URL del endpoint (ej: https://api.elevenlabs.io/v1)")
    token_env = ask("Variable de entorno del token (ej: ELEVENLABS_API_KEY)")
    token = os.environ.get(token_env, "")
    if not token:
        token = console.input("  Token (se usará solo ahora, no se guarda): ").strip()

    console.print("  Consultando voces disponibles...")
    voices_raw = asyncio.run(list_voices_openai(url, token))

    if voices_raw:
        console.print(f"  ✓ {len(voices_raw)} voces encontradas")
        for i, v in enumerate(voices_raw[:20], 1):
            name = v.get("name") or v.get("voice_id") or str(v)
            console.print(f"    {i}. {name}")

        raw = ask("Selecciona voces (números, o 'todas')", default="todas")
        if raw.lower() == "todas":
            selected = [v.get("name") or v.get("voice_id") for v in voices_raw]
        else:
            try:
                indices = [int(x.strip()) - 1 for x in raw.split(",")]
                selected = [
                    (voices_raw[i].get("name") or voices_raw[i].get("voice_id"))
                    for i in indices if 0 <= i < len(voices_raw)
                ]
            except ValueError:
                selected = [v.get("name") or v.get("voice_id") for v in voices_raw]
    else:
        console.print("  [yellow]No se pudo obtener la lista de voces. Entrada manual.[/yellow]")
        raw = ask("Nombres de las voces (separados por coma)")
        selected = [v.strip() for v in raw.split(",") if v.strip()]

    return TtsSource(
        type="openai",
        url=url,
        token_env=token_env,
        selected_voices=selected,
    )


# ─── Grabación / importación ──────────────────────────────────────────────────

def _show_preparation_summary(project: Project):
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


def _wizard_record_or_import_voice(project: Project, voice):
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
            voice.status = "done"  # aceptamos clips parciales

    save_project(project)


# ─── Síntesis ─────────────────────────────────────────────────────────────────

def _wizard_synthesize(project: Project):
    from trainer.synthesizer import run_synthesis
    console.print("\n[bold]Generando muestras sintéticas...[/bold]")
    generated = run_synthesis(project)
    console.print(f"  [green]✓ {generated} clips generados[/green]")


# ─── Entrenamiento ────────────────────────────────────────────────────────────

def _wizard_train(project: Project):
    from trainer.trainer_core import run_training, TrainingConfig
    from rich.progress import Progress, SpinnerColumn, TextColumn

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


# ─── Evaluación ───────────────────────────────────────────────────────────────

def _wizard_evaluate(project: Project):
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


# ─── Funciones públicas para subcomandos CLI ──────────────────────────────────

def run_record_step(model_name: str, voice_name: str | None):
    project = load_project(model_name)
    if voice_name:
        voice = next((v for v in project.voices if v.name == voice_name), None)
        if not voice:
            console.print(f"[red]Voz '{voice_name}' no encontrada en el proyecto.[/red]")
            return
        _wizard_record_or_import_voice(project, voice)
    else:
        pending = [v for v in project.voices if v.status == "pending"]
        for voice in pending:
            _wizard_record_or_import_voice(project, voice)


def run_import_step(model_name: str, voice_name: str | None, directory: str | None):
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


def run_synthesize_step(model_name: str):
    project = load_project(model_name)
    if not project.synthesis.sources:
        _wizard_configure_synthesis(project)
    _wizard_synthesize(project)


def run_train_step(model_name: str):
    project = load_project(model_name)
    _wizard_train(project)


def run_evaluate_step(model_name: str):
    project = load_project(model_name)
    _wizard_evaluate(project)


def _wizard_add_provider() -> None:
    """Wizard interactivo para añadir un provider TTS global."""
    import asyncio
    from trainer.providers import ProviderConfig, add_or_update_provider

    console.print(Panel("[bold]Nuevo provider TTS[/bold]", box=box.ROUNDED))

    name = ask("Nombre del provider (ej: elevenlabs, jspeaker)")
    if not name:
        return

    type_ = ask_choice("Tipo", ["openai", "piper"], default="openai")

    if type_ == "openai":
        url = ask("URL del endpoint (ej: https://api.elevenlabs.io/v1)")
        token_env_raw = ask("Variable de entorno del token (vacío si no hay autenticación)")
        token_env = token_env_raw.strip() or None

        token = os.environ.get(token_env, "") if token_env else ""
        console.print("  Consultando voces disponibles...")
        from trainer.synthesizer import list_voices_openai
        voices_raw = asyncio.run(list_voices_openai(url, token))

        if voices_raw:
            console.print(f"  ✓ {len(voices_raw)} voces encontradas")
            for i, v in enumerate(voices_raw[:20], 1):
                vname = v.get("name") or v.get("voice_id") or str(v)
                console.print(f"    {i}. {vname}")
            raw = ask("Selecciona voces (números separados por coma, o 'todas')", default="todas")
            if raw.lower() == "todas":
                voices = [v.get("name") or v.get("voice_id") for v in voices_raw]
            else:
                try:
                    indices = [int(x.strip()) - 1 for x in raw.split(",")]
                    voices = [
                        (voices_raw[i].get("name") or voices_raw[i].get("voice_id"))
                        for i in indices if 0 <= i < len(voices_raw)
                    ]
                except ValueError:
                    voices = [v.get("name") or v.get("voice_id") for v in voices_raw]
        else:
            console.print("  [yellow]No se encontraron voces automáticamente.[/yellow]")
            raw = ask("Introduce las voces manualmente (separadas por coma, vacío si solo hay una)")
            voices = [v.strip() for v in raw.split(",") if v.strip()] if raw.strip() else []

        provider = ProviderConfig(
            name=name,
            type="openai",
            url=url,
            token_env=token_env,
            voices=voices,
        )

    else:  # piper
        from trainer.synthesizer import list_voices_piper
        voices_dir = ask("Directorio de voces Piper", default="piper/voices")
        binary = ask("Ruta al binario piper", default="piper/piper")
        available = list_voices_piper(voices_dir)
        if not available:
            console.print(f"  [red]No se encontraron modelos .onnx en {voices_dir}[/red]")
            return
        console.print("  Voces disponibles:")
        for i, v in enumerate(available, 1):
            console.print(f"    {i}. {Path(v).stem}")
        raw = ask("Selecciona voces (números separados por coma, o 'todas')", default="todas")
        if raw.lower() == "todas":
            voices = available
        else:
            try:
                indices = [int(x.strip()) - 1 for x in raw.split(",")]
                voices = [available[i] for i in indices if 0 <= i < len(available)]
            except ValueError:
                voices = available
        provider = ProviderConfig(
            name=name,
            type="piper",
            binary=binary,
            voices_dir=voices_dir,
            voices=voices,
        )

    raw_speeds = ask(
        "Velocidades por defecto (ej: 0.8,1.0,1.2)",
        default=",".join(str(s) for s in provider.speeds),
    )
    try:
        provider.speeds = [float(s.strip()) for s in raw_speeds.split(",") if s.strip()]
    except ValueError:
        pass

    add_or_update_provider(provider)
    console.print(f"  [green]✅ Provider '{name}' guardado en configs/providers.local.json[/green]")


def _provider_to_tts_source(provider: "ProviderConfig", project: Project) -> TtsSource | None:
    """Convierte un ProviderConfig global en un TtsSource para el proyecto."""
    import asyncio

    if provider.voices:
        selected = provider.voices
    elif provider.type == "openai":
        from trainer.synthesizer import list_voices_openai
        token = os.environ.get(provider.token_env, "") if provider.token_env else ""
        console.print(f"  Consultando voces de [bold]{provider.name}[/bold]...")
        voices_raw = asyncio.run(list_voices_openai(provider.url, token))
        if voices_raw:
            console.print(f"  ✓ {len(voices_raw)} voces disponibles")
            for i, v in enumerate(voices_raw[:20], 1):
                vname = v.get("name") or v.get("voice_id") or str(v)
                console.print(f"    {i}. {vname}")
            raw = ask("Selecciona voces (números, o 'todas')", default="todas")
            if raw.lower() == "todas":
                selected = [v.get("name") or v.get("voice_id") for v in voices_raw]
            else:
                try:
                    indices = [int(x.strip()) - 1 for x in raw.split(",")]
                    selected = [
                        (voices_raw[i].get("name") or voices_raw[i].get("voice_id"))
                        for i in indices if 0 <= i < len(voices_raw)
                    ]
                except ValueError:
                    selected = [v.get("name") or v.get("voice_id") for v in voices_raw]
        else:
            console.print("  [yellow]No se pudieron obtener voces automáticamente.[/yellow]")
            raw = ask("Introduce las voces manualmente (separadas por coma)")
            selected = [v.strip() for v in raw.split(",") if v.strip()]
    else:  # piper
        from trainer.synthesizer import list_voices_piper
        console.print(f"  Escaneando modelos en {provider.voices_dir}...")
        available = list_voices_piper(provider.voices_dir or "piper/voices")
        if not available:
            console.print(f"  [red]No se encontraron modelos .onnx en {provider.voices_dir}[/red]")
            selected = []
        else:
            console.print(f"  ✓ {len(available)} modelos encontrados")
            for i, v in enumerate(available, 1):
                console.print(f"    {i}. {Path(v).stem}")
            raw = ask("Selecciona voces (números, o 'todas')", default="todas")
            if raw.lower() == "todas":
                selected = available
            else:
                try:
                    indices = [int(x.strip()) - 1 for x in raw.split(",")]
                    selected = [available[i] for i in indices if 0 <= i < len(available)]
                except ValueError:
                    selected = available

    if not selected:
        return None

    raw_speeds = ask(
        f"Velocidades para {provider.name} en este proyecto",
        default=",".join(str(s) for s in provider.speeds),
    )
    try:
        speeds = [float(s.strip()) for s in raw_speeds.split(",") if s.strip()]
    except ValueError:
        speeds = provider.speeds

    return TtsSource(
        type=provider.type,
        url=provider.url,
        token_env=provider.token_env,
        binary=provider.binary,
        voices_dir=provider.voices_dir,
        selected_voices=selected,
        speeds=speeds,
    )
