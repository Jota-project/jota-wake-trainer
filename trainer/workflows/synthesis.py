# trainer/workflows/synthesis.py
from __future__ import annotations
import os
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich import box

from trainer.state import Project, TtsSource, save_project, load_project
from trainer.ui.prompts import ask, ask_choice
from trainer.ui.voice_selection import select_openai_voices, select_piper_voices

console = Console()


def configure_piper_source() -> TtsSource | None:
    from trainer.synthesizer import list_voices_piper
    voices_dir = ask("Directorio de voces Piper", default="piper/voices")
    available = list_voices_piper(voices_dir)
    if not available:
        console.print(f"  [red]No se encontraron modelos .onnx en {voices_dir}[/red]")
        return None
    selected = select_piper_voices(available)
    return TtsSource(type="piper", voices_dir=voices_dir, selected_voices=selected)


def configure_openai_source() -> TtsSource | None:
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
        selected = select_openai_voices(voices_raw)
    else:
        console.print("  [yellow]No se pudo obtener la lista de voces. Entrada manual.[/yellow]")
        raw = ask("Nombres de las voces (separados por coma)")
        selected = [v.strip() for v in raw.split(",") if v.strip()]

    return TtsSource(type="openai", url=url, token_env=token_env, selected_voices=selected)


def add_provider_interactive() -> None:
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
            voices = select_openai_voices(voices_raw)
        else:
            console.print("  [yellow]No se encontraron voces automáticamente.[/yellow]")
            raw = ask("Introduce las voces manualmente (separadas por coma, vacío si solo hay una)")
            voices = [v.strip() for v in raw.split(",") if v.strip()] if raw.strip() else []

        provider = ProviderConfig(name=name, type="openai", url=url, token_env=token_env, voices=voices)

    else:  # piper
        from trainer.synthesizer import list_voices_piper
        voices_dir = ask("Directorio de voces Piper", default="piper/voices")
        binary = ask("Ruta al binario piper", default="piper/piper")
        available = list_voices_piper(voices_dir)
        if not available:
            console.print(f"  [red]No se encontraron modelos .onnx en {voices_dir}[/red]")
            return
        voices = select_piper_voices(available)
        provider = ProviderConfig(
            name=name, type="piper", binary=binary, voices_dir=voices_dir, voices=voices
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


def _provider_to_tts_source(provider) -> TtsSource | None:
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
            selected = select_openai_voices(voices_raw)
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
            selected = select_piper_voices(available)

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


def configure_synthesis(project: Project) -> None:
    from trainer.providers import load_providers

    providers = load_providers()

    if not providers:
        add_now = ask_choice(
            "No hay providers TTS configurados. ¿Configurar uno ahora?",
            ["s", "n"],
            default="s",
        )
        if add_now == "s":
            add_provider_interactive()
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
                    add_provider_interactive()
                    providers = load_providers()
                elif 1 <= idx <= len(providers):
                    source = _provider_to_tts_source(providers[idx - 1])
                    if source and source.selected_voices:
                        project.synthesis.sources.append(source)
                        save_project(project)

            more = ask_choice("¿Añadir otro provider al proyecto?", ["s", "n"], default="n")
            if more != "s":
                break
    else:
        while True:
            source_type = ask_choice(
                "Tipo de fuente TTS",
                ["piper", "openai", "ninguna"],
                default="piper",
            )
            if source_type == "ninguna":
                break
            if source_type == "piper":
                source = configure_piper_source()
            else:
                source = configure_openai_source()
            if source and source.selected_voices:
                project.synthesis.sources.append(source)
                save_project(project)
            more = ask_choice("¿Añadir otra fuente TTS?", ["s", "n"], default="n")
            if more != "s":
                break


def synthesize_project(project: Project) -> None:
    from trainer.synthesizer import run_synthesis
    console.print("\n[bold]Generando muestras sintéticas...[/bold]")
    generated = run_synthesis(project)
    console.print(f"  [green]✓ {generated} clips generados[/green]")


def run_synthesize_step(model_name: str) -> None:
    project = load_project(model_name)
    if not project.synthesis.sources:
        configure_synthesis(project)
    synthesize_project(project)
