# trainer/ui/prompts.py
from rich.console import Console
from rich.panel import Panel
from rich import box

console = Console()


def explain(text: str):
    """Imprime un panel de contexto antes de una pregunta."""
    console.print(Panel(text, box=box.SIMPLE, style="dim"))


def ask(prompt: str, default: str | None = None) -> str:
    """Pregunta con prompt enriquecido. Devuelve la respuesta."""
    if default:
        display = f"[bold cyan]{prompt}[/bold cyan] [[dim]{default}[/dim]]: "
    else:
        display = f"[bold cyan]{prompt}[/bold cyan]: "
    value = console.input(display).strip()
    return value or (default or "")


def ask_choice(prompt: str, options: list[str], default: str | None = None) -> str:
    """Pregunta con opciones explícitas. Devuelve la clave elegida."""
    opts_display = "  ".join(
        f"[[bold]{o}[/bold]]" if o == default else f"[{o}]"
        for o in options
    )
    value = console.input(f"[bold cyan]{prompt}[/bold cyan]  {opts_display}: ").strip().lower()
    if not value and default:
        return default
    while value not in options:
        value = console.input(f"  Opción inválida. Elige {'/'.join(options)}: ").strip().lower()
    return value


def ask_int(prompt: str, default: int | None = None, minimum: int = 1) -> int:
    """Pregunta un número entero con validación."""
    while True:
        raw = ask(prompt, str(default) if default is not None else None)
        try:
            value = int(raw)
            if value >= minimum:
                return value
            console.print(f"  [red]Debe ser al menos {minimum}.[/red]")
        except ValueError:
            console.print("  [red]Introduce un número entero.[/red]")
