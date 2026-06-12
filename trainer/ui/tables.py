# trainer/ui/tables.py
from __future__ import annotations
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


def print_providers_table(providers: list) -> None:
    if not providers:
        console.print(
            "[dim]No hay providers configurados. "
            "Usa [bold]wake-trainer providers add[/bold] para añadir uno.[/dim]"
        )
        return
    table = Table(box=box.SIMPLE_HEAD)
    table.add_column("Nombre", style="bold")
    table.add_column("Tipo")
    table.add_column("URL / Directorio")
    table.add_column("Token", justify="center")
    table.add_column("Voces", justify="right")
    table.add_column("Velocidades")
    for p in providers:
        location = p.url or (str(p.voices_dir) if p.voices_dir else "—")
        token_str = "✅" if p.token_env else "—"
        voices_str = str(len(p.voices)) if p.voices else "auto"
        speeds_str = ", ".join(str(s) for s in p.speeds)
        table.add_row(p.name, p.type, location, token_str, voices_str, speeds_str)
    console.print(table)
