# tests/test_ui_tables.py
from __future__ import annotations
from io import StringIO
import pytest
from rich.console import Console
from trainer.providers import ProviderConfig


def _make_console():
    buf = StringIO()
    return Console(file=buf, width=120, highlight=False), buf


def test_print_providers_table_empty():
    import trainer.ui.tables as tables_mod
    con, buf = _make_console()
    tables_mod.console = con
    tables_mod.print_providers_table([])
    assert "providers add" in buf.getvalue()


def test_print_providers_table_shows_provider():
    import trainer.ui.tables as tables_mod
    con, buf = _make_console()
    tables_mod.console = con
    p = ProviderConfig(name="elevenlabs", type="openai", url="https://api.elevenlabs.io/v1")
    tables_mod.print_providers_table([p])
    assert "elevenlabs" in buf.getvalue()
