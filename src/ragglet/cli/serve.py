from __future__ import annotations
import typer
from rich import print as rprint

def serve_command(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
) -> None:
    rprint("[yellow]TODO[/yellow] serve")
    rprint(f"host={host} port={port}")