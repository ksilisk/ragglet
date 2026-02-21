from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint

from ragglet.config.index_profile import load_index_profile
from ragglet.indexing.build_index import build_index

index_app = typer.Typer(help="Build and manage indexes (offline).")


@index_app.command("build")
def build(config: Path = typer.Argument(..., exists=True, dir_okay=False)) -> None:
    cfg = load_index_profile(config)
    rprint("[bold green]Building index[/bold green]")
    rprint(cfg.model_dump())
    build_index(cfg)
    rprint("[bold green]Done[/bold green]")
