from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint

from ragglet.cli.render import echo_warnings, exit_on_validation_error, progress_bar
from ragglet.config.index_profile import load_index_profile
from ragglet.indexing.build_index import build_index

index_app = typer.Typer(help="Build and manage indexes (offline).")


@index_app.command("build")
def build(
    config: Path = typer.Argument(..., exists=True, dir_okay=False),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Only check the dataset and the storage, recreate no collections"
    ),
) -> None:
    cfg = load_index_profile(config)
    rprint("[bold]Checking index profile[/bold]" if dry_run else "[bold green]Building index[/bold green]")
    rprint(cfg.model_dump())
    with exit_on_validation_error(), progress_bar() as progress:
        res = build_index(cfg, dry_run=dry_run, progress=progress)
    echo_warnings("Dataset warnings", res.warnings)
    if dry_run:
        typer.echo(f"dataset: {res.dataset}, documents: {res.documents}")
        typer.echo(f"corpus sha256: {res.corpus_sha256}")
        for target in res.targets:
            typer.echo(f"will recreate: {target.collection} at {target.endpoint} (target {target.id})")
    else:
        rprint("[bold green]Done[/bold green]")
