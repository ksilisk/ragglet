from pathlib import Path

import anyio
import typer
from rich import print as rprint

from ragglet.bench.preflight import validate_scenario
from ragglet.bench.run_scenario import run_scenario
from ragglet.config.scenario import load_scenario
from ragglet.core.issues import ValidationError, render_issues
from ragglet.datasets.sources import describe_dataset

bench_app = typer.Typer(help="Run benchmarks and evaluate scenarios.")


@bench_app.command("run")
def run(scenario: Path = typer.Argument(..., exists=True, dir_okay=False)) -> None:
    cfg = load_scenario(scenario)
    try:
        res = anyio.run(run_scenario, cfg)
    except ValidationError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(2) from e
    rprint("[bold green]Done[/bold green]")
    rprint(res["summary"])
    rprint(f"Artifacts: {res['run_dir']}")
    if res["warnings"]:
        typer.echo(render_issues("Preflight warnings", res["warnings"]))


@bench_app.command("validate")
def validate(scenario: Path = typer.Argument(..., exists=True, dir_okay=False)) -> None:
    """Проверки датасета и индекса, которые `bench run` делает перед прогоном."""
    cfg = load_scenario(scenario)
    try:
        checked = anyio.run(validate_scenario, cfg)
    except ValidationError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from e
    except NotImplementedError as e:
        rprint(f"[bold yellow]Validation skipped[/bold yellow]: {e}")
        raise typer.Exit(2) from e
    except Exception as e:
        rprint(f"[bold red]Validation failed[/bold red]: {e}")
        raise typer.Exit(2) from e

    dataset, index = checked.dataset, checked.index
    chunking = index["chunking_method"]
    if index["chunk_size"] is not None:
        chunking += f" {index['chunk_size']}/{index['chunk_overlap']}"
    rprint("Preflight: [bold green]OK[/bold green]")
    typer.echo(f"scenario: {cfg.name}")
    typer.echo(f"dataset: {describe_dataset(cfg.dataset.source, cfg.dataset.id, dataset.revision)}")
    typer.echo(f"split: {cfg.dataset.split}, queries: {len(dataset.items)}")
    typer.echo(f"corpus sha256: {dataset.corpus_sha256}")
    typer.echo(f"eval sha256: {dataset.eval_sha256}")
    typer.echo(f"index: {index['index_profile_id']}, built {index['built_at']}")
    typer.echo(f"embedding: {index['embedding_model']}@{index['embedding_revision']}")
    typer.echo(f"chunking: {chunking}")
    if checked.warnings:
        typer.echo(render_issues("Preflight warnings", checked.warnings))
