from pathlib import Path

import typer
from rich import print as rprint
import anyio

from ragglet.config.scenario import load_scenario
from ragglet.bench.run_scenario import run_scenario
from ragglet.bench.validate_dataset import validate_scenario_dataset
from ragglet.indexing.metadata import IndexMetadataError

bench_app = typer.Typer(help="Run benchmarks and evaluate scenarios.")


@bench_app.command("run")
def run(scenario: Path = typer.Argument(..., exists=True, dir_okay=False)) -> None:
    cfg = load_scenario(scenario)
    try:
        res = anyio.run(run_scenario, cfg)
    except IndexMetadataError as e:
        rprint(f"[bold red]Index metadata check failed[/bold red]: {e}")
        raise typer.Exit(2) from e
    rprint("[bold green]Done[/bold green]")
    rprint(res["summary"])
    rprint(f"Artifacts: {res['run_dir']}")


@bench_app.command("validate")
def validate(
    scenario: Path = typer.Argument(..., exists=True, dir_okay=False),
    max_missing: int = typer.Option(20, "--max-missing", min=1, help="Maximum missing ids to print."),
) -> None:
    cfg = load_scenario(scenario)
    try:
        report = anyio.run(validate_scenario_dataset, cfg)
    except NotImplementedError as e:
        rprint(f"[bold yellow]Validation skipped[/bold yellow]: {e}")
        raise typer.Exit(2) from e
    except Exception as e:
        rprint(f"[bold red]Validation failed[/bold red]: {e}")
        raise typer.Exit(2) from e

    status = "[bold green]OK[/bold green]" if report.ok else "[bold red]FAILED[/bold red]"
    rprint(f"Dataset validation: {status}")
    rprint(f"scenario: {report.scenario}")
    rprint(f"dataset: {report.dataset_id}")
    rprint(f"backend: {report.backend_type}")
    rprint(f"storage mode: {report.storage_mode}")
    rprint(f"collection: {report.collection}")
    rprint(f"backend counts: {report.backend_counts}")
    rprint(f"queries checked: {report.queries_checked}")
    rprint(f"truth doc ids checked: {report.truth_ids_checked}")
    rprint(f"indexed doc ids: {report.indexed_doc_ids}")
    rprint(f"duplicate doc ids: {report.duplicate_doc_ids}")
    rprint(f"missing truth doc ids: {report.missing_count}")

    if report.missing:
        rprint("\n[bold]Missing:[/bold]")
        for item in report.missing[:max_missing]:
            rprint(f"- {item.qid}: {item.truth_id}")
        remaining = report.missing_count - max_missing
        if remaining > 0:
            rprint(f"... and {remaining} more")
        raise typer.Exit(1)
