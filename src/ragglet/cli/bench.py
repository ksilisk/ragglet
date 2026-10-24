from functools import partial
from pathlib import Path

import anyio
import typer
from rich import print as rprint

from ragglet.bench.preflight import Preflight
from ragglet.bench.run_scenario import run_scenario
from ragglet.cli.render import echo_warnings, exit_on_validation_error, progress_bar
from ragglet.config.scenario import ScenarioConfig, load_scenario
from ragglet.datasets.sources import describe_dataset

bench_app = typer.Typer(help="Run benchmarks and evaluate scenarios.")


@bench_app.command("run")
def run(
    scenario: Path = typer.Argument(..., exists=True, dir_okay=False),
    dry_run: bool = typer.Option(False, "--dry-run", help="Only check the dataset and the index, run no queries"),
) -> None:
    cfg = load_scenario(scenario)
    with exit_on_validation_error(), progress_bar() as progress:
        res = anyio.run(partial(run_scenario, cfg, dry_run=dry_run, progress=progress))
    if dry_run:
        _echo_preflight(cfg, res.preflight)
    else:
        rprint("[bold green]Done[/bold green]")
        rprint(res.summary)
        rprint(f"Artifacts: {res.run_dir}")
    echo_warnings("Preflight warnings", res.preflight.warnings)


def _echo_preflight(cfg: ScenarioConfig, checked: Preflight) -> None:
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
