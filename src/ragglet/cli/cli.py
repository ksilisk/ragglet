from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint

app = typer.Typer(
    name="ragglet",
    help="Ragglet — RAG benchmarking playground (indexing + scenarios).",
    no_args_is_help=True,
)

bench_app = typer.Typer(help="Run benchmarks and evaluate scenarios.")
index_app = typer.Typer(help="Build and manage indexes (offline).")
app.add_typer(bench_app, name="bench")
app.add_typer(index_app, name="index")


@bench_app.command("run")
def bench_run(
        scenario: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to scenario.yaml"),
        out_dir: Path = typer.Option("results/runs", "--out-dir", help="Directory for results artifacts"),
) -> None:
    """
    Run a scenario: load scenario.yaml, execute retrieval pipeline, compute metrics, save results.
    (Implementation will be added next.)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rprint("[bold green]OK[/bold green] bench run")
    rprint(f"scenario = {scenario}")
    rprint(f"out_dir  = {out_dir}")
    # TODO: call ragglet.bench.run_scenario.run(scenario, out_dir)


@index_app.command("build")
def index_build(
        config: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to index_profile.yaml"),
) -> None:
    """
    Build / rebuild an index from index_profile.yaml (parse → chunk → embed → upsert).
    (Implementation will be added later.)
    """
    rprint("[bold green]OK[/bold green] index build")
    rprint(f"config = {config}")
    # TODO: call ragglet.indexing.build_index.build(config)


@app.command("serve")
def serve(
        host: str = typer.Option("0.0.0.0", "--host", help="Bind host"),
        port: int = typer.Option(8000, "--port", help="Bind port"),
) -> None:
    """
    Serve API (optional). Stub for now.
    """
    rprint("[bold yellow]TODO[/bold yellow] serve")
    rprint(f"host={host} port={port}")
    # TODO: uvicorn ragglet.service.app:app --host host --port port


def main() -> None:
    app()


if __name__ == "__main__":
    main()
