from pathlib import Path
import typer
from rich import print as rprint
import anyio

from ragglet.config.scenario import load_scenario
from ragglet.bench.run_scenario import run_scenario

bench_app = typer.Typer(help="Run benchmarks and evaluate scenarios.")

@bench_app.command("run")
def run(scenario: Path = typer.Argument(..., exists=True, dir_okay=False)) -> None:
    cfg = load_scenario(scenario)
    res = anyio.run(run_scenario, cfg)
    rprint("[bold green]Done[/bold green]")
    rprint(res["summary"])
    rprint(f"Artifacts: {res['run_dir']}")