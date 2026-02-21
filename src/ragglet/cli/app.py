from __future__ import annotations

import typer

from ragglet.cli.bench import bench_app
from ragglet.cli.index import index_app
from ragglet.cli.serve import serve_command

app = typer.Typer(
    name="ragglet",
    help="Ragglet — RAG benchmarking playground (indexing + scenarios).",
    no_args_is_help=True,
)

app.add_typer(bench_app, name="bench")
app.add_typer(index_app, name="index")
app.command("serve")(serve_command)