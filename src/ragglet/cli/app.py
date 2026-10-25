from __future__ import annotations

import typer

from ragglet.cli.bench import bench_app
from ragglet.cli.dataset import dataset_app
from ragglet.cli.index import index_app

app = typer.Typer(
    name="ragglet",
    help="Ragglet — RAG benchmarking playground (indexing + scenarios).",
    no_args_is_help=True,
)

app.add_typer(bench_app, name="bench")
app.add_typer(index_app, name="index")
app.add_typer(dataset_app, name="dataset")


@app.command("serve")
def serve(
    host: str | None = typer.Option(None, "--host", help="Bind address (default 127.0.0.1, or RAGGLET_HOST)"),
    port: int | None = typer.Option(None, "--port", help="Port (default 8000, or RAGGLET_PORT)"),
) -> None:
    """Start the REST API server. Other settings: RAGGLET_* variables or .env."""
    import uvicorn

    from ragglet.api.app import create_app
    from ragglet.api.settings import ServerSettings

    settings = ServerSettings(**{k: v for k, v in {"host": host, "port": port}.items() if v is not None})
    typer.echo(settings.describe())
    # один процесс: очередь задач и блокировки библиотеки живут в его памяти
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, workers=1)
