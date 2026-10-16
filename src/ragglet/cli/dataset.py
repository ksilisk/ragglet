from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint

from ragglet.datasets.import_dir import import_directory

dataset_app = typer.Typer(help="Dataset utilities (import external corpora to the BEIR format).")


@dataset_app.command("import")
def import_cmd(
    corpus_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Directory of .md/.txt files"),
    out_dir: Path = typer.Argument(..., help="Output BEIR dataset directory"),
    queries: Path = typer.Option(None, "--queries", exists=True, dir_okay=False, help="Legacy queries manifest (jsonl)"),
    split: str = typer.Option("test", "--split", help="qrels split name"),
) -> None:
    """
    Конвертирует директорию с текстовым корпусом (+ опциональный legacy-манифест запросов)
    в BEIR-датасет (corpus.jsonl / queries.jsonl / qrels/<split>.tsv).
    """
    counts = import_directory(corpus_dir, out_dir, queries, split)
    rprint("[bold green]Imported to BEIR[/bold green]")
    rprint(counts)
    rprint(f"out: {out_dir}")
