from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

import typer
from rich.progress import Progress as ProgressBar
from rich.progress import TaskID

from ragglet.core.issues import Issue, ValidationError, render_issues
from ragglet.core.progress import Progress


@contextmanager
def exit_on_validation_error() -> Iterator[None]:
    try:
        yield
    except ValidationError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(2) from e


@contextmanager
def progress_bar() -> Iterator[Progress]:
    # Без фонового потока отрисовки: в процессе прогона он отнимал бы GIL у замеров.
    # Перерисовка не чаще раза в секунду; полоса появляется с первым событием.
    bar = ProgressBar(auto_refresh=False)
    tasks: dict[str, TaskID] = {}
    last_refresh = 0.0

    def report(stage: str, done: int, total: int) -> None:
        nonlocal last_refresh
        if not tasks:
            bar.start()
        if stage not in tasks:
            tasks[stage] = bar.add_task(stage, total=total)
        bar.update(tasks[stage], completed=done, total=total)
        if time.monotonic() - last_refresh >= 1.0:
            bar.refresh()
            last_refresh = time.monotonic()

    try:
        yield report
    finally:
        if tasks:
            bar.stop()


def echo_warnings(title: str, warnings: list[Issue]) -> None:
    if warnings:
        typer.echo(render_issues(title, warnings))
