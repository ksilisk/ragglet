from __future__ import annotations

from typing import Protocol


class Progress(Protocol):
    """Ход операции ядра: `done` из `total` на этапе `stage`."""

    def __call__(self, stage: str, done: int, total: int) -> None: ...


def no_progress(stage: str, done: int, total: int) -> None:
    pass
