from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter

from ragglet.core.errors import SourceNetworkTimeout


@dataclass
class RetrievalTimings:
    """Замеры одного запроса. Завершённые ветки остаются доступными после отмены fanout."""

    completed: dict[str, float] = field(default_factory=dict)
    network_timeouts: dict[int, float] = field(default_factory=dict)

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        start = perf_counter()
        yield
        self.completed[stage] = (perf_counter() - start) * 1000.0

    @contextmanager
    def source(self, index: int) -> Iterator[None]:
        start = perf_counter()
        try:
            yield
        except SourceNetworkTimeout:
            self.network_timeouts[index] = (perf_counter() - start) * 1000.0
            raise
        else:
            self.completed[f"source_{index}"] = (perf_counter() - start) * 1000.0

    def columns(self, source_count: int) -> dict[str, float]:
        stages = ["sources", "merge", *(f"source_{i}" for i in range(source_count))]
        return {
            **{f"lat_{stage}_ms": self.completed.get(stage, math.nan) for stage in stages},
            **{f"lat_source_{i}_network_timeout_ms": self.network_timeouts.get(i, math.nan)
               for i in range(source_count)},
        }
