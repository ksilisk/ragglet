from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

RetrievalStatus = Literal["not_run", "complete", "partial", "failed", "cache_hit"]
SourceStatus = Literal["not_started", "ok", "timeout", "unavailable", "cancelled"]


@dataclass
class RetrievalDiagnostics:
    """Состояние одного запроса; индексы источников совпадают с колонками задержек."""

    status: RetrievalStatus = "not_run"
    sources: dict[int, SourceStatus] = field(default_factory=dict)

    def columns(self, source_count: int) -> dict[str, str]:
        return {
            "retrieval_status": self.status,
            **{f"source_{i}_status": self.sources.get(i, "not_started") for i in range(source_count)},
        }
