from __future__ import annotations

from dataclasses import dataclass

from ragglet.retrieval.merge import RetrievedItem


@dataclass(frozen=True)
class ContextChunk:
    item: RetrievedItem
    text: str
    token_count: int
    rank: int
    score: float


@dataclass(frozen=True)
class ContextResult:
    chunks: list[ContextChunk]
    text: str
    token_count: int

    @property
    def ids(self) -> list[str]:
        return [chunk.item.external_id for chunk in self.chunks]
