from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ragglet.retrieval.merge import RetrievedItem


@dataclass(frozen=True)
class RerankedItem:
    item: RetrievedItem
    score: float


class Reranker(Protocol):
    revision: str | None  # разрешённая ревизия модели; None, если модели нет

    async def rerank(
        self,
        query: str,
        items: list[RetrievedItem],
        input_top_k: int,
        output_top_n: int,
    ) -> list[RerankedItem]:
        ...
