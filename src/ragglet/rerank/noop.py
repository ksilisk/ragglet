from __future__ import annotations

from ragglet.rerank.base import RerankedItem
from ragglet.retrieval.merge import RetrievedItem


class NoopReranker:
    async def rerank(
        self,
        query: str,
        items: list[RetrievedItem],
        input_top_k: int,
        output_top_n: int,
    ) -> list[RerankedItem]:
        del query
        limited = items[:input_top_k]
        return [RerankedItem(item=item, score=float(item.score_raw)) for item in limited[:output_top_n]]
