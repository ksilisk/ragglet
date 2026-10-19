from __future__ import annotations

import anyio

from ragglet.modules.hub import cached_model_revision
from ragglet.rerank.base import RerankedItem
from ragglet.retrieval.merge import RetrievedItem


class CrossEncoderReranker:
    def __init__(self, model_name: str, revision: str | None = None):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, revision=revision)
        # CrossEncoder, в отличие от SentenceTransformer, не дописывает организацию к короткому имени
        self.revision = cached_model_revision(model_name, revision)

    async def rerank(
        self,
        query: str,
        items: list[RetrievedItem],
        input_top_k: int,
        output_top_n: int,
    ) -> list[RerankedItem]:
        candidates = items[:input_top_k]
        if not candidates:
            return []

        pairs = [(query, _text_for_item(item)) for item in candidates]
        scores = await anyio.to_thread.run_sync(
            self.model.predict,
            pairs,
            abandon_on_cancel=True,
        )
        reranked = [
            RerankedItem(item=item, score=float(score))
            for item, score in zip(candidates, scores, strict=True)
        ]
        reranked.sort(key=lambda result: result.score, reverse=True)
        return reranked[:output_top_n]


def _text_for_item(item: RetrievedItem) -> str:
    payload = item.payload or {}
    text = payload.get("text", "")
    return str(text) if text is not None else ""
