from __future__ import annotations

import anyio

from ragglet.config.scenario import ScenarioConfig
from ragglet.core.errors import StageTimeout
from ragglet.rerank.base import RerankedItem, Reranker
from ragglet.rerank.cross_encoder import CrossEncoderReranker
from ragglet.rerank.noop import NoopReranker
from ragglet.retrieval.merge import RetrievedItem


def build_reranker(cfg: ScenarioConfig) -> Reranker:
    if not cfg.rerank.enabled or cfg.rerank.provider == "noop":
        return NoopReranker()
    if cfg.rerank.provider == "cross_encoder":
        if not cfg.rerank.model:
            raise ValueError("rerank.model is required for provider='cross_encoder'")
        return CrossEncoderReranker(cfg.rerank.model)
    raise NotImplementedError(f"rerank.provider='{cfg.rerank.provider}' is not implemented")


async def rerank_items(
    cfg: ScenarioConfig,
    reranker: Reranker,
    query: str,
    items: list[RetrievedItem],
) -> list[RerankedItem]:
    if not cfg.rerank.enabled:
        return [RerankedItem(item=item, score=float(item.score_raw)) for item in items]

    try:
        with anyio.fail_after(cfg.timeouts.rerank_ms / 1000.0):
            return await reranker.rerank(
                query=query,
                items=items,
                input_top_k=cfg.rerank.input_top_k,
                output_top_n=cfg.rerank.output_top_n,
            )
    except TimeoutError:
        raise StageTimeout("rerank")
