from __future__ import annotations

from typing import Any
import anyio

from ragglet.config.scenario import ScenarioConfig
from ragglet.retrieval.fanout_rrf import SourceSpec, RetrievedItem, rrf_merge, run_sources
from ragglet.stores.qdrant_store import QdrantStore


def _to_retrieved(candidate, source_name: str) -> RetrievedItem:
    payload = candidate.payload or {}
    ext = payload.get("external_id")
    external_id = str(ext) if ext is not None else str(candidate.id)
    return RetrievedItem(
        point_id=str(candidate.id),
        external_id=external_id,
        score_raw=float(candidate.score_raw),
        source=source_name,
        payload=candidate.payload,
    )


async def retrieve_ids(cfg: ScenarioConfig, store: QdrantStore, vec: list[float]) -> list[str]:
    """
    Возвращает список retrieved external_ids (то, по чему считаем метрики).
    """
    strategy = cfg.retrieval.strategy
    top_k = cfg.retrieval.top_k

    if strategy == "single":
        if cfg.retrieval.sources:
            s0 = cfg.retrieval.sources[0]
            src = SourceSpec(name=s0.name, kind=s0.kind, weight=s0.weight, params=s0.params)
        else:
            src = SourceSpec(name="vec", kind="vector", weight=1.0, params={})

        if src.kind != "vector":
            raise NotImplementedError("Only vector source is implemented in baseline")

        candidates = store.search(vec, top_k=top_k, params=src.params)
        retrieved_items = [_to_retrieved(c, src.name) for c in candidates]
        return [it.external_id for it in retrieved_items]

    if strategy == "fanout_merge":
        sources: list[SourceSpec] = [
            SourceSpec(name=s.name, kind=s.kind, weight=s.weight, params=s.params)
            for s in cfg.retrieval.sources
        ]

        for s in sources:
            if s.kind != "vector":
                raise NotImplementedError(f"Source kind '{s.kind}' is not implemented yet")

        async def _query_source(s: SourceSpec):
            return store.search(vec, top_k=top_k, params=s.params)

        tasks = [(s.name, (lambda s=s: _query_source(s))) for s in sources]

        with anyio.fail_after(cfg.timeouts.retrieval_ms / 1000.0):
            parallel = bool(cfg.retrieval.async_enabled)
            res_map = await run_sources(tasks, parallel=parallel)

        per_source_ranked: dict[str, list[RetrievedItem]] = {}
        weights: dict[str, float] = {}

        for s in sources:
            weights[s.name] = s.weight
            per_source_ranked[s.name] = [_to_retrieved(c, s.name) for c in res_map.get(s.name, [])]

        merged = rrf_merge(
            per_source_ranked=per_source_ranked,
            weights=weights,
            rrf_k=cfg.retrieval.merge.rrf_k,
            top_k=top_k,
            deduplicate=cfg.retrieval.merge.deduplicate,
        )
        return [it.external_id for it in merged]

    raise NotImplementedError(f"retrieval.strategy='{strategy}' is not implemented")