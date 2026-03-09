from __future__ import annotations

import anyio

from ragglet.config.scenario import ScenarioConfig
from ragglet.core.errors import StageTimeout
from ragglet.retrieval.backends import BackendRegistry
from ragglet.retrieval.fanout_rrf import SourceSpec, RetrievedItem, rrf_merge, run_sources


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


async def retrieve_ids(cfg: ScenarioConfig, backends: BackendRegistry, query_text: str, vec: list[float]) -> list[str]:
    """
    Возвращает список retrieved external_ids (то, по чему считаем метрики).
    """
    strategy = cfg.retrieval.strategy
    top_k = cfg.retrieval.top_k
    try:
        with anyio.fail_after(cfg.timeouts.retrieval_ms / 1000.0):
            if strategy == "single":
                if cfg.retrieval.sources:
                    s0 = cfg.retrieval.sources[0]
                    src = SourceSpec(name=s0.name, kind=s0.kind, weight=s0.weight, params=s0.params)
                else:
                    src = SourceSpec(name="vec", kind="vector", weight=1.0, params={})

                if src.kind != "vector":
                    raise NotImplementedError("Only vector source is implemented in baseline")

                if src.kind == "vector":
                    candidates = await backends.vector.search(vec, top_k=top_k, params=src.params)
                elif src.kind == "keyword":
                    candidates = backends.keyword.search(query_text, top_k=top_k)
                else:
                    raise NotImplementedError("Only 'vector' or 'keyword' source is implemented yet")

                retrieved_items = [_to_retrieved(c, src.name) for c in candidates]
                return [it.external_id for it in retrieved_items]

            if strategy == "fanout_merge":
                sources: list[SourceSpec] = [
                    SourceSpec(name=s.name, kind=s.kind, weight=s.weight, params=s.params)
                    for s in cfg.retrieval.sources
                ]

                async def _query_source(s: SourceSpec):
                    if s.kind == "vector":
                        return await backends.vector.search(vec, top_k=top_k, params=s.params)
                    elif s.kind == "keyword":
                        return backends.keyword.search(query_text, top_k=top_k)
                    else:
                        raise NotImplementedError("Only 'vector' or 'keyword' source is implemented yet")

                tasks = [(s.name, (lambda s=s: _query_source(s))) for s in sources]

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
    except TimeoutError:
        raise StageTimeout("retrieval")
