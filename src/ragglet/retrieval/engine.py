from __future__ import annotations

import anyio

from ragglet.config.scenario import ScenarioConfig
from ragglet.core.errors import StageTimeout
from ragglet.retrieval.backends import BackendRegistry
from ragglet.retrieval.merge import SourceSpec, RetrievedItem, rrf_merge, run_sources, interleave_merge, \
    weighted_sum_merge


def _search_params(params: dict) -> dict:
    return {k: v for k, v in (params or {}).items() if k != "backend_ref"}


def _backend_ref(params: dict) -> str | None:
    value = (params or {}).get("backend_ref")
    return str(value) if value is not None else None


def _to_retrieved(candidate, source_name: str) -> RetrievedItem:
    payload = candidate.payload or {}
    chunk_id = payload.get("chunk_id")
    doc_id = payload.get("doc_id")
    if chunk_id is None or doc_id is None:
        raise ValueError(
            f"Point {candidate.id} from source '{source_name}' has no chunk_id/doc_id in payload: "
            "the index was built with an outdated payload contract, rebuild it with "
            "`ragglet index build`."
        )
    return RetrievedItem(
        point_id=str(candidate.id),
        chunk_id=str(chunk_id),
        doc_id=str(doc_id),
        score_raw=float(candidate.score_raw),
        source=source_name,
        payload=candidate.payload,
    )


async def retrieve_items(
    cfg: ScenarioConfig,
    backends: BackendRegistry,
    query_text: str,
    vec: list[float],
) -> list[RetrievedItem]:
    """
    Возвращает retrieved items with payloads.
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
                    store = backends.vector_for(_backend_ref(src.params))
                    candidates = await store.search(vec, top_k=top_k, params=_search_params(src.params))
                elif src.kind == "keyword":
                    candidates = await backends.keyword.search(query_text, top_k=top_k)
                else:
                    raise NotImplementedError("Only 'vector' or 'keyword' source is implemented yet")

                return [_to_retrieved(c, src.name) for c in candidates]

            if strategy == "fanout_merge":
                sources: list[SourceSpec] = [
                    SourceSpec(name=s.name, kind=s.kind, weight=s.weight, params=s.params)
                    for s in cfg.retrieval.sources
                ]

                async def _query_source(s: SourceSpec):
                    if s.kind == "vector":
                        store = backends.vector_for(_backend_ref(s.params))
                        return await store.search(vec, top_k=top_k, params=_search_params(s.params))
                    elif s.kind == "keyword":
                        return await backends.keyword.search(query_text, top_k=top_k)
                    else:
                        raise NotImplementedError("Only 'vector' or 'keyword' sources are implemented yet")

                tasks = [(s.name, (lambda s=s: _query_source(s))) for s in sources]

                parallel = bool(cfg.retrieval.async_enabled)
                res_map = await run_sources(tasks, parallel=parallel)

                per_source_ranked: dict[str, list[RetrievedItem]] = {}
                weights: dict[str, float] = {}

                for s in sources:
                    weights[s.name] = s.weight
                    per_source_ranked[s.name] = [_to_retrieved(c, s.name) for c in res_map.get(s.name, [])]
                if cfg.retrieval.merge.method == "rrf":
                    merged = rrf_merge(
                        per_source_ranked=per_source_ranked,
                        weights=weights,
                        rrf_k=cfg.retrieval.merge.rrf_k,
                        top_k=top_k,
                        deduplicate=cfg.retrieval.merge.deduplicate,
                    )
                elif cfg.retrieval.merge.method == "interleave":
                    merged = interleave_merge(per_source_ranked, weights, top_k,
                                              deduplicate=cfg.retrieval.merge.deduplicate)
                elif cfg.retrieval.merge.method == "weighted_sum":
                    merged = weighted_sum_merge(per_source_ranked, weights, top_k,
                                                deduplicate=cfg.retrieval.merge.deduplicate)
                else:
                    raise NotImplementedError("Only 'rrf' or 'interleave' merge methods are implemented yet")
                return merged

            raise NotImplementedError(f"retrieval.strategy='{strategy}' is not implemented")
    except TimeoutError:
        raise StageTimeout("retrieval")


async def retrieve_ids(cfg: ScenarioConfig, backends: BackendRegistry, query_text: str, vec: list[float]) -> list[str]:
    """
    Возвращает список retrieved chunk_id (внутренние id чанков; метрики считаются по doc_id).
    """
    retrieved_items = await retrieve_items(cfg, backends, query_text, vec)
    return [it.chunk_id for it in retrieved_items]
