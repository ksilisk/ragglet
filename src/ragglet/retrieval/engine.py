from __future__ import annotations

import anyio

from ragglet.config.scenario import ScenarioConfig
from ragglet.core.errors import SourceNetworkTimeout, StageTimeout
from ragglet.retrieval.backends import BackendRegistry
from ragglet.retrieval.merge import (
    RetrievedItem,
    SourceSpec,
    interleave_merge,
    rrf_merge,
    run_sources,
    validate_item,
    weighted_sum_merge,
)
from ragglet.retrieval.sources import active_sources
from ragglet.retrieval.timing import RetrievalTimings

# стратегии, которые умеет retrieve_items; схема знает и cascade
IMPLEMENTED_STRATEGIES = ("single", "fanout_merge")


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
    item = RetrievedItem(
        point_id=str(candidate.id),
        chunk_id=str(chunk_id),
        doc_id=str(doc_id),
        score_raw=float(candidate.score_raw),
        source=source_name,
        payload=candidate.payload,
    )
    validate_item(item)
    return item


async def retrieve_items(
    cfg: ScenarioConfig,
    backends: BackendRegistry,
    query_text: str,
    vec: list[float],
    *,
    timings: RetrievalTimings | None = None,
) -> list[RetrievedItem]:
    """
    Возвращает retrieved items with payloads.
    """
    strategy = cfg.retrieval.strategy
    top_k = cfg.retrieval.top_k
    sources = active_sources(cfg.retrieval)
    timings = timings if timings is not None else RetrievalTimings()

    async def query_source(index: int, source: SourceSpec):
        if source.kind == "vector":
            store = backends.vector_for(_backend_ref(source.params))
            params = _search_params(source.params)
            with timings.source(index):
                return await store.search(vec, top_k=top_k, params=params)
        if source.kind == "keyword":
            with timings.source(index):
                return await backends.keyword.search(query_text, top_k=top_k)
        raise NotImplementedError("Only 'vector' or 'keyword' sources are implemented yet")

    try:
        with anyio.fail_after(cfg.timeouts.retrieval_ms / 1000.0):
            if strategy == "single":
                src = sources[0]
                if src.kind != "vector":
                    raise NotImplementedError("Only vector source is implemented in baseline")
                with timings.measure("sources"):
                    candidates = await query_source(0, src)
                return [_to_retrieved(c, src.name) for c in candidates]

            if strategy == "fanout_merge":
                tasks = [(s.name, (lambda i=i, s=s: query_source(i, s))) for i, s in enumerate(sources)]
                parallel = bool(cfg.retrieval.async_enabled)
                with timings.measure("sources"):
                    res_map = await run_sources(tasks, parallel=parallel)

                with timings.measure("merge"):
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
                                                  deduplicate=cfg.retrieval.merge.deduplicate,
                                                  weighted=cfg.retrieval.merge.interleave_weighted)
                    elif cfg.retrieval.merge.method == "weighted_sum":
                        merged = weighted_sum_merge(per_source_ranked, weights, top_k,
                                                    deduplicate=cfg.retrieval.merge.deduplicate,
                                                    normalize=cfg.retrieval.merge.normalize)
                    else:
                        raise NotImplementedError("Only 'rrf', 'interleave' or 'weighted_sum' merge methods are implemented")
                return merged

            raise NotImplementedError(f"retrieval.strategy='{strategy}' is not implemented")
    except (TimeoutError, SourceNetworkTimeout) as e:
        raise StageTimeout("retrieval") from e
    except ExceptionGroup as e:
        # TaskGroup отменил соседей. Только чистая группа сетевых таймаутов — исход
        # запроса; смешанная группа остаётся ошибкой стенда и останавливает прогон.
        _, other = e.split(SourceNetworkTimeout)
        if other is None:
            raise StageTimeout("retrieval") from e
        raise


async def retrieve_ids(cfg: ScenarioConfig, backends: BackendRegistry, query_text: str, vec: list[float]) -> list[str]:
    """
    Возвращает список retrieved chunk_id (внутренние id чанков; метрики считаются по doc_id).
    """
    retrieved_items = await retrieve_items(cfg, backends, query_text, vec)
    return [it.chunk_id for it in retrieved_items]
