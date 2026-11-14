from __future__ import annotations

import anyio

from ragglet.config.scenario import ScenarioConfig
from ragglet.core.errors import SourceNetworkTimeout, SourceUnavailable, StageTimeout
from ragglet.retrieval.backends import BackendRegistry
from ragglet.retrieval.diagnostics import RetrievalDiagnostics
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
    diagnostics: RetrievalDiagnostics | None = None,
) -> list[RetrievedItem]:
    """
    Возвращает retrieved items with payloads.
    """
    strategy = cfg.retrieval.strategy
    top_k = cfg.retrieval.top_k
    sources = active_sources(cfg.retrieval)
    timings = timings if timings is not None else RetrievalTimings()
    diagnostics = diagnostics if diagnostics is not None else RetrievalDiagnostics()
    diagnostics.status = "failed"
    results = {}
    failed_fast = False

    if strategy not in IMPLEMENTED_STRATEGIES:
        raise NotImplementedError(f"retrieval.strategy='{strategy}' is not implemented")
    if strategy == "single" and sources[0].kind != "vector":
        raise NotImplementedError("Only vector source is implemented in baseline")

    async def query_source(index: int, source: SourceSpec):
        nonlocal failed_fast
        if wait_scope.cancel_called:
            return []  # последовательный fanout не запускает следующие источники после отказа
        diagnostics.sources[index] = "cancelled"  # при отмене await успешного ответа нет
        try:
            with timings.source(index):
                if source.kind == "vector":
                    store = backends.vector_for(_backend_ref(source.params))
                    found = await store.search(vec, top_k=top_k, params=_search_params(source.params))
                elif source.kind == "keyword":
                    found = await backends.keyword.search(query_text, top_k=top_k)
                else:
                    raise NotImplementedError("Only 'vector' or 'keyword' sources are implemented yet")
        except SourceNetworkTimeout:
            diagnostics.sources[index] = "timeout"
        except SourceUnavailable:
            diagnostics.sources[index] = "unavailable"
        else:
            diagnostics.sources[index] = "ok"
            results[source.name] = found
            return found
        if cfg.retrieval.failure_policy == "strict":
            failed_fast = True
            wait_scope.cancel()
        return []

    with timings.measure("sources"):
        with anyio.move_on_after(cfg.timeouts.retrieval_ms / 1000.0) as wait_scope:
            if strategy == "single":
                await query_source(0, sources[0])
            else:
                tasks = [(s.name, (lambda i=i, s=s: query_source(i, s))) for i, s in enumerate(sources)]
                await run_sources(tasks, parallel=cfg.retrieval.async_enabled)
        # Внешняя отмена/total_ms имеют приоритет над возвратом частичной выдачи.
        await anyio.lowlevel.checkpoint()
        if wait_scope.cancel_called and not failed_fast:
            for index, status in diagnostics.sources.items():
                if status == "cancelled":
                    diagnostics.sources[index] = "timeout"
        complete = len(results) == len(sources)
        usable = any(s.weight > 0 and s.name in results for s in sources)
        if not complete and (cfg.retrieval.failure_policy == "strict" or not usable):
            if "unavailable" in diagnostics.sources.values():
                raise SourceUnavailable("Retrieval sources unavailable")
            raise StageTimeout("retrieval")

    if strategy == "single":
        merged = [_to_retrieved(c, sources[0].name) for c in results[sources[0].name]]
    else:
        with timings.measure("merge"):
            per_source_ranked = {
                s.name: [_to_retrieved(c, s.name) for c in results[s.name]]
                for s in sources if s.name in results
            }
            weights = {s.name: s.weight for s in sources}
            if cfg.retrieval.merge.method == "rrf":
                merged = rrf_merge(per_source_ranked, weights, cfg.retrieval.merge.rrf_k,
                                   top_k, cfg.retrieval.merge.deduplicate)
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
    await anyio.lowlevel.checkpoint()  # доставляем внешнюю отмену до возврата результата
    diagnostics.status = "complete" if complete else "partial"
    return merged


async def retrieve_ids(cfg: ScenarioConfig, backends: BackendRegistry, query_text: str, vec: list[float]) -> list[str]:
    """
    Возвращает список retrieved chunk_id (внутренние id чанков; метрики считаются по doc_id).
    """
    retrieved_items = await retrieve_items(cfg, backends, query_text, vec)
    return [it.chunk_id for it in retrieved_items]
