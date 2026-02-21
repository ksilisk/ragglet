from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import anyio


@dataclass(frozen=True)
class RetrievedItem:
    # внутренний id (UUID point id в Qdrant и т.п.)
    point_id: str
    # то, по чему мы считаем метрики и делаем dedup (external_id)
    external_id: str
    score_raw: float
    source: str
    payload: dict[str, Any] | None


@dataclass(frozen=True)
class SourceSpec:
    name: str
    kind: str           # "vector" | "keyword" | "hybrid"
    weight: float
    params: dict[str, Any]


def deduplicate_by_external_id(items: Iterable[RetrievedItem]) -> list[RetrievedItem]:
    seen: set[str] = set()
    out: list[RetrievedItem] = []
    for it in items:
        if it.external_id in seen:
            continue
        seen.add(it.external_id)
        out.append(it)
    return out


def rrf_merge(
    per_source_ranked: dict[str, list[RetrievedItem]],
    weights: dict[str, float],
    rrf_k: int,
    top_k: int,
    deduplicate: bool,
) -> list[RetrievedItem]:
    """
    Reciprocal Rank Fusion:
      score(doc) = sum_s w_s * 1/(rrf_k + rank_s(doc))
    где rank начинается с 1.
    """
    fused_score: dict[str, float] = {}
    best_item: dict[str, RetrievedItem] = {}

    for source_name, items in per_source_ranked.items():
        w = float(weights.get(source_name, 1.0))
        for rank, it in enumerate(items, start=1):
            key = it.external_id
            fused_score[key] = fused_score.get(key, 0.0) + w * (1.0 / (rrf_k + rank))
            # сохраняем один representative item (payload пригодится потом)
            if key not in best_item:
                best_item[key] = it

    # сортируем по fused_score desc
    ordered = sorted(best_item.keys(), key=lambda k: fused_score.get(k, 0.0), reverse=True)
    merged = [best_item[k] for k in ordered]

    if deduplicate:
        merged = deduplicate_by_external_id(merged)

    return merged[:top_k]

async def run_sources(
    tasks: list[tuple[str, Any]],
    parallel: bool,
) -> dict[str, Any]:
    """
    tasks: список (name, awaitable_fn)
    parallel=True  -> запускаем конкурентно
    parallel=False -> выполняем по очереди
    """
    results: dict[str, Any] = {}

    if parallel:
        async def _run_one(name: str, fn):
            results[name] = await fn()

        async with anyio.create_task_group() as tg:
            for name, fn in tasks:
                tg.start_soon(_run_one, name, fn)
        return results

    # sequential
    for name, fn in tasks:
        results[name] = await fn()
    return results