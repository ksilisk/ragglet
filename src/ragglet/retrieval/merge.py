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

def interleave_merge(
    per_source_ranked: dict[str, list[RetrievedItem]],
    weights: dict[str, float],
    top_k: int,
    deduplicate: bool = True,
    weighted: bool = False,
) -> list[RetrievedItem]:
    """
    weighted=False: round-robin 1 item per source
    weighted=True: "weighted round robin" — источники с большим weight чаще дают элементы
    """
    src_names = [s for s in per_source_ranked.keys()]
    if not src_names:
        return []

    # для weighted RR сделаем простую схему: квоты пропорционально весам
    # (не идеальная, но работает). Можно заменить на smooth WRR позже.
    if weighted:
        total = sum(max(0.0, weights.get(s, 1.0)) for s in src_names) or 1.0
        quotas = {s: max(1, int(round(top_k * (weights.get(s, 1.0) / total)))) for s in src_names}
    else:
        quotas = {s: top_k for s in src_names}

    idx = {s: 0 for s in src_names}
    used: set[str] = set()
    out: list[RetrievedItem] = []

    # цикл пока можем добавлять
    progressed = True
    while len(out) < top_k and progressed:
        progressed = False
        for s in src_names:
            if len(out) >= top_k:
                break
            if quotas[s] <= 0:
                continue

            lst = per_source_ranked.get(s, [])
            while idx[s] < len(lst):
                item = lst[idx[s]]
                idx[s] += 1
                progressed = True

                key = item.external_id
                if deduplicate and key in used:
                    continue

                out.append(item)
                used.add(key)
                quotas[s] -= 1
                break

    return out

def weighted_sum_merge(
    per_source_ranked: dict[str, list[RetrievedItem]],
    weights: dict[str, float],
    top_k: int,
    deduplicate: bool = True,
    normalize: str = "minmax",  # "minmax" | "zscore" (позже) | "none"
    eps: float = 1e-9,
) -> list[RetrievedItem]:
    # накопим по external_id
    agg_score: dict[str, float] = {}
    best_item: dict[str, RetrievedItem] = {}  # чтобы payload/source/etc сохранить

    for src, items in per_source_ranked.items():
        if not items:
            continue
        w = float(weights.get(src, 1.0))

        scores = [float(it.score_raw) for it in items]
        if normalize == "minmax":
            mn, mx = min(scores), max(scores)
            denom = (mx - mn) + eps
            def norm(x: float) -> float:
                return (x - mn) / denom
        elif normalize == "none":
            def norm(x: float) -> float:
                return x
        else:
            raise NotImplementedError(f"normalize='{normalize}' not implemented")

        for it in items:
            key = it.external_id
            sc = w * norm(float(it.score_raw))
            agg_score[key] = agg_score.get(key, 0.0) + sc
            # сохраним "лучший" item как репрезентативный (например с max raw score)
            if key not in best_item or float(it.score_raw) > float(best_item[key].score_raw):
                best_item[key] = it

    merged = []
    for ext_id, sc in agg_score.items():
        it = best_item[ext_id]
        # можно положить итоговый score в score_raw (или добавить score_fused отдельным полем)
        merged.append(RetrievedItem(
            point_id=it.point_id,
            external_id=it.external_id,
            score_raw=float(sc),          # fused score
            source=it.source,             # можно поставить "fused"
            payload=it.payload,
        ))

    merged.sort(key=lambda x: x.score_raw, reverse=True)

    if deduplicate:
        # тут дубликатов уже нет (ключ ext_id), но оставим на будущее
        pass

    return merged[:top_k]