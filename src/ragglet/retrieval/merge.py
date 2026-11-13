from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

import anyio


@dataclass(frozen=True)
class RetrievedItem:
    point_id: str
    chunk_id: str
    doc_id: str
    score_raw: float
    source: str
    payload: dict[str, Any] | None
    score_fused: float | None = None

    @property
    def ranking_score(self) -> float:
        """Interleave/single retain raw scores; a fused zero is a valid score."""
        return self.score_raw if self.score_fused is None else self.score_fused


@dataclass(frozen=True)
class SourceSpec:
    name: str
    kind: str
    weight: float
    params: dict[str, Any]


def validate_item(item: RetrievedItem) -> None:
    if not item.chunk_id.strip() or not item.doc_id.strip():
        raise ValueError("retrieval candidates must have nonempty chunk_id and doc_id")
    if not math.isfinite(item.score_raw):
        raise ValueError(f"non-finite score from source {item.source!r}")


def deduplicate_by_chunk_id(items: Iterable[RetrievedItem]) -> list[RetrievedItem]:
    seen: set[str] = set()
    out: list[RetrievedItem] = []
    for item in items:
        if item.chunk_id not in seen:
            seen.add(item.chunk_id)
            out.append(item)
    return out


def _prepare(
    ranked: dict[str, list[RetrievedItem]], weights: dict[str, float], top_k: int,
) -> tuple[dict[str, list[RetrievedItem]], dict[str, float]]:
    """Dict order is the configured source order, not completion order."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    active: dict[str, list[RetrievedItem]] = {}
    positive: dict[str, float] = {}
    docs: dict[str, str] = {}
    for source, items in ranked.items():
        weight = float(weights.get(source, 1.0))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("source weights must be finite and >= 0")
        for item in items:
            validate_item(item)
            if docs.setdefault(item.chunk_id, item.doc_id) != item.doc_id:
                raise ValueError(f"conflicting doc_id for chunk {item.chunk_id!r}")
        if weight > 0:
            active[source] = deduplicate_by_chunk_id(items)
            positive[source] = weight
    if ranked and not positive:
        raise ValueError("at least one source weight must be positive")
    return active, positive


def _fused_items(
    ranked: dict[str, list[RetrievedItem]], contributions: dict[str, list[float]], top_k: int,
) -> list[RetrievedItem]:
    representatives: dict[str, RetrievedItem] = {}
    ties: dict[str, tuple[int, int, str]] = {}
    for index, items in enumerate(ranked.values()):
        for rank, item in enumerate(items, 1):
            tie = (rank, index, item.chunk_id)
            if item.chunk_id not in ties or tie < ties[item.chunk_id]:
                ties[item.chunk_id] = tie
                representatives[item.chunk_id] = item
    try:
        scores = {cid: math.fsum(parts) for cid, parts in contributions.items()}
    except OverflowError as exc:
        raise ValueError("merge score overflow") from exc
    if any(not math.isfinite(score) for score in scores.values()):
        raise ValueError("merge score overflow")
    ordered = sorted(scores, key=lambda cid: (-scores[cid], ties[cid]))
    return [replace(representatives[cid], score_fused=scores[cid]) for cid in ordered[:top_k]]


def rrf_merge(
    per_source_ranked: dict[str, list[RetrievedItem]],
    weights: dict[str, float],
    rrf_k: int,
    top_k: int,
    deduplicate: bool,
) -> list[RetrievedItem]:
    """sum(weight / (rrf_k + rank)); fusion always aggregates by chunk_id.

    deduplicate is retained for configuration compatibility; only interleave can
    return the same chunk from different sources. Local duplicates never vote twice.
    """
    if rrf_k < 1:
        raise ValueError("rrf_k must be >= 1")
    ranked, weights = _prepare(per_source_ranked, weights, top_k)
    contributions: dict[str, list[float]] = {}
    for source, items in ranked.items():
        for rank, item in enumerate(items, 1):
            contributions.setdefault(item.chunk_id, []).append(weights[source] / (rrf_k + rank))
    return _fused_items(ranked, contributions, top_k)


async def run_sources(tasks: list[tuple[str, Any]], parallel: bool) -> dict[str, Any]:
    results: dict[str, Any] = {}
    if parallel:
        async def _run_one(name: str, fn):
            results[name] = await fn()

        async with anyio.create_task_group() as tg:
            for name, fn in tasks:
                tg.start_soon(_run_one, name, fn)
        return results
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
    """Round-robin or smooth weighted round-robin, without fixed output quotas."""
    ranked, weights = _prepare(per_source_ranked, weights, top_k)
    # Scale weights to avoid overflowing the scheduler; this does not affect ratios.
    scale = max(weights.values(), default=1.0)
    weights = {s: w / scale if weighted else 1.0 for s, w in weights.items()}
    current = {s: 0.0 for s in ranked}
    positions = {s: 0 for s in ranked}
    used: set[str] = set()
    out: list[RetrievedItem] = []
    order = {s: i for i, s in enumerate(ranked)}
    cursor = 0
    while len(out) < top_k:
        available = []
        for source, items in ranked.items():
            while positions[source] < len(items) and deduplicate and items[positions[source]].chunk_id in used:
                positions[source] += 1
            if positions[source] < len(items):
                available.append(source)
        if not available:
            break
        if weighted:
            for source in available:
                current[source] += weights[source]
            source = max(available, key=lambda s: current[s])
            current[source] -= math.fsum(weights[s] for s in available)
        else:
            source = next((s for s in available if order[s] >= cursor), available[0])
            cursor = (order[source] + 1) % len(ranked)
        item = ranked[source][positions[source]]
        positions[source] += 1
        used.add(item.chunk_id)
        out.append(replace(item, score_fused=None))
    return out


def weighted_sum_merge(
    per_source_ranked: dict[str, list[RetrievedItem]],
    weights: dict[str, float],
    top_k: int,
    deduplicate: bool = True,
    normalize: str = "minmax",
    eps: float = 1e-9,
) -> list[RetrievedItem]:
    """Sum scores after per-source normalization; always aggregate by chunk_id."""
    if normalize not in ("minmax", "none"):
        raise ValueError(f"unsupported normalization: {normalize!r}")
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and >= 0")
    ranked, weights = _prepare(per_source_ranked, weights, top_k)
    contributions: dict[str, list[float]] = {}
    for source, items in ranked.items():
        if not items:
            continue
        scores = [item.score_raw for item in items]
        if normalize == "minmax":
            low, high = min(scores), max(scores)
            span = high - low
            if not math.isfinite(span):
                raise ValueError("normalization range overflow")
            scores = [(score - low) / span for score in scores] if span > eps else [0.0] * len(items)
        for item, score in zip(items, scores, strict=True):
            contributions.setdefault(item.chunk_id, []).append(weights[source] * score)
    return _fused_items(ranked, contributions, top_k)
