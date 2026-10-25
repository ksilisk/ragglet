from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Ranking:
    grades: list[int]  # grade каждого ранга после дедупа, 0 — нерелевантен
    ideal: list[int] | None  # grades всех релевантных из qrels по убыванию; None — неизвестно


def _hits(ranking: Ranking, k: int) -> int:
    return sum(1 for grade in ranking.grades[:k] if grade > 0)


def recall_at_k(ranking: Ranking, k: int) -> float:
    if not ranking.ideal:
        return 0.0
    return _hits(ranking, k) / len(ranking.ideal)


def hit_at_k(ranking: Ranking, k: int) -> float:
    return 1.0 if _hits(ranking, k) else 0.0


def mrr_at_k(ranking: Ranking, k: int) -> float:
    for rank, grade in enumerate(ranking.grades[:k], start=1):
        if grade > 0:
            return 1.0 / rank
    return 0.0


def precision_at_k(ranking: Ranking, k: int) -> float:
    return _hits(ranking, k) / k


def r_cap_at_k(ranking: Ranking, k: int) -> float:
    if not ranking.ideal:
        return 0.0
    return _hits(ranking, k) / min(k, len(ranking.ideal))


def _dcg(grades: list[int], k: int) -> float:
    # линейный gain = grade, как ndcg_cut в trec_eval (BEIR, MTEB)
    return sum(grade / math.log2(rank + 1) for rank, grade in enumerate(grades[:k], start=1) if grade > 0)


def ndcg_at_k(ranking: Ranking, k: int) -> float:
    ideal = _dcg(ranking.ideal or [], k)
    if not ideal:
        return 0.0
    return _dcg(ranking.grades, k) / ideal


@dataclass(frozen=True)
class MetricSpec:
    fn: Callable[[Ranking, int], float]
    # метрике нужно знать все релевантные единицы (Ranking.ideal), а это известно только на doc-level
    needs_ideal: bool = False


METRICS: dict[str, MetricSpec] = {
    "recall": MetricSpec(recall_at_k, needs_ideal=True),
    "hit": MetricSpec(hit_at_k),
    "mrr": MetricSpec(mrr_at_k),
    "precision": MetricSpec(precision_at_k),
    "r_cap": MetricSpec(r_cap_at_k, needs_ideal=True),
    "ndcg": MetricSpec(ndcg_at_k, needs_ideal=True),
}


def metrics_for_level(level: str) -> list[str]:
    """Метрики, допустимые на уровне: chunk — без тех, кому нужны все релевантные единицы."""
    if level == "chunk":
        return [name for name, spec in METRICS.items() if not spec.needs_ideal]
    return list(METRICS)
