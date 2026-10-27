"""Фильтры, сортировка и колонки сводки списков прогонов и сборок: всё в памяти."""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from datetime import datetime
from typing import Any

from ragglet.registry.protocol import BuildQuery, BuildRecord, Page, RunMetrics, RunQuery, RunRecord

# колонка списка -> ключ RunMetrics.means; {k} — из RunQuery.k
SUMMARY_COLUMNS = {
    "ndcg": "doc_ndcg@{k}_mean",
    "recall": "doc_recall@{k}_mean",
    "p50": "lat_total_p50",
    "p95": "lat_total_p95",
    "qps": "throughput_qps",
    "timeout_rate": "timeout_rate",
}
RUN_SORT_KEYS = frozenset({
    "enqueued_at", "started_at", "finished_at", "duration", "scenario", "state", "dataset", "collection",
    *SUMMARY_COLUMNS,
})
BUILD_SORT_KEYS = frozenset({"enqueued_at", "started_at", "finished_at", "duration", "profile", "state"})

MEAN_SUFFIX = "_mean_over_repeats"


def run_metrics(aggregate: dict[str, Any]) -> RunMetrics:
    """Средние по повторам из summary.aggregate.json; колонки старых поколений — как есть."""
    means = {
        key.removesuffix(MEAN_SUFFIX): float(value)
        for key, value in aggregate.items()
        if key.endswith(MEAN_SUFFIX) and isinstance(value, int | float) and math.isfinite(value)
    }
    repeats = aggregate.get("repeats")
    return RunMetrics(repeats=int(repeats) if isinstance(repeats, int | float) else None, means=means)


def summary_column(record: RunRecord, column: str, k: int) -> float | None:
    if record.metrics is None:
        return None
    return record.metrics.means.get(SUMMARY_COLUMNS[column].format(k=k))


def duration_s(record: RunRecord | BuildRecord) -> float | None:
    if record.started_at is None or record.finished_at is None:
        return None
    return (record.finished_at - record.started_at).total_seconds()


def run_tags(record: RunRecord) -> list[str]:
    """Теги записи и сценария, без повторов."""
    scenario_tags = record.scenario.get("tags") or []
    return list(dict.fromkeys([*record.tags, *(str(tag) for tag in scenario_tags)]))


def dataset_matches(label: str, wanted: str) -> bool:
    """`hf:BeIR/nfcorpus` совпадает с любой ревизией, `hf:BeIR/nfcorpus@…` — только с этой."""
    return label == wanted or label.startswith(wanted + "@")


def _contains(text: str, values: Iterable[Any]) -> bool:
    needle = text.casefold()
    return any(needle in str(value).casefold() for value in values if value is not None)


def _in_period(at: datetime, since: datetime | None, until: datetime | None) -> bool:
    return (since is None or at >= since) and (until is None or at < until)


def _run_matches(r: RunRecord, q: RunQuery) -> bool:
    return (
        (q.scenario_id is None or r.scenario_id == q.scenario_id)
        and (q.name is None or r.name == q.name)
        and (q.state is None or r.state == q.state)
        and (q.tag is None or q.tag in run_tags(r))
        and _in_period(r.enqueued_at, q.since, q.until)
        and (q.dataset is None or dataset_matches(r.dataset, q.dataset))
        and (q.collection is None or r.collection == q.collection)
        and (q.text is None or _contains(q.text, [
            r.id, r.name, r.scenario.get("description"), *run_tags(r), r.dataset, r.collection,
        ]))
    )


def _sorted[T](items: list[T], sort: str, key_of: Callable[[T, str], Any], when: Callable[[T], datetime]) -> list[T]:
    """По ключу, `-` — по убыванию; null всегда в конце; при равенстве — новые выше."""
    descending, key = sort.startswith("-"), sort.removeprefix("-")
    items = sorted(items, key=when, reverse=True)
    present = [item for item in items if key_of(item, key) is not None]
    missing = [item for item in items if key_of(item, key) is None]
    return sorted(present, key=lambda item: key_of(item, key), reverse=descending) + missing


def _run_key(k: int) -> Callable[[RunRecord, str], Any]:
    def key_of(r: RunRecord, key: str) -> Any:
        if key in SUMMARY_COLUMNS:
            return summary_column(r, key, k)
        if key == "duration":
            return duration_s(r)
        if key == "scenario":
            return r.name
        return getattr(r, key)
    return key_of


def _page[T](items: list[T], offset: int, limit: int) -> Page[T]:
    return Page(items=items[offset:offset + limit], total=len(items))


def select_runs(records: Iterable[RunRecord], query: RunQuery) -> Page[RunRecord]:
    found = [r for r in records if _run_matches(r, query)]
    return _page(_sorted(found, query.sort, _run_key(query.k), lambda r: r.enqueued_at), query.offset, query.limit)


def _build_key(b: BuildRecord, key: str) -> Any:
    if key == "duration":
        return duration_s(b)
    if key == "profile":
        return b.name
    return getattr(b, key)


def select_builds(records: Iterable[BuildRecord], query: BuildQuery) -> Page[BuildRecord]:
    found = [
        b for b in records
        if (query.profile_id is None or b.profile_id == query.profile_id)
        and (query.state is None or b.state == query.state)
        and _in_period(b.enqueued_at, query.since, query.until)
        and (query.dataset is None or dataset_matches(b.dataset, query.dataset))
        and (query.collection is None or any(t.collection == query.collection for t in b.targets))
        and (query.text is None or _contains(query.text, [
            b.id, b.name, b.dataset, *(t.collection for t in b.targets),
        ]))
    ]
    return _page(_sorted(found, query.sort, _build_key, lambda b: b.enqueued_at), query.offset, query.limit)
