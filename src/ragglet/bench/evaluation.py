from __future__ import annotations

from collections.abc import Iterable, Sequence

from ragglet.bench.metrics import METRICS, Ranking
from ragglet.config.scenario import MetricsQualityConfig
from ragglet.retrieval.merge import RetrievedItem


def _dedup(ids: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in ids:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def project_to_docs(items: Iterable[RetrievedItem]) -> list[str]:
    """
    doc-level ранжирование: документ занимает ранг своего лучшего (первого) чанка.
    Без дедупа несколько чанков одного документа заняли бы несколько рангов.
    """
    return _dedup(item.doc_id for item in items)


def quality_columns(cfg: MetricsQualityConfig) -> list[str]:
    return [
        f"{level}_{name}@{k}"
        for level, names in (("doc", cfg.doc), ("chunk", cfg.chunk))
        for name in names
        for k in cfg.k
    ]


def evaluate_quality(
    cfg: MetricsQualityConfig,
    items: Sequence[RetrievedItem],
    qrels: dict[str, int],
) -> dict[str, float | int]:
    """
    Оценка финального списка чанков против doc-level qrels на двух уровнях:
    doc — уникальные doc_id; chunk — уникальные chunk_id с релевантностью своего документа.
    Точные дубликаты чанка схлопываются и не добавляют релевантности.
    """
    doc_of = {item.chunk_id: item.doc_id for item in items}
    chunk_ids = _dedup(item.chunk_id for item in items)
    doc_ids = project_to_docs(items)

    levels = {
        "doc": (cfg.doc, Ranking(
            grades=[qrels.get(doc_id, 0) for doc_id in doc_ids],
            ideal=sorted(qrels.values(), reverse=True),
        )),
        "chunk": (cfg.chunk, Ranking(
            grades=[qrels.get(doc_of[chunk_id], 0) for chunk_id in chunk_ids],
            ideal=None,
        )),
    }

    row: dict[str, float | int] = {}
    for level, (names, ranking) in levels.items():
        for name in names:
            for k in cfg.k:
                row[f"{level}_{name}@{k}"] = METRICS[name].fn(ranking, k)
    row["n_chunks"] = len(chunk_ids)
    row["n_docs"] = len(doc_ids)
    row["dup_chunks"] = len(items) - len(chunk_ids)
    return row
