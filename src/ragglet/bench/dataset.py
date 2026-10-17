from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from ragglet.config.scenario import DatasetConfig
from ragglet.core.issues import Issue
from ragglet.datasets.beir import ParsedDataset
from ragglet.datasets.checks import validate_dataset
from ragglet.datasets.sources import describe_dataset, resolve_dataset


@dataclass(frozen=True)
class QueryItem:
    qid: str
    query: str
    qrels: dict[str, int]  # relevant doc_id -> grade (> 0), document-level ground truth

    @property
    def truth_ids(self) -> list[str]:
        return sorted(self.qrels)


@dataclass(frozen=True)
class EvalDataset:
    revision: str | None
    items: list[QueryItem]
    corpus_sha256: str
    eval_sha256: str
    warnings: list[Issue]


def eval_items(ds: ParsedDataset, split: str, limit: int) -> list[QueryItem]:
    """Запросы split с релевантными документами, по qid; первые `limit`."""
    relevant: dict[str, dict[str, int]] = {}
    for record in ds.qrels[split]:
        if record.grade > 0:
            relevant.setdefault(record.qid, {})[record.doc_id] = record.grade
    items = [
        QueryItem(qid=qid, query=ds.query_index[qid].text, qrels=relevant[qid])
        for qid in sorted(relevant)
    ]
    return items[:limit] if limit else items


def eval_sha256(items: list[QueryItem]) -> str:
    payload = [[it.qid, it.query, sorted(it.qrels.items())] for it in sorted(items, key=lambda it: it.qid)]
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_eval_dataset(cfg: DatasetConfig) -> EvalDataset:
    """Валидирует датасет сценария (corpus + его split) и собирает eval-набор с хэшами."""
    resolved = resolve_dataset(cfg.source, cfg.id, cfg.revision)
    label = describe_dataset(cfg.source, cfg.id, resolved.revision)
    ds, warnings = validate_dataset(resolved.path, [cfg.split], label=label)
    items = eval_items(ds, cfg.split, cfg.limit)
    return EvalDataset(
        revision=resolved.revision,
        items=items,
        corpus_sha256=ds.corpus_sha256,
        eval_sha256=eval_sha256(items),
        warnings=warnings,
    )
