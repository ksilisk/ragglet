from __future__ import annotations

from dataclasses import dataclass

from ragglet.datasets.beir import read_queries
from ragglet.datasets.sources import resolve_dataset_dir


@dataclass(frozen=True)
class QueryItem:
    qid: str
    query: str
    qrels: dict[str, int]  # relevant doc_id -> grade (> 0), document-level ground truth

    @property
    def truth_ids(self) -> list[str]:
        return sorted(self.qrels)


def load_dataset_queries(source: str, dataset_id: str, split: str, limit: int) -> list[QueryItem]:
    """
    Грузит запросы+qrels из BEIR-датасета (local или hf) как список QueryItem с
    document-level ground truth.
    """
    dataset_dir = resolve_dataset_dir(source, dataset_id)
    queries = read_queries(dataset_dir, split)
    items = [QueryItem(qid=q.qid, query=q.text, qrels=dict(q.qrels)) for q in queries]
    if limit:
        items = items[:limit]
    return items
