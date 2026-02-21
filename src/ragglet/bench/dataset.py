from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass(frozen=True)
class QueryItem:
    qid: str
    query: str
    truth_ids: list[str]

def load_jsonl_dataset(path: Path, query_field: str, truth_field: str, limit: int) -> list[QueryItem]:
    items: list[QueryItem] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and len(items) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = str(obj.get("id", i))
            query = str(obj[query_field])
            truth = obj.get(truth_field, [])
            if isinstance(truth, str):
                truth = [truth]
            items.append(QueryItem(qid=qid, query=query, truth_ids=[str(x) for x in truth]))
    return items