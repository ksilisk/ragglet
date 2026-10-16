from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cachetools import LRUCache

if TYPE_CHECKING:
    from ragglet.retrieval.merge import RetrievedItem


@dataclass(frozen=True)
class RetrievalCacheEntry:
    retrieved_ids: list[str]
    retrieved_items: list[RetrievedItem] | None = None
    # doc_id параллельно retrieved_ids (та же длина) — для восстановления chunk->doc
    # маппинга на items-less cache hit
    chunk_doc_ids: list[str] | None = None


class LRURetrievalCache:
    def __init__(self, size: int):
        self.cache = LRUCache(maxsize=int(size))

    @staticmethod
    def make_key(
        scenario_name: str,
        query_text: str,
        top_k: int,
        strategy: str,
        sources_sig: Any,
        merge_sig: Any,
    ) -> str:
        """
        Делай ключ зависимым от того, что влияет на результат retrieval.
        """
        payload = {
            "scenario": scenario_name,
            "q": query_text,
            "top_k": top_k,
            "strategy": strategy,
            "sources": sources_sig,
            "merge": merge_sig,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha1(raw).hexdigest()

    def get(self, key: str) -> RetrievalCacheEntry | None:
        return self.cache.get(key)

    def set(self, key: str, entry: RetrievalCacheEntry) -> None:
        self.cache[key] = entry
