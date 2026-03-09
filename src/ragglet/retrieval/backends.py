from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ragglet.stores.qdrant_store_async import AsyncQdrantStore
from ragglet.stores.bm25_store import KeywordBm25Store
from ragglet.stores.qdrant_export import export_all_payloads


@dataclass
class BackendRegistry:
    vector: AsyncQdrantStore
    keyword: KeywordBm25Store | None = None

    @classmethod
    async def build(cls, cfg, vector_store: AsyncQdrantStore) -> "BackendRegistry":
        reg = cls(vector=vector_store, keyword=None)

        kw_sources = [s for s in (cfg.retrieval.sources or []) if s.kind == "keyword"]
        if kw_sources:
            if len(kw_sources) > 1:
                raise NotImplementedError("Multiple keyword sources are not supported yet (different k1/b).")

            kw = kw_sources[0]
            params = kw.params or {}
            k1 = float(params.get("k1", 1.2))
            b = float(params.get("b", 0.75))

            payloads = await export_all_payloads(vector_store, batch_size=256)
            docs = []
            for p in payloads:
                if p and "external_id" in p and "text" in p:
                    docs.append({"external_id": p["external_id"], "text": p["text"], **p})

            reg.keyword = KeywordBm25Store(docs, k1=k1, b=b)

        return reg