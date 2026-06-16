from __future__ import annotations

from dataclasses import dataclass, field

from ragglet.stores.qdrant_store_async import AsyncQdrantStore
from ragglet.stores.bm25_store import KeywordBm25Store
from ragglet.stores.qdrant_export import export_all_payloads


@dataclass
class BackendRegistry:
    vector: AsyncQdrantStore
    vectors: dict[str, AsyncQdrantStore] = field(default_factory=dict)
    keyword: KeywordBm25Store | None = None

    def vector_for(self, backend_ref: str | None) -> AsyncQdrantStore:
        if backend_ref is None:
            return self.vector
        try:
            return self.vectors[backend_ref]
        except KeyError as e:
            raise KeyError(f"Unknown vector backend_ref: {backend_ref}") from e

    @classmethod
    async def build(
        cls,
        cfg,
        vector_store: AsyncQdrantStore,
        vector_stores: dict[str, AsyncQdrantStore] | None = None,
    ) -> "BackendRegistry":
        vector_stores = vector_stores or {"default": vector_store}
        reg = cls(vector=vector_store, vectors=vector_stores, keyword=None)

        kw_sources = [s for s in (cfg.retrieval.sources or []) if s.kind == "keyword"]
        if kw_sources:
            if len(kw_sources) > 1:
                raise NotImplementedError("Multiple keyword sources are not supported yet (different k1/b).")

            kw = kw_sources[0]
            params = kw.params or {}
            k1 = float(params.get("k1", 1.2))
            b = float(params.get("b", 0.75))

            payloads = []
            seen_external_ids: set[str] = set()
            for store in vector_stores.values():
                for payload in await export_all_payloads(store, batch_size=256):
                    external_id = payload.get("external_id") if payload else None
                    if external_id is not None and str(external_id) in seen_external_ids:
                        continue
                    if external_id is not None:
                        seen_external_ids.add(str(external_id))
                    payloads.append(payload)
            docs = []
            for p in payloads:
                if p and "external_id" in p and "text" in p:
                    docs.append({"external_id": p["external_id"], "text": p["text"], **p})

            reg.keyword = KeywordBm25Store(docs, k1=k1, b=b)

        return reg
