from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ragglet.keyword.bm25 import BM25Index, BM25Params, tokenize


@dataclass(frozen=True)
class Candidate:
    id: str
    score_raw: float
    payload: dict[str, Any] | None


class KeywordBm25Store:
    """
    In-memory BM25 поверх коллекции чанков.
    """

    def __init__(self, docs: list[dict[str, Any]], k1: float = 1.2, b: float = 0.75):
        """
        docs: список документов/чанков вида:
          {
            "chunk_id": str,
            "text": str,
            ... любые поля (включая doc_id) ...
          }
        """
        self.docs = docs
        docs_tokens = [tokenize(d.get("text", "")) for d in docs]
        self.index = BM25Index(docs_tokens, BM25Params(k1=k1, b=b))

    def search(self, query: str, top_k: int) -> list[Candidate]:
        pairs = self.index.top_k(query, top_k)
        out: list[Candidate] = []
        for doc_idx, sc in pairs:
            d = self.docs[doc_idx]
            cid = str(d.get("chunk_id"))
            payload = dict(d)
            # chunk_id должен быть в payload (dedup); doc_id уже внутри d и переносится
            payload["chunk_id"] = cid
            out.append(Candidate(id=cid, score_raw=float(sc), payload=payload))
        return out
