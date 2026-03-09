from __future__ import annotations

import math
import re
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [t.group(0).lower() for t in _TOKEN_RE.finditer(text)]


@dataclass(frozen=True)
class BM25Params:
    k1: float = 1.2
    b: float = 0.75


class BM25Index:
    def __init__(self, docs_tokens: list[list[str]], params: BM25Params = BM25Params()):
        self.params = params
        self.docs_tokens = docs_tokens
        self.N = len(docs_tokens)

        # df + avgdl
        self.doc_len = [len(toks) for toks in docs_tokens]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0

        df: dict[str, int] = {}
        for toks in docs_tokens:
            seen = set(toks)
            for w in seen:
                df[w] = df.get(w, 0) + 1
        self.df = df

        # idf (BM25Okapi)
        self.idf: dict[str, float] = {}
        for w, dfi in df.items():
            self.idf[w] = math.log(1 + (self.N - dfi + 0.5) / (dfi + 0.5))

        # tf per doc (sparse dict)
        self.tf: list[dict[str, int]] = []
        for toks in docs_tokens:
            tfd: dict[str, int] = {}
            for w in toks:
                tfd[w] = tfd.get(w, 0) + 1
            self.tf.append(tfd)

    def score(self, query_tokens: list[str], doc_idx: int) -> float:
        k1, b = self.params.k1, self.params.b
        dl = self.doc_len[doc_idx]
        denom_norm = (1 - b) + b * (dl / self.avgdl) if self.avgdl > 0 else 1.0

        tf = self.tf[doc_idx]
        s = 0.0
        for w in query_tokens:
            if w not in tf:
                continue
            f = tf[w]
            idf = self.idf.get(w, 0.0)
            s += idf * (f * (k1 + 1)) / (f + k1 * denom_norm)
        return s

    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        q = tokenize(query)
        if not q or self.N == 0:
            return []
        scored = [(i, self.score(q, i)) for i in range(self.N)]
        scored.sort(key=lambda x: x[1], reverse=True)
        # отсекаем нули
        out = [(i, sc) for i, sc in scored[:k] if sc > 0]
        return out
