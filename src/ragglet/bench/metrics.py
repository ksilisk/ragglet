from __future__ import annotations

def recall_at_k(retrieved: list[str], truth: set[str], k: int) -> float:
    if not truth:
        return 0.0
    topk = retrieved[:k]
    return 1.0 if any(doc_id in truth for doc_id in topk) else 0.0

def mrr_at_k(retrieved: list[str], truth: set[str], k: int) -> float:
    topk = retrieved[:k]
    for rank, doc_id in enumerate(topk, start=1):
        if doc_id in truth:
            return 1.0 / rank
    return 0.0