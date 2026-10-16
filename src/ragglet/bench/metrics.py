from __future__ import annotations


def project_to_docs(chunk_ids: list[str], chunk_to_doc: dict[str, str]) -> list[str]:
    """
    Проецирует последовательность chunk_id в doc_id, сохраняя лучший (первый) ранг и
    дедуплицируя. Если маппинг для чанка неизвестен — fallback: отрезаем `#idx`.
    Дедуп обязателен: иначе несколько чанков одного документа занимают несколько рангов.
    """
    out: list[str] = []
    seen: set[str] = set()
    for chunk_id in chunk_ids:
        doc_id = chunk_to_doc.get(chunk_id) or chunk_id.rsplit("#", 1)[0]
        if doc_id in seen:
            continue
        seen.add(doc_id)
        out.append(doc_id)
    return out


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