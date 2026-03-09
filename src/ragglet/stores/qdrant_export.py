from __future__ import annotations
from typing import Any

# Подстрой под свой клиент; названия методов могут слегка отличаться по версии.
# Идея: scroll пачками, with_payload=True.

async def export_all_payloads(store, batch_size: int = 256) -> list[dict[str, Any]]:
    """
    Возвращает список payload для всех points в коллекции.
    Требуются payload поля: external_id, text (и любые другие полезные).
    """
    out: list[dict[str, Any]] = []
    offset = None

    while True:
        # В qdrant-client встречаются варианты: scroll, scroll_points, query_points+offset.
        # У тебя уже AsyncQdrantStore — лучше сделать метод store.scroll_payloads().
        page, offset = await store.scroll_payloads(limit=batch_size, offset=offset)
        out.extend(page)
        if offset is None:
            break

    return out