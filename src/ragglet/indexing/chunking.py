from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    metadata: dict


def recursive_chunk(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Простая реализация: режем по символам с overlap.
    Для НИР-2 достаточно. Потом можно улучшить: по предложениям/абзацам.
    """
    text = text.strip()
    if not text:
        return []

    step = max(1, chunk_size - chunk_overlap)
    chunks = []
    for start in range(0, len(text), step):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
    return chunks