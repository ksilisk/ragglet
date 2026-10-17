from __future__ import annotations

from dataclasses import dataclass

from ragglet.config.index_profile import ChunkingConfig


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


def chunk_document(text: str, cfg: ChunkingConfig) -> list[str]:
    if cfg.method == "none":
        text = text.strip()
        return [text] if text else []
    if cfg.method == "recursive":
        assert cfg.chunk_size is not None and cfg.chunk_overlap is not None
        return recursive_chunk(text, cfg.chunk_size, cfg.chunk_overlap)
    raise NotImplementedError(f"chunking.method='{cfg.method}' is not implemented")
