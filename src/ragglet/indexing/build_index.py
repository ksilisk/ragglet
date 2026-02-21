from __future__ import annotations

from pathlib import Path
from hashlib import sha1
from typing import Iterable

from rich.progress import track

from ragglet.config.index_profile import IndexProfileConfig
from ragglet.indexing.parsers import parse_plain_text
from ragglet.indexing.chunking import recursive_chunk
from ragglet.modules.embedders.st_embedder import SentenceTransformerEmbedder
from ragglet.indexing.qdrant_indexer import QdrantIndexSpec, QdrantIndexer


def _iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def _make_chunk_id(doc_path: str, chunk_idx: int) -> str:
    """
    Стабильный id: sha1(path + idx) — чтобы при пересборке было воспроизводимо.
    Важно: это id должен попадать в ground-truth датасета.
    """
    raw = f"{doc_path}#{chunk_idx}".encode("utf-8")
    return sha1(raw).hexdigest()

from uuid import uuid5, NAMESPACE_URL

def _make_point_id(external_id: str) -> str:
    # детерминированный UUID по строке -> одинаковый при пересборке
    return str(uuid5(NAMESPACE_URL, external_id))

def build_index(cfg: IndexProfileConfig) -> None:
    corpus_root = Path(cfg.corpus)
    if not corpus_root.exists():
        raise FileNotFoundError(f"Corpus path not found: {corpus_root}")

    if cfg.storage.type != "qdrant":
        raise NotImplementedError("Only qdrant storage is implemented in the first iteration")

    embedder = SentenceTransformerEmbedder(cfg.embedding.model, normalize=cfg.embedding.normalize)

    # We need vector size before creating collection.
    # Easiest: embed a tiny sample.
    sample_vec = embedder.embed_batch(["vector_size_probe"])[0]
    vector_size = len(sample_vec)

    spec = QdrantIndexSpec(
        endpoint=cfg.storage.endpoint,
        collection=cfg.storage.collection,
        vector_size=vector_size,
        distance=cfg.index_params.distance,
        hnsw_m=cfg.index_params.hnsw.m,
        hnsw_ef_construct=cfg.index_params.hnsw.ef_construct,
    )
    q = QdrantIndexer(spec)
    q.recreate_collection()

    batch_texts: list[str] = []
    batch_ids: list[str] = []
    batch_payloads: list[dict] = []

    def flush() -> None:
        if not batch_ids:
            return
        vectors = embedder.embed_batch(batch_texts)
        q.upsert(batch_ids, vectors, batch_payloads)
        batch_texts.clear()
        batch_ids.clear()
        batch_payloads.clear()

    for file_path in track(list(_iter_files(corpus_root)), description="Indexing corpus"):
        text = parse_plain_text(file_path)
        chunks = recursive_chunk(text, cfg.chunking.chunk_size, cfg.chunking.chunk_overlap)
        rel_path = str(file_path.relative_to(corpus_root))

        for idx, chunk_text in enumerate(chunks):
            external_id = f"{rel_path}#{idx}"
            point_id = _make_point_id(external_id)
            batch_ids.append(point_id)
            batch_texts.append(chunk_text)
            batch_payloads.append(
                {
                    "external_id": external_id,
                    "doc_path": rel_path,
                    "chunk_index": idx,
                    "text": chunk_text,   # полезно для отладки/потом контекст строить
                }
            )

            if len(batch_ids) >= cfg.embedding.batch_size:
                flush()

    flush()