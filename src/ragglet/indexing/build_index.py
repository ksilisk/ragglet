from __future__ import annotations

from pathlib import Path
from hashlib import sha1
from typing import Iterable
from copy import deepcopy

from rich.progress import track

from ragglet.config.index_profile import IndexProfileConfig
from ragglet.indexing.parsers import parse_plain_text
from ragglet.indexing.chunking import recursive_chunk
from ragglet.indexing.metadata import build_index_metadata
from ragglet.modules.embedders.st_embedder import SentenceTransformerEmbedder
from ragglet.indexing.qdrant_indexer import QdrantIndexSpec, QdrantIndexer
from ragglet.storage.layout import StorageTarget, build_storage_layout


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

    layout = build_storage_layout(cfg.storage)
    if any(target.type != "qdrant" for target in layout.targets):
        raise NotImplementedError("Only qdrant storage is implemented in the first iteration")

    embedder = SentenceTransformerEmbedder(cfg.embedding.model, normalize=cfg.embedding.normalize)

    # We need vector size before creating collection.
    # Easiest: embed a tiny sample.
    sample_vec = embedder.embed_batch(["vector_size_probe"])[0]
    vector_size = len(sample_vec)

    def target_metadata(target: StorageTarget) -> dict:
        metadata = deepcopy(build_index_metadata(cfg))
        metadata["ragglet"].update(
            {
                "storage_mode": layout.mode,
                "storage_backend_id": target.id,
                "storage_backend_count": len(layout.targets),
                "storage_routing_key": layout.routing_key,
            }
        )
        return metadata

    indexers: dict[str, QdrantIndexer] = {}
    for target in layout.targets:
        spec = QdrantIndexSpec(
            endpoint=target.endpoint,
            collection=target.collection,
            vector_size=vector_size,
            distance=cfg.index_params.distance,
            hnsw_m=cfg.index_params.hnsw.m,
            hnsw_ef_construct=cfg.index_params.hnsw.ef_construct,
            metadata=target_metadata(target),
        )
        indexer = QdrantIndexer(spec)
        indexer.recreate_collection()
        indexers[target.id] = indexer

    batch_texts: list[str] = []
    batch_ids: list[str] = []
    batch_payloads: list[dict] = []
    batch_targets: list[list[StorageTarget]] = []

    def flush() -> None:
        if not batch_ids:
            return
        vectors = embedder.embed_batch(batch_texts)
        grouped: dict[str, tuple[list[str], list[list[float]], list[dict]]] = {}
        for idx, targets in enumerate(batch_targets):
            for target in targets:
                ids, vecs, payloads = grouped.setdefault(target.id, ([], [], []))
                ids.append(batch_ids[idx])
                vecs.append(vectors[idx])
                payloads.append(batch_payloads[idx])
        for target_id, (ids, vecs, payloads) in grouped.items():
            indexers[target_id].upsert(ids, vecs, payloads)
        batch_texts.clear()
        batch_ids.clear()
        batch_payloads.clear()
        batch_targets.clear()

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
            batch_targets.append(layout.route_write(batch_payloads[-1]))

            if len(batch_ids) >= cfg.embedding.batch_size:
                flush()

    flush()
