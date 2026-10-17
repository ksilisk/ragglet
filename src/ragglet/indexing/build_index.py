from __future__ import annotations

from copy import deepcopy
from uuid import NAMESPACE_URL, uuid5

from rich.progress import track

from ragglet.config.index_profile import IndexProfileConfig
from ragglet.datasets.beir import read_corpus
from ragglet.datasets.sources import resolve_dataset_dir
from ragglet.indexing.chunking import chunk_document
from ragglet.indexing.metadata import build_index_metadata
from ragglet.indexing.qdrant_indexer import QdrantIndexer, QdrantIndexSpec
from ragglet.modules.embedders.st_embedder import SentenceTransformerEmbedder
from ragglet.storage.layout import StorageTarget, build_storage_layout


def _make_point_id(chunk_id: str) -> str:
    # детерминированный UUID по строке -> одинаковый при пересборке
    return str(uuid5(NAMESPACE_URL, chunk_id))


def build_index(cfg: IndexProfileConfig) -> None:
    dataset_dir = resolve_dataset_dir(cfg.dataset.source, cfg.dataset.id)

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

    for doc in track(list(read_corpus(dataset_dir)), description="Indexing corpus"):
        doc_text = f"{doc.title}\n\n{doc.text}".strip() if doc.title else doc.text
        chunks = chunk_document(doc_text, cfg.chunking)

        for idx, chunk_text in enumerate(chunks):
            chunk_id = f"{doc.doc_id}#{idx}"
            point_id = _make_point_id(chunk_id)
            batch_ids.append(point_id)
            batch_texts.append(chunk_text)
            batch_payloads.append(
                {
                    "doc_id": doc.doc_id,          # id документа (BEIR corpus _id) — ground truth
                    "chunk_id": chunk_id,          # внутренний id чанка: dedup/debug
                    "chunk_index": idx,
                    "title": doc.title,
                    "text": chunk_text,            # полезно для отладки/контекста
                }
            )
            batch_targets.append(layout.route_write(batch_payloads[-1]))

            if len(batch_ids) >= cfg.embedding.batch_size:
                flush()

    flush()
