from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

from ragglet.config.index_profile import IndexProfileConfig
from ragglet.core.environment import capture_env
from ragglet.core.issues import Issue
from ragglet.core.progress import Progress, no_progress
from ragglet.datasets.beir import document_text, read_corpus
from ragglet.datasets.checks import validate_dataset
from ragglet.datasets.sources import describe_dataset, resolve_dataset
from ragglet.indexing.chunking import chunk_document
from ragglet.indexing.metadata import RAGGLET_METADATA_KEY, build_index_metadata, chunk_point_id
from ragglet.indexing.qdrant_indexer import QdrantIndexer, QdrantIndexSpec
from ragglet.modules.embedders.st_embedder import SentenceTransformerEmbedder
from ragglet.storage.layout import StorageTarget, build_storage_layout, require_implemented


@dataclass(frozen=True)
class BuildResult:
    dataset: str
    corpus_sha256: str
    documents: int
    targets: list[StorageTarget]  # коллекции, которые пересоздаются
    warnings: list[Issue]  # валидации датасета
    index: dict[str, Any] | None = None  # записанные метаданные ragglet; у dry_run их нет


def build_index(
        cfg: IndexProfileConfig,
        *,
        dry_run: bool = False,
        progress: Progress = no_progress,
) -> BuildResult:
    """Собирает индекс; `dry_run` — только проверки датасета и storage, без модели и Qdrant."""
    layout = build_storage_layout(cfg.storage)
    require_implemented(layout)

    dataset = resolve_dataset(cfg.dataset.source, cfg.dataset.id, cfg.dataset.revision)
    dataset_dir = dataset.path
    label = describe_dataset(cfg.dataset.source, cfg.dataset.id, dataset.revision)
    # весь датасет со всеми split — до эмбеддинга, а не после
    parsed, warnings = validate_dataset(dataset_dir, None, label=label)
    result = BuildResult(
        dataset=label,
        corpus_sha256=parsed.corpus_sha256,
        documents=len(parsed.corpus),
        targets=layout.targets,
        warnings=warnings,
    )
    if dry_run:
        return result

    embedder = SentenceTransformerEmbedder(
        cfg.embedding.model, normalize=cfg.embedding.normalize, revision=cfg.embedding.revision
    )
    build_env = capture_env(embedder.compute_info())

    # We need vector size before creating collection.
    # Easiest: embed a tiny sample.
    sample_vec = embedder.embed_batch(["vector_size_probe"])[0]
    vector_size = len(sample_vec)

    indexers: dict[str, QdrantIndexer] = {}
    for target in layout.targets:
        spec = QdrantIndexSpec(
            endpoint=target.endpoint,
            collection=target.collection,
            vector_size=vector_size,
            distance=cfg.index_params.distance,
            hnsw_m=cfg.index_params.hnsw.m,
            hnsw_ef_construct=cfg.index_params.hnsw.ef_construct,
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

    docs = list(read_corpus(dataset_dir))
    for n_done, doc in enumerate(docs, start=1):
        chunks = chunk_document(document_text(doc), cfg.chunking)

        for idx, chunk_text in enumerate(chunks):
            chunk_id = f"{doc.doc_id}#{idx}"
            point_id = chunk_point_id(chunk_id)
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

        progress("corpus", n_done, len(docs))

    flush()

    # метаданные — только после всех upsert: их наличие и означает, что индекс полный
    metadata = build_index_metadata(
        cfg,
        built_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        corpus_sha256=parsed.corpus_sha256,
        embedding_revision=embedder.revision,
        storage_mode=layout.mode,
        storage_backend_count=len(layout.targets),
        build_env=build_env,
    )
    for indexer in indexers.values():
        indexer.write_metadata(metadata)
    return replace(result, index=metadata[RAGGLET_METADATA_KEY])
