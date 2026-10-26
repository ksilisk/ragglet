from __future__ import annotations

import pytest

from ragglet.config.index_profile import ChunkingConfig
from ragglet.indexing.chunking import chunk_document
from ragglet.retrieval.engine import _to_retrieved
from ragglet.stores.qdrant_store_async import Candidate


def test_to_retrieved_takes_doc_id_from_payload():
    item = _to_retrieved(
        Candidate(id="p1", score_raw=0.7, payload={"chunk_id": "d1#2", "doc_id": "d1"}),
        "vec",
    )
    assert (item.chunk_id, item.doc_id, item.source) == ("d1#2", "d1", "vec")


@pytest.mark.parametrize("payload", [{"chunk_id": "d1#0"}, {"doc_id": "d1"}, None])
def test_to_retrieved_rejects_outdated_payload(payload):
    with pytest.raises(ValueError, match="rebuild"):
        _to_retrieved(Candidate(id="p1", score_raw=0.7, payload=payload), "vec")


def test_chunking_none_indexes_whole_document():
    cfg = ChunkingConfig(method="none")
    assert (cfg.chunk_size, cfg.chunk_overlap) == (None, None)
    text = "word " * 1000
    assert chunk_document(text, cfg) == [text.strip()]
    assert chunk_document("   ", cfg) == []


@pytest.mark.parametrize("field", ["chunk_size", "chunk_overlap"])
def test_chunking_none_rejects_sizes(field):
    with pytest.raises(ValueError, match="not allowed"):
        ChunkingConfig.model_validate({"method": "none", field: 100})


def test_chunking_recursive_defaults_and_overlap_check():
    cfg = ChunkingConfig()
    assert (cfg.method, cfg.chunk_size, cfg.chunk_overlap) == ("recursive", 800, 120)
    assert len(chunk_document("x" * 2000, cfg)) == 3
    with pytest.raises(ValueError, match="chunk_overlap must be less"):
        ChunkingConfig(chunk_size=100, chunk_overlap=100)


async def test_payloads_by_chunk_id_use_build_point_ids():
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.http import models as qm

    from ragglet.indexing.metadata import chunk_point_id
    from ragglet.stores.qdrant_store_async import AsyncQdrantStore

    store = AsyncQdrantStore("http://unused:6333", "chunks_v1")
    store.client = AsyncQdrantClient(location=":memory:")
    await store.client.create_collection("chunks_v1", vectors_config=qm.VectorParams(size=2, distance=qm.Distance.COSINE))
    await store.client.upsert("chunks_v1", points=[
        qm.PointStruct(id=chunk_point_id(c), vector=[1.0, 0.0], payload={"chunk_id": c, "doc_id": c.split("#")[0],
                                                                         "title": "T", "text": f"text of {c}"})
        for c in ("d1#0", "d1#1", "d2#0")
    ])

    found = await store.payloads(["d1#1", "d2#0", "missing#0"])
    await store.close()

    assert {c: p["text"] for c, p in found.items()} == {"d1#1": "text of d1#1", "d2#0": "text of d2#0"}
