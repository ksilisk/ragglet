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
