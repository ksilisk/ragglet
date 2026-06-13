from __future__ import annotations

import anyio
import pytest

from ragglet.generation.base import GenerationRequest
from ragglet.generation.noop import NoopGenerator
from ragglet.generation.prompt import build_prompt


def _scenario(**generation):
    pytest.importorskip("yaml")
    pytest.importorskip("pydantic")
    from ragglet.config.scenario import ScenarioConfig

    data = {
        "version": 1,
        "name": "generation-test",
        "dataset": {"id": "data/queries/truth_manifest.jsonl"},
        "storage": {
            "mode": "single",
            "backend": {
                "type": "qdrant",
                "endpoint": "http://localhost:6333",
                "collection": "chunks_v1",
            },
        },
        "context": {"enabled": True},
        "generation": {"enabled": False, **generation},
    }
    return ScenarioConfig.model_validate(data)


def test_prompt_contains_question_and_context():
    prompt = build_prompt("What is RAG?", "[doc#0]\nRAG uses retrieval.")

    assert "What is RAG?" in prompt
    assert "RAG uses retrieval." in prompt


def test_noop_generation_returns_usage_without_external_call():
    request = GenerationRequest(
        query="question",
        context_text="context",
        context_ids=["doc#0"],
        model="noop",
        temperature=0.0,
        max_output_tokens=10,
        prompt="question context",
    )

    result = anyio.run(NoopGenerator().generate, request)

    assert result.provider == "noop"
    assert result.answer == ""
    assert result.input_tokens == 2
    assert result.output_tokens == 0


def test_generation_requires_model_when_enabled():
    with pytest.raises(ValueError, match="generation.model is required"):
        _scenario(enabled=True, provider="noop", model=None)


def test_generation_requires_context_when_enabled():
    pytest.importorskip("yaml")
    pytest.importorskip("pydantic")
    from ragglet.config.scenario import ScenarioConfig

    with pytest.raises(ValueError, match="context.enabled must be true"):
        ScenarioConfig.model_validate(
            {
                "version": 1,
                "name": "generation-test",
                "dataset": {"id": "data/queries/truth_manifest.jsonl"},
                "storage": {
                    "mode": "single",
                    "backend": {
                        "type": "qdrant",
                        "endpoint": "http://localhost:6333",
                        "collection": "chunks_v1",
                    },
                },
                "context": {"enabled": False},
                "generation": {"enabled": True, "provider": "noop", "model": "noop"},
            }
        )


def test_summary_includes_generation_latency_and_tokens():
    pd = pytest.importorskip("pandas")
    from ragglet.bench.summarize import summarize_repeat

    cfg = _scenario(enabled=True, provider="noop", model="noop")
    cfg.metrics.latency_steps.append("generation")
    df = pd.DataFrame(
        [
            {
                "mrr": 1.0,
                "recall@1": 1.0,
                "recall@3": 1.0,
                "recall@5": 1.0,
                "recall@10": 1.0,
                "recall@30": 1.0,
                "lat_embed_ms": 1.0,
                "lat_retrieve_ms": 2.0,
                "lat_generation_ms": 5.0,
                "lat_total_ms": 8.0,
                "emb_hit_rate": 0.0,
                "ret_hit_rate": 0.0,
                "gen_input_tokens": 11,
                "gen_output_tokens": 3,
                "gen_total_tokens": 14,
                "generation_error": None,
                "timed_out": False,
                "timeout_stage": None,
            }
        ]
    )

    summary = summarize_repeat(cfg, df)

    assert summary["lat_generation_p50"] == 5.0
    assert summary["gen_input_tokens_mean"] == 11.0
    assert summary["generation_error_rate"] == 0.0
