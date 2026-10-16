from __future__ import annotations

import time
from typing import Any

import anyio
import pandas as pd

from ragglet.bench.metrics import mrr_at_k, project_to_docs, recall_at_k
from ragglet.cache.manager import CacheManager
from ragglet.config.scenario import ScenarioConfig
from ragglet.context.base import ContextResult
from ragglet.context.engine import assemble_context
from ragglet.core.errors import StageTimeout
from ragglet.generation.base import Generator
from ragglet.generation.engine import generate_answer
from ragglet.modules.embedders.st_embedder import SentenceTransformerEmbedder
from ragglet.rerank.base import Reranker
from ragglet.rerank.engine import rerank_items
from ragglet.retrieval.backends import BackendRegistry
from ragglet.retrieval.engine import retrieve_items


def _chunk_to_doc_map(items) -> dict[str, str]:
    out: dict[str, str] = {}
    for it in items:
        payload = it.payload or {}
        out[it.chunk_id] = str(payload.get("doc_id", it.chunk_id))
    return out


async def run_queries_once(
        cfg: ScenarioConfig,
        items,
        backends: BackendRegistry,
        embedder: SentenceTransformerEmbedder,
        caches: CacheManager,
        reranker: Reranker,
        generator: Generator,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def embed_one(q: str) -> list[float]:
        return embedder.embed_batch([q])[0]

    for it in items:
        t0 = time.perf_counter()
        timed_out = False
        timeout_stage = None
        retrieved_ids = []
        reranked_ids = []
        context_ids = []
        final_ids = []
        retrieved_doc_ids = []
        final_doc_ids = []
        chunk_to_doc: dict[str, str] = {}
        rerank_scores = []
        context_text = ""
        context_tokens = 0
        context_chunks = 0
        answer = ""
        prompt_text = ""
        raw_generation_response = None
        generation_error = None
        gen_input_tokens = 0
        gen_output_tokens = 0
        gen_total_tokens = 0
        gen_cost_estimated = None
        lat_embed_ms = lat_retrieve_ms = lat_rerank_ms = lat_context_ms = lat_generation_ms = lat_total_ms = 0.0

        try:
            with anyio.fail_after(cfg.timeouts.total_ms / 1000.0):
                t1 = time.perf_counter()
                vec = await caches.get_or_compute_embedding(it.query, embed_one)
                t2 = time.perf_counter()

                # retrieval cache
                key = caches.make_retrieval_key(it.query)
                cached = caches.get_retrieval(key)
                if cached is not None and cached.retrieved_items is not None:
                    retrieved_items = cached.retrieved_items
                    retrieved_ids = cached.retrieved_ids
                    chunk_to_doc = _chunk_to_doc_map(retrieved_items)
                elif cached is not None and not cfg.rerank.enabled and not cfg.context.enabled:
                    retrieved_ids = cached.retrieved_ids
                    retrieved_items = []
                    # items-less hit: восстанавливаем chunk->doc из параллельного списка
                    chunk_to_doc = dict(zip(retrieved_ids, cached.chunk_doc_ids or [], strict=False))
                else:
                    retrieved_items = await retrieve_items(cfg, backends, it.query, vec)
                    retrieved_ids = [item.chunk_id for item in retrieved_items]
                    chunk_to_doc = _chunk_to_doc_map(retrieved_items)
                    chunk_doc_ids = [chunk_to_doc[cid] for cid in retrieved_ids]
                    caches.set_retrieval(key, retrieved_ids, retrieved_items, chunk_doc_ids)

                # doc-level список retrieved (дедуп по лучшему рангу) — для отладки
                retrieved_doc_ids = project_to_docs(retrieved_ids, chunk_to_doc)

                t3 = time.perf_counter()

                if cfg.rerank.enabled:
                    reranked = await rerank_items(cfg, reranker, it.query, retrieved_items)
                    reranked_ids = [result.item.chunk_id for result in reranked]
                    rerank_scores = [result.score for result in reranked]
                    context_candidates = reranked
                    final_ids = reranked_ids
                else:
                    context_candidates = retrieved_items
                    final_ids = retrieved_ids

                t4 = time.perf_counter()

                context = assemble_context(cfg, context_candidates)
                if cfg.context.enabled:
                    context_ids = context.ids
                    context_text = context.text if cfg.output.include_debug_fields else ""
                    context_tokens = context.token_count
                    context_chunks = len(context.chunks)
                    final_ids = context_ids
                else:
                    context = ContextResult(chunks=[], text="", token_count=0)

                t5 = time.perf_counter()

                if cfg.generation.enabled:
                    try:
                        generation, prompt = await generate_answer(cfg, generator, it.query, context)
                        answer = generation.answer if cfg.generation.save.answer else ""
                        prompt_text = prompt if cfg.generation.save.prompt else ""
                        raw_generation_response = (
                            generation.raw_response if cfg.generation.save.raw_response else None
                        )
                        gen_input_tokens = generation.input_tokens
                        gen_output_tokens = generation.output_tokens
                        gen_total_tokens = generation.total_tokens
                        gen_cost_estimated = generation.cost_estimated
                    except StageTimeout:
                        raise
                    except Exception as e:
                        generation_error = str(e)

                t6 = time.perf_counter()

                lat_embed_ms = (t2 - t1) * 1000.0
                lat_retrieve_ms = (t3 - t2) * 1000.0
                lat_rerank_ms = (t4 - t3) * 1000.0
                lat_context_ms = (t5 - t4) * 1000.0
                lat_generation_ms = (t6 - t5) * 1000.0
        except StageTimeout as e:
            timed_out = True
            timeout_stage = e.stage
        except TimeoutError:
            timed_out = True
            timeout_stage = "total"
        finally:
            lat_total_ms = (time.perf_counter() - t0) * 1000.0

        # метрики считаем на document-level: проецируем финальные chunk_id в doc_id с дедупом
        truth = set(it.truth_ids)
        final_doc_ids = project_to_docs(final_ids, chunk_to_doc)
        rec = {f"recall@{k}": recall_at_k(final_doc_ids, truth, k) for k in cfg.metrics.quality.recall_at_k}
        rr = mrr_at_k(final_doc_ids, truth, cfg.metrics.quality.mrr_at_k)

        rows.append(
            {
                "qid": it.qid,
                "query": it.query,
                "truth_ids": it.truth_ids,
                "retrieved_ids": retrieved_ids,
                "retrieved_doc_ids": retrieved_doc_ids,
                "reranked_ids": reranked_ids,
                "context_ids": context_ids,
                "context_text": context_text,
                "context_tokens": context_tokens,
                "context_chunks": context_chunks,
                "answer": answer,
                "prompt_text": prompt_text,
                "raw_generation_response": raw_generation_response,
                "generation_error": generation_error,
                "gen_input_tokens": gen_input_tokens,
                "gen_output_tokens": gen_output_tokens,
                "gen_total_tokens": gen_total_tokens,
                "gen_cost_estimated": gen_cost_estimated,
                "final_ids": final_ids,
                "final_doc_ids": final_doc_ids,
                "rerank_scores": rerank_scores,
                "mrr": rr,
                **rec,
                "lat_embed_ms": lat_embed_ms,
                "lat_retrieve_ms": lat_retrieve_ms,
                "lat_rerank_ms": lat_rerank_ms,
                "lat_context_ms": lat_context_ms,
                "lat_generation_ms": lat_generation_ms,
                "lat_total_ms": lat_total_ms,
                "emb_hit_rate": caches.stats.emb_hit_rate,
                "ret_hit_rate": caches.stats.ret_hit_rate,
                "timed_out": timed_out,
                "timeout_stage": timeout_stage,
            }
        )

    return pd.DataFrame(rows)
