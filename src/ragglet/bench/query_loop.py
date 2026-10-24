from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

import anyio
import anyio.lowlevel
import pandas as pd

from ragglet.bench.evaluation import evaluate_quality, project_to_docs
from ragglet.bench.load import LoadModel, build_load_model
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
from ragglet.retrieval.merge import RetrievedItem


class _Launcher:
    """
    Ядро прогона: выполняет запросы по вызовам модели нагрузки и ведёт учёт. Гарантии не
    зависят от модели: каждый запрос запущен ровно один раз, строки в порядке списка,
    первая неожиданная ошибка останавливает всё и выходит наружу как есть, с qid.
    """

    def __init__(
            self,
            items: list,
            run_one: Callable[[Any, float], Awaitable[dict[str, Any]]],
            on_done: Callable[[], None],
    ):
        self._items = items
        self._run_one = run_one
        self._on_done = on_done
        self._rows: list[dict[str, Any] | None] = [None] * len(items)
        self._launched = [False] * len(items)
        self._fatal: Exception | None = None
        self._scope: anyio.CancelScope | None = None

    async def __call__(self, i: int, planned_at: float | None = None) -> None:
        try:
            if not 0 <= i < len(self._items):
                raise IndexError(f"load model launched query #{i} of {len(self._items)}")
            if self._launched[i]:
                raise RuntimeError(f"load model launched query #{i} twice")
            self._launched[i] = True
            if planned_at is None:
                planned_at = anyio.current_time()
            self._rows[i] = await self._run_one(self._items[i], planned_at)
            self._on_done()
        except Exception as e:
            if self._fatal is None:
                if 0 <= i < len(self._items):
                    e.add_note(f"qid={self._items[i].qid}")
                self._fatal = e
            self._scope.cancel()
            await anyio.lowlevel.checkpoint()

    async def run(self, model: LoadModel) -> list[dict[str, Any]]:
        with anyio.CancelScope() as self._scope:
            await model.drive(len(self._items), self)
        if self._fatal is not None:
            raise self._fatal
        missing = [self._items[i].qid for i, row in enumerate(self._rows) if row is None]
        if missing:
            raise RuntimeError(
                f"load model {type(model).__name__} finished without launching "
                f"{len(missing)} queries, e.g. {missing[:3]}"
            )
        return self._rows


async def run_queries_once(
        cfg: ScenarioConfig,
        items,
        backends: BackendRegistry,
        embedder: SentenceTransformerEmbedder,
        caches: CacheManager,
        reranker: Reranker,
        generator: Generator,
        on_done: Callable[[], None] = lambda: None,
) -> pd.DataFrame:
    total_budget = cfg.timeouts.total_ms / 1000.0

    async def run_one(it, planned_at: float) -> dict[str, Any]:
        timed_out = False
        timeout_stage = None
        retrieved_ids = []
        reranked_ids = []
        context_ids = []
        final_items: list[RetrievedItem] = []
        retrieved_doc_ids = []
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
            # бюджет идёт с планового момента: ожидание старта входит в него
            with anyio.fail_after(max(0.0, planned_at + total_budget - anyio.current_time())):
                t1 = time.perf_counter()
                vec = await caches.get_or_compute_embedding(it.query, embedder.embed_query)
                t2 = time.perf_counter()

                # retrieval cache
                key = caches.make_retrieval_key(it.query)
                cached = caches.get_retrieval(key)
                if cached is not None:
                    retrieved_items = cached.retrieved_items
                else:
                    retrieved_items = await retrieve_items(cfg, backends, it.query, vec)
                    caches.set_retrieval(key, retrieved_items)
                retrieved_ids = [item.chunk_id for item in retrieved_items]
                retrieved_doc_ids = project_to_docs(retrieved_items)

                t3 = time.perf_counter()

                if cfg.rerank.enabled:
                    reranked = await rerank_items(cfg, reranker, it.query, retrieved_items)
                    reranked_ids = [result.item.chunk_id for result in reranked]
                    rerank_scores = [result.score for result in reranked]
                    context_candidates = reranked
                    final_items = [result.item for result in reranked]
                else:
                    context_candidates = retrieved_items
                    final_items = retrieved_items

                t4 = time.perf_counter()

                context = assemble_context(cfg, context_candidates)
                if cfg.context.enabled:
                    context_ids = context.ids
                    context_text = context.text if cfg.output.include_debug_fields else ""
                    context_tokens = context.token_count
                    context_chunks = len(context.chunks)
                    final_items = [chunk.item for chunk in context.chunks]
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
            lat_total_ms = (anyio.current_time() - planned_at) * 1000.0

        quality = evaluate_quality(cfg.metrics.quality, final_items, it.qrels)

        return {
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
            "final_ids": [item.chunk_id for item in final_items],
            "final_doc_ids": project_to_docs(final_items),
            "rerank_scores": rerank_scores,
            **quality,
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

    model = build_load_model(cfg.run.load.type, cfg.run.load.params, cfg.seed)
    return pd.DataFrame(await _Launcher(list(items), run_one, on_done).run(model))
