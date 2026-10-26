from __future__ import annotations

import math
import time
from collections.abc import Awaitable, Callable
from typing import Any

import anyio
import anyio.lowlevel
import pandas as pd

from ragglet.bench.evaluation import evaluate_quality, project_to_docs
from ragglet.bench.load import LoadModel, build_load_model
from ragglet.cache.manager import CacheManager, Lookup
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

STAGES = ("embed", "retrieve", "rerank", "context", "generation")


def _lap(lat: dict[str, float], stage: str, start: float, done: bool = True) -> float:
    """Задержка пишется, только если этап отработал: пустое значение не попадает в перцентили."""
    now = time.perf_counter()
    if done:
        lat[stage] = (now - start) * 1000.0
    return now


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
        outcome = "ok"
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
        lat: dict[str, float] = {}
        lat_total_ms = 0.0
        emb_cache: Lookup = None
        ret_cache: Lookup = None

        try:
            # бюджет идёт с планового момента: ожидание старта входит в него
            with anyio.fail_after(max(0.0, planned_at + total_budget - anyio.current_time())):
                t = time.perf_counter()
                retrieved_items, ret_cache = caches.get_retrieval(it.query)
                if retrieved_items is None:
                    vec, emb_cache = await caches.get_embedding(it.query)
                    if vec is None:
                        vec = await embedder.embed_query(it.query)
                        await caches.set_embedding(it.query, vec)
                    t = _lap(lat, "embed", t)
                    retrieved_items = await retrieve_items(cfg, backends, it.query, vec)
                    caches.set_retrieval(it.query, retrieved_items)
                retrieved_ids = [item.chunk_id for item in retrieved_items]
                retrieved_doc_ids = project_to_docs(retrieved_items)
                t = _lap(lat, "retrieve", t)

                if cfg.rerank.enabled:
                    reranked = await rerank_items(cfg, reranker, it.query, retrieved_items)
                    reranked_ids = [result.item.chunk_id for result in reranked]
                    rerank_scores = [result.score for result in reranked]
                    context_candidates = reranked
                    final_items = [result.item for result in reranked]
                else:
                    context_candidates = retrieved_items
                    final_items = retrieved_items
                t = _lap(lat, "rerank", t, cfg.rerank.enabled)

                context = assemble_context(cfg, context_candidates)
                if cfg.context.enabled:
                    context_ids = context.ids
                    context_text = context.text if cfg.output.include_debug_fields else ""
                    context_tokens = context.token_count
                    context_chunks = len(context.chunks)
                    final_items = [chunk.item for chunk in context.chunks]
                else:
                    context = ContextResult(chunks=[], text="", token_count=0)
                t = _lap(lat, "context", t, cfg.context.enabled)

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
                        outcome = "error"
                _lap(lat, "generation", t, cfg.generation.enabled and outcome == "ok")
        except StageTimeout as e:
            outcome = "timeout"
            timeout_stage = e.stage
        except TimeoutError:
            outcome = "timeout"
            timeout_stage = "total"
        finally:
            lat_total_ms = (anyio.current_time() - planned_at) * 1000.0

        if outcome != "ok":
            final_items = []  # ответа нет — промах, на каком бы этапе запрос ни остановился
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
            "t_start_ms": (planned_at - t0) * 1000.0,
            **{f"lat_{stage}_ms": lat.get(stage, math.nan) for stage in STAGES},
            "lat_total_ms": lat_total_ms,
            "emb_cache": emb_cache,
            "ret_cache": ret_cache,
            "outcome": outcome,
            "timeout_stage": timeout_stage,
        }

    model = build_load_model(cfg.run.load.type, cfg.run.load.params, cfg.seed)
    launcher = _Launcher(list(items), run_one, on_done)
    t0 = anyio.current_time()  # начало повтора; конец запроса = t_start_ms + lat_total_ms
    return pd.DataFrame(await launcher.run(model))
