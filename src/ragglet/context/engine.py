from __future__ import annotations

from collections.abc import Sequence

from ragglet.config.scenario import ScenarioConfig
from ragglet.context.base import ContextChunk, ContextResult
from ragglet.context.token_count import estimate_tokens
from ragglet.rerank.base import RerankedItem
from ragglet.retrieval.merge import RetrievedItem


def assemble_context(
    cfg: ScenarioConfig,
    items: Sequence[RetrievedItem] | Sequence[RerankedItem],
) -> ContextResult:
    candidates = _to_context_candidates(items)
    if not cfg.context.enabled:
        return ContextResult(chunks=[], text="", token_count=0)

    if cfg.context.policy == "top_n":
        selected = _select_by_budget(candidates, max_tokens=cfg.context.max_tokens)
    elif cfg.context.policy == "threshold":
        selected = _select_by_budget(
            [chunk for chunk in candidates if chunk.score >= cfg.context.threshold],
            max_tokens=cfg.context.max_tokens,
        )
    elif cfg.context.policy == "adaptive":
        selected = _select_adaptive(
            candidates,
            max_tokens=cfg.context.max_tokens,
            threshold=cfg.context.threshold,
        )
    else:
        raise NotImplementedError(f"context.policy='{cfg.context.policy}' is not implemented")

    if not selected and candidates:
        selected = _select_by_budget(candidates[:1], max_tokens=cfg.context.max_tokens)

    return ContextResult(
        chunks=selected,
        text=_render_context(selected, include_metadata=cfg.context.include_metadata),
        token_count=sum(chunk.token_count for chunk in selected),
    )


def _to_context_candidates(
    items: Sequence[RetrievedItem] | Sequence[RerankedItem],
) -> list[ContextChunk]:
    out: list[ContextChunk] = []
    for rank, entry in enumerate(items, start=1):
        if isinstance(entry, RerankedItem):
            item = entry.item
            score = entry.score
        else:
            item = entry
            score = float(item.score_raw)

        text = _text_for_item(item)
        out.append(
            ContextChunk(
                item=item,
                text=text,
                token_count=estimate_tokens(text),
                rank=rank,
                score=float(score),
            )
        )
    return out


def _select_by_budget(candidates: list[ContextChunk], max_tokens: int) -> list[ContextChunk]:
    selected: list[ContextChunk] = []
    used = 0
    for chunk in candidates:
        if chunk.token_count <= 0:
            continue
        if used and used + chunk.token_count > max_tokens:
            continue
        if not used and chunk.token_count > max_tokens:
            selected.append(chunk)
            break
        selected.append(chunk)
        used += chunk.token_count
    return selected


def _select_adaptive(
    candidates: list[ContextChunk],
    max_tokens: int,
    threshold: float,
) -> list[ContextChunk]:
    if not candidates:
        return []

    max_score = max((chunk.score for chunk in candidates), default=0.0)
    min_score = min((chunk.score for chunk in candidates), default=0.0)
    denom = max(max_score - min_score, 1e-9)

    filtered: list[ContextChunk] = []
    for chunk in candidates:
        normalized = (chunk.score - min_score) / denom
        if normalized >= threshold:
            filtered.append(chunk)

    return _select_by_budget(filtered, max_tokens=max_tokens)


def _render_context(chunks: list[ContextChunk], include_metadata: bool) -> str:
    parts: list[str] = []
    for chunk in chunks:
        if include_metadata:
            parts.append(f"[{chunk.item.chunk_id}]\n{chunk.text}")
        else:
            parts.append(chunk.text)
    return "\n\n".join(parts)


def _text_for_item(item: RetrievedItem) -> str:
    payload = item.payload or {}
    text = payload.get("text", "")
    return str(text) if text is not None else ""
