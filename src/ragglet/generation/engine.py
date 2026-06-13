from __future__ import annotations

import anyio

from ragglet.config.scenario import ScenarioConfig
from ragglet.context.base import ContextResult
from ragglet.core.errors import StageTimeout
from ragglet.generation.base import GenerationRequest, GenerationResult, Generator
from ragglet.generation.noop import NoopGenerator
from ragglet.generation.prompt import build_prompt


def build_generator(cfg: ScenarioConfig) -> Generator:
    if not cfg.generation.enabled or cfg.generation.provider == "noop":
        return NoopGenerator()
    if cfg.generation.provider == "openai_compatible":
        from ragglet.generation.openai_compatible import OpenAICompatibleGenerator

        return OpenAICompatibleGenerator(
            base_url=cfg.generation.base_url,
            api_key_env=cfg.generation.api_key_env,
        )
    raise NotImplementedError(f"generation.provider='{cfg.generation.provider}' is not implemented")


async def generate_answer(
    cfg: ScenarioConfig,
    generator: Generator,
    query: str,
    context: ContextResult,
) -> tuple[GenerationResult, str]:
    if not cfg.generation.enabled:
        empty = GenerationResult(
            answer="",
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            cost_estimated=None,
            provider=cfg.generation.provider,
            model=cfg.generation.model or "",
            raw_response=None,
        )
        return empty, ""

    if not cfg.generation.model:
        raise ValueError("generation.model is required when generation.enabled=true")

    prompt = build_prompt(query=query, context_text=context.text)
    request = GenerationRequest(
        query=query,
        context_text=context.text,
        context_ids=context.ids,
        model=cfg.generation.model,
        temperature=cfg.generation.temperature,
        max_output_tokens=cfg.generation.max_output_tokens,
        prompt=prompt,
    )
    try:
        with anyio.fail_after(cfg.timeouts.llm_ms / 1000.0):
            return await generator.generate(request), prompt
    except TimeoutError:
        raise StageTimeout("generation")
