from __future__ import annotations

from ragglet.context.token_count import estimate_tokens
from ragglet.generation.base import GenerationRequest, GenerationResult


class NoopGenerator:
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        answer = ""
        input_tokens = estimate_tokens(request.prompt)
        output_tokens = estimate_tokens(answer)
        return GenerationResult(
            answer=answer,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_estimated=None,
            provider="noop",
            model=request.model,
            raw_response=None,
        )
