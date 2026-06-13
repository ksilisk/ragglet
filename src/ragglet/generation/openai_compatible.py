from __future__ import annotations

import os
from typing import Any

import httpx

from ragglet.context.token_count import estimate_tokens
from ragglet.generation.base import GenerationRequest, GenerationResult


class OpenAICompatibleGenerator:
    def __init__(self, base_url: str, api_key_env: str):
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Environment variable {self.api_key_env} is required for generation")

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
        }
        headers = {"Authorization": f"Bearer {api_key}"}

        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        answer = _extract_answer(data)
        usage = data.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens") or estimate_tokens(request.prompt))
        output_tokens = int(usage.get("completion_tokens") or estimate_tokens(answer))
        total_tokens = int(usage.get("total_tokens") or input_tokens + output_tokens)

        return GenerationResult(
            answer=answer,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_estimated=None,
            provider="openai_compatible",
            model=request.model,
            raw_response=data,
        )


def _extract_answer(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    return str(content) if content is not None else ""
