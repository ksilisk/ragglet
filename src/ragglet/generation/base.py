from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class GenerationRequest:
    query: str
    context_text: str
    context_ids: list[str]
    model: str
    temperature: float
    max_output_tokens: int
    prompt: str


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_estimated: float | None
    provider: str
    model: str
    raw_response: dict[str, Any] | None = None


class Generator(Protocol):
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        ...
