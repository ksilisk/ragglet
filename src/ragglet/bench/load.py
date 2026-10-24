from __future__ import annotations

from typing import Any, ClassVar, Protocol

import anyio
from pydantic import BaseModel, ConfigDict, Field


class Launch(Protocol):
    async def __call__(self, i: int, planned_at: float | None = None) -> None:
        """
        Выполнить запрос номер i. planned_at — когда он должен был стартовать по правилу
        модели (часы anyio.current_time()); с него идут latency и бюджет запроса.
        None — фактический старт.
        """


class LoadModel(Protocol):
    """
    Модель нагрузки решает только, когда и в каком порядке звать launch. Строки, ошибки и
    отмену ведёт ядро. Модель не должна глотать отмену, а случайность берёт из seed.
    """

    Params: ClassVar[type[BaseModel]]

    def __init__(self, params: Any, seed: int) -> None: ...

    async def drive(self, n: int, launch: Launch) -> None: ...


class ClosedLoad:
    """N пользователей: каждый шлёт следующий запрос, только получив ответ на предыдущий."""

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")

        concurrency: int = Field(1, ge=1)

    def __init__(self, params: Params, seed: int):
        self.params = params

    async def drive(self, n: int, launch: Launch) -> None:
        order = iter(range(n))

        async def worker() -> None:
            for i in order:  # общий итератор: next() синхронный, гонки нет
                await launch(i)

        async with anyio.create_task_group() as tg:
            for _ in range(min(self.params.concurrency, n)):
                tg.start_soon(worker)


LOAD_MODELS: dict[str, type[LoadModel]] = {"closed": ClosedLoad}


def load_params(load_type: str, raw: dict[str, Any]) -> BaseModel:
    if load_type not in LOAD_MODELS:
        raise ValueError(f"unknown load model {load_type!r}; available: {sorted(LOAD_MODELS)}")
    return LOAD_MODELS[load_type].Params.model_validate(raw)


def build_load_model(load_type: str, raw: dict[str, Any], seed: int) -> LoadModel:
    return LOAD_MODELS[load_type](load_params(load_type, raw), seed)
