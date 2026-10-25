from __future__ import annotations


class FieldError(ValueError):
    """
    Ошибка model-валидатора, относящаяся к полю. `path` — от модели валидатора: у
    RerankConfig это "model", у корня ScenarioConfig — "metrics.quality.k". API склеивает
    его с `loc` pydantic, чтобы показать ошибку у поля.
    """

    def __init__(self, path: str, message: str):
        super().__init__(message)
        self.path = path
