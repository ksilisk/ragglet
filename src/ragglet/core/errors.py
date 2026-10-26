class StageTimeout(Exception):
    def __init__(self, stage: str):
        super().__init__(f"Timeout at stage: {stage}")
        self.stage = stage


class ServiceUnavailable(Exception):
    """Сервис (Qdrant, Redis) не ответил: соединение, таймаут. Ответ с ошибкой — не это."""

    def __init__(self, endpoint: str, reason: str):
        super().__init__(f"{endpoint}: {reason}")
        self.endpoint = endpoint
        self.reason = reason
