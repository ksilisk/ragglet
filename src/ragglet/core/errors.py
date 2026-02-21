class StageTimeout(Exception):
    def __init__(self, stage: str):
        super().__init__(f"Timeout at stage: {stage}")
        self.stage = stage