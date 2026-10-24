from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from ragglet.core.lanes import Lane
from ragglet.modules.hub import cached_model_revision


def hub_model_id(model_name: str) -> str:
    """Короткое имя -> репозиторий на Hub, как это делает sentence-transformers."""
    if "/" in model_name or Path(model_name).exists():
        return model_name
    return f"sentence-transformers/{model_name}"


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, normalize: bool = True, revision: str | None = None):
        self.model = SentenceTransformer(model_name, revision=revision)
        self.normalize = normalize
        self.revision = cached_model_revision(hub_model_id(model_name), revision)
        self._lane = Lane()

    def compute_info(self) -> dict[str, Any]:
        import torch

        device = self.model.device
        return {
            "device": str(device),
            "accelerator": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch_threads": torch.get_num_threads(),
        }

    async def embed_query(self, text: str) -> list[float]:
        return (await self._lane.run(self.embed_batch, [text]))[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        v = self.model.encode(
            texts,
            normalize_embeddings=self.normalize,
            batch_size=min(64, max(1, len(texts))),
            show_progress_bar=False,
        )
        v = np.asarray(v, dtype=np.float32)
        return [row.tolist() for row in v]
