from __future__ import annotations

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


def hub_model_id(model_name: str) -> str:
    """Короткое имя -> репозиторий на Hub, как это делает sentence-transformers."""
    return model_name if "/" in model_name else f"sentence-transformers/{model_name}"


def cached_model_revision(model_name: str, revision: str | None) -> str | None:
    """Коммит snapshot модели в кэше HF (после загрузки); None для локального пути."""
    if Path(model_name).exists():
        return None
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(hub_model_id(model_name), "config.json", revision=revision)
    if not isinstance(path, str):
        raise RuntimeError(
            f"Cannot resolve revision of embedding model {model_name!r} ({revision or 'main'}) "
            "in the HuggingFace cache; pin a full commit sha"
        )
    return Path(path).parent.name


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, normalize: bool = True, revision: str | None = None):
        self.model = SentenceTransformer(model_name, revision=revision)
        self.normalize = normalize
        self.revision = cached_model_revision(model_name, revision)

    def embed_one(self, text: str) -> list[float]:
        v = self.model.encode([text], normalize_embeddings=self.normalize)[0]
        return v.astype(np.float32).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        v = self.model.encode(
            texts,
            normalize_embeddings=self.normalize,
            batch_size=min(64, max(1, len(texts))),
            show_progress_bar=False,
        )
        v = np.asarray(v, dtype=np.float32)
        return [row.tolist() for row in v]
