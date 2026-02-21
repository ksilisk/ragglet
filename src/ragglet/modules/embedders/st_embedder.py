from __future__ import annotations
from sentence_transformers import SentenceTransformer
import numpy as np

class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, normalize: bool = True):
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize

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