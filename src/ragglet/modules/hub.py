from __future__ import annotations

from pathlib import Path


def cached_model_revision(repo_id: str, revision: str | None) -> str | None:
    """Коммит snapshot модели в кэше HF (после загрузки); None для локального пути."""
    if Path(repo_id).exists():
        return None
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(repo_id, "config.json", revision=revision)
    if not isinstance(path, str):
        raise RuntimeError(
            f"Cannot resolve revision of model {repo_id!r} ({revision or 'main'}) "
            "in the HuggingFace cache; pin a full commit sha"
        )
    return Path(path).parent.name
