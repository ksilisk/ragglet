from __future__ import annotations

from pathlib import Path


def resolve_dataset_dir(source: str, dataset_id: str) -> Path:
    """
    Приводит (source, id) к локальной директории с BEIR-триадой.

    - local: `id` — путь к директории датасета;
    - hf:    `id` — repo id на HuggingFace Hub; скачиваем snapshot в кэш.
    """
    if source == "local":
        path = Path(dataset_id)
        if not path.exists():
            raise FileNotFoundError(f"Local dataset directory not found: {path}")
        return path
    if source == "hf":
        return _resolve_hf(dataset_id)
    raise ValueError(f"Unknown dataset source: {source!r} (expected 'local' or 'hf')")


def _resolve_hf(repo_id: str) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "source='hf' requires 'huggingface_hub' (project dependency)."
        ) from e
    local = snapshot_download(repo_id=repo_id, repo_type="dataset")
    return Path(local)
