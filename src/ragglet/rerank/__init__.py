from ragglet.rerank.base import RerankedItem, Reranker

__all__ = ["RerankedItem", "Reranker", "build_reranker", "rerank_items"]


def __getattr__(name: str):
    if name in {"build_reranker", "rerank_items"}:
        from ragglet.rerank import engine

        return getattr(engine, name)
    raise AttributeError(name)
