from ragglet.context.base import ContextChunk, ContextResult
from ragglet.context.token_count import estimate_tokens

__all__ = ["ContextChunk", "ContextResult", "assemble_context", "estimate_tokens"]


def __getattr__(name: str):
    if name == "assemble_context":
        from ragglet.context.engine import assemble_context

        return assemble_context
    raise AttributeError(name)
