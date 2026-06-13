from __future__ import annotations


def build_prompt(query: str, context_text: str) -> str:
    return (
        "Answer the question using only the provided context. "
        "If the context is insufficient, say that the context is insufficient.\n\n"
        f"Question:\n{query}\n\n"
        f"Context:\n{context_text}\n\n"
        "Answer:"
    )
