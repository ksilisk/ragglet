from ragglet.generation.base import GenerationRequest, GenerationResult, Generator

__all__ = ["GenerationRequest", "GenerationResult", "Generator", "build_generator", "generate_answer"]


def __getattr__(name: str):
    if name in {"build_generator", "generate_answer"}:
        from ragglet.generation import engine

        return getattr(engine, name)
    raise AttributeError(name)
