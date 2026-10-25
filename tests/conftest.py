from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


import pytest  # noqa: E402


@pytest.fixture
def stand(tmp_path, monkeypatch):
    """API-сервер на копии библиотеки, с фейковыми Qdrant и Redis и заглушками jobs/registry."""
    from api_support import make_stand

    return make_stand(tmp_path, monkeypatch)
