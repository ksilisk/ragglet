from __future__ import annotations

from pathlib import Path

import pytest

from ragglet.config.index_profile import load_index_profile
from ragglet.config.scenario import load_scenario

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize(
    "path",
    sorted([*CONFIGS.glob("scenarios/*.yaml"), *CONFIGS.glob("experiments/**/*.yaml")]),
    ids=lambda p: str(p.relative_to(CONFIGS)),
)
def test_committed_scenarios_match_schema(path):
    load_scenario(path)


@pytest.mark.parametrize(
    "path", sorted(CONFIGS.glob("indexes/*.yaml")), ids=lambda p: str(p.relative_to(CONFIGS))
)
def test_committed_index_profiles_match_schema(path):
    load_index_profile(path)
