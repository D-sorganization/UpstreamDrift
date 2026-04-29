from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


REPO_ROOT = Path(__file__).resolve().parents[2]
PHYSICS_ENGINE = (
    REPO_ROOT
    / "src"
    / "engines"
    / "physics_engines"
    / "mujoco"
    / "python"
    / "mujoco_humanoid_golf"
    / "physics_engine.py"
)


def test_contact_force_body_lookup_guards_negative_geom_ids() -> None:
    source = PHYSICS_ENGINE.read_text(encoding="utf-8")
    guard = "if contact.geom1 < 0 or contact.geom2 < 0:"
    lookup = "geom1_body = self.model.geom_bodyid[contact.geom1]"

    assert guard in source
    assert source.index(guard) < source.index(lookup)
