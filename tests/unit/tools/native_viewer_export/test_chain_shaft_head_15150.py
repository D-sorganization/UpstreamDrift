"""A chain shaft keeps the club parts, grip and face on the right bodies (#15150, C4).

With ``club.shaft.model = "chain"`` the head rides the tip link and bends
with the shaft, while the grip stays on the club body. The plain MeshCat
scene must draw each part on its own body, and the hands close-up must read
the grip of the club body, not of the tip link.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("mujoco")

from src.shared.python.motion_matching.club_models import DRIVER  # noqa: E402
from src.shared.python.motion_matching.club_shaft_chain import (  # noqa: E402
    apply_shaft_chain,
    build_shaft_chain,
    passive_coordinates,
)
from src.tools.native_viewer_export.backends._club import (  # noqa: E402
    club_parts_by_body,
)
from src.tools.native_viewer_export.club_pose import ClubPoseModel  # noqa: E402

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[4]
SPEC_PATH = (
    ROOT
    / "docs/development/full_body_models/evidence/ground_support"
    / "anthro_driver_seeds/full_body_spec_hipcal_scaled.json"
)
#: bend (rad) about the first shaft axis of every link; enough to move the head
#: by decimetres so the checks below cannot pass by rounding
BEND_RAD = 0.1


@pytest.fixture(scope="module")
def chain_spec() -> dict[str, Any]:
    rigid = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    return apply_shaft_chain(rigid, build_shaft_chain(DRIVER))


def _coords(spec: dict[str, Any], bend: float) -> dict[str, float]:
    q = dict.fromkeys(spec["coordinate_order"], 0.0)
    for name in passive_coordinates(spec):
        if "bend_x" in name:
            q[name] = bend
    return q


def test_plain_scene_draws_the_head_on_the_tip_link(chain_spec: dict[str, Any]) -> None:
    by_body = club_parts_by_body(chain_spec)
    shaft = chain_spec["club"]["shaft"]
    on_tip = {p.name for p in by_body[shaft["tip_link"]]}
    on_club = {p.name for p in by_body[shaft["club_body"]]}
    assert "club_head" in on_tip
    assert "club_head" not in on_club
    assert "club_grip" in on_club
    links = [b for b in by_body if b != shaft["club_body"]]
    assert len(links) == chain_spec["club"]["shaft"]["n_links"]
    for body in links:
        assert any(p.name.startswith("club_shaft_link") for p in by_body[body])


def test_rigid_spec_keeps_every_part_on_the_club_body() -> None:
    rigid = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    by_body = club_parts_by_body(rigid)
    assert len(by_body) == 1


def test_grip_midpoint_does_not_move_with_the_shaft_bend(
    chain_spec: dict[str, Any],
) -> None:
    model = ClubPoseModel(json.dumps(chain_spec).encode())
    straight = model.grip_midpoint_world(_coords(chain_spec, 0.0))
    bent = model.grip_midpoint_world(_coords(chain_spec, BEND_RAD))
    np.testing.assert_allclose(bent, straight, atol=1e-9)


def test_face_centre_follows_the_bent_head(chain_spec: dict[str, Any]) -> None:
    model = ClubPoseModel(json.dumps(chain_spec).encode())
    straight = model.face_world(_coords(chain_spec, 0.0)).centre_m
    bent = model.face_world(_coords(chain_spec, BEND_RAD)).centre_m
    assert np.linalg.norm(bent - straight) > 0.05
