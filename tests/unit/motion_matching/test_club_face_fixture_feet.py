"""#12222: the club-face fixtures stand on the capture's address feet.

MuJoCo FK of each fixture's address pose, on the run's own scaled document
(``spec_<club>.json``, committed with the fixture because the calibrated hip
frames set the foot yaw), must put each foot's toe-out within 2 degrees of
the capture's address toe-out (OSV-4 targets, recorded in the provenance from
the run receipt) and keep the feet on the ground.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.address_foot_progression_engines import (
    TOLERANCE_DEG,
    plant_points,
    score_engine,
)
from src.shared.python.motion_matching.pipeline.address_feet import (
    split_address_coordinates,
)

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).resolve().parents[3] / "tests/fixtures/club_face"
CLUBS = ("driver", "iron7")
#: Acceptance: calcn and toes origins within 5 cm of the floor at address.
FOOT_HEIGHT_LIMIT_M = 0.05


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def provenance() -> dict:
    return _load("provenance.json")["clubs"]


@pytest.fixture(scope="module")
def poses() -> dict:
    return _load("address_poses.json")["poses"]


def _address_points(club: str, poses: dict) -> tuple[dict, float]:
    pytest.importorskip("mujoco")
    from src.shared.python.motion_matching.pipeline.plant import get_plant

    spec = _load(f"spec_{club}.json")
    pose = poses[club]
    angles, translations = split_address_coordinates(
        pose["coordinate_order"], np.asarray(pose["q_rad"], dtype=float)
    )
    points, _ = plant_points("mujoco", spec, angles, translations)
    return points, float(get_plant("mujoco", spec).ground_plane.height_m)


@pytest.mark.parametrize("club", CLUBS)
def test_fixture_spec_is_the_recorded_run_document(club: str, provenance) -> None:
    data = (FIXTURES / f"spec_{club}.json").read_bytes()
    assert hashlib.sha256(data).hexdigest() == provenance[club]["spec_sha256"]


@pytest.mark.parametrize("club", CLUBS)
def test_address_pose_is_fixture_frame_zero(club: str, poses) -> None:
    q = np.load(FIXTURES / f"swing_q_{club}.npz")["q"].astype(float)
    np.testing.assert_allclose(q[0], poses[club]["q_rad"], atol=1e-6)


@pytest.mark.parametrize("club", CLUBS)
def test_fixture_toe_out_matches_the_capture_at_address(
    club: str, provenance, poses
) -> None:
    feet = provenance[club]["foot_progression"]
    assert feet["source"] == "capture"
    points, _ = _address_points(club, poses)
    score = score_engine(points, feet["target_deg"])
    assert score["within_tolerance"], (
        f"{club}: toe-out {score['model_deg']} against the capture "
        f"{feet['target_deg']} (limit {TOLERANCE_DEG} deg)"
    )


@pytest.mark.parametrize("club", CLUBS)
def test_fixture_feet_stand_on_the_ground_at_address(club: str, poses) -> None:
    points, ground_m = _address_points(club, poses)
    heights = {
        name: float(p[2]) - ground_m
        for name, p in points.items()
        if name.startswith(("calcn", "toes"))
    }
    assert len(heights) == 4
    assert all(0.0 <= h <= FOOT_HEIGHT_LIMIT_M for h in heights.values()), heights
