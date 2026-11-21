"""Mobility points sit on their specification frames; the clubhead is a point (#15150)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from src.engines.physics_engines.mujoco.python.hand_mobility import (  # noqa: E402
    HandMobilitySource,
)
from src.shared.python.motion_matching.full_body_spec import (  # noqa: E402
    load_full_body_spec,
)
from src.shared.python.spatial_algebra.segment_intersection import (  # noqa: E402
    IntersectionPoint,
    resolve_intersection_points,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[4]
UPPER = (
    ROOT
    / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
)
FULL = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"
#: specification frame that each declarative point is read from
FRAME_OF = {
    "lead_wrist": "LW",
    "trail_wrist": "RW",
    "lead_elbow": "LE",
    "trail_elbow": "RE",
    "lead_shoulder": "LS",
    "trail_shoulder": "RS",
    "pelvis": "Hip",
    "clubhead": "Clubhead",
}
POSITION_TOL_M = 1e-9


@pytest.fixture(scope="module")
def spec() -> dict:
    return load_full_body_spec(FULL, json.loads(UPPER.read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def source(spec) -> HandMobilitySource:
    return HandMobilitySource(json.dumps(spec).encode("utf-8"))


@pytest.fixture(scope="module")
def states(source) -> list[np.ndarray]:
    rng = np.random.default_rng(15150)
    n = len(source.coordinate_order)
    return [rng.normal(0.0, 0.15, n) for _ in range(3)]


def _frame_site_world(source: HandMobilitySource, spec: dict, frame: str) -> np.ndarray:
    """World position of the model's own site of a specification frame."""
    index = [f["name"] for f in spec["frames"]].index(frame)
    site = source.model.site(f"native_frame_{index}").id
    return np.array(source.data.site_xpos[site], dtype=float)


def test_frame_points_sit_on_their_specification_frames(source, spec, states) -> None:
    points = resolve_intersection_points(
        spec, "wrists,elbows,shoulders,pelvis,clubhead"
    )
    assert [p.name for p in points] == list(FRAME_OF)
    for q in states:
        mobility = source.segment_mobility(points, q, mode="constrained")
        for point in points:
            expected = _frame_site_world(source, spec, FRAME_OF[point.name])
            centre = np.array(mobility[point.name].centre)
            assert np.abs(centre - expected).max() < POSITION_TOL_M, point.name


def test_clubhead_is_a_point_of_its_own_group_and_not_a_default(spec) -> None:
    (clubhead,) = resolve_intersection_points(spec, "clubhead")
    frame = next(f for f in spec["frames"] if f["name"] == "Clubhead")
    assert clubhead.group == "club" and clubhead.body == frame["body"]
    assert [p.name for p in resolve_intersection_points(spec, "club")] == ["clubhead"]
    assert [p.name for p in resolve_intersection_points(spec)] == [
        "lead_hand",
        "trail_hand",
    ]
    assert resolve_intersection_points(spec, "all")[-1].name == "clubhead"


def test_clubhead_mobility_matches_its_own_jacobian(source, spec, states) -> None:
    (clubhead,) = resolve_intersection_points(spec, "clubhead")
    mj, model, data = source._mj, source.model, source.data
    for q in states:
        free = source.point_mobility(clubhead, q, mode="unconstrained")
        held = source.point_mobility(clubhead, q, mode="constrained")
        body = model.body(clubhead.body).id
        jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mj.mj_jac(model, data, jacp, jacr, np.array(free.centre), body)
        mass = np.zeros((model.nv, model.nv))
        mj.mj_fullM(model, mass, data.qM)
        expected = jacp @ np.linalg.solve(mass, jacp.T)
        assert np.allclose(free.mobility, expected, atol=1e-9)
        gap = free.mobility - held.mobility
        assert np.linalg.eigvalsh(0.5 * (gap + gap.T)).min() >= -1e-9
        assert np.isfinite(held.effective_mass_kg).all()
        assert (held.effective_mass_kg > 0).all()


def test_unknown_point_body_is_rejected(source, states) -> None:
    stray = IntersectionPoint(
        "stray", "wrists", "no_such_body", (0.0, 0.0, 0.0), (0.5, 0.5, 0.5, 0.8)
    )
    with pytest.raises(ValueError, match="no_such_body"):
        source.point_mobility(stray, states[0])
