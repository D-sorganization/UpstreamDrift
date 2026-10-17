"""The lifter body of the same-input lift bundles (OSL-14, #13057).

The bundles used to replay the golf hip-calibrated spec, whose trunk-thigh
fold landed on the hip (hip flexion 159-172 deg against the 120 deg limit).
These tests pin the replacement: the AN-1 de Leva body with a lumbar joint,
cited ranges as IK bounds, de Leva hip spacing, a symmetric standing pose and
committed bundles whose every coordinate stays inside its cited range.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.shared.python.lifting.bundle_targets import bundle_pelvis_rotation
from src.shared.python.lifting.bundles import (
    BACK_SQUAT_BARBELL_BUNDLE,
    BACK_SQUAT_BUNDLE,
    LIFT_SPEC,
    LiftIKOptions,
    load_seed_pose,
)
from src.shared.python.lifting.lift_body import (
    LIFT_BODY_MASS_KG,
    LIFT_BODY_SPEC,
    LIFT_BODY_STATURE_M,
    lift_bounds,
)
from src.shared.python.motion_matching.anthropometric_geometry import (
    HIP_HALF_WIDTH_FRACTION_OF_STATURE,
)
from src.shared.python.motion_matching.range_of_motion import (
    HUMAN_RANGES_DEG,
    violations,
)
from src.shared.python.motion_matching.same_input import InputBundle

pytestmark = pytest.mark.unit

_EVIDENCE = BACK_SQUAT_BUNDLE.parents[1]
COMMITTED_BUNDLES = (
    BACK_SQUAT_BUNDLE,
    BACK_SQUAT_BARBELL_BUNDLE,
    _EVIDENCE / "deadlift" / "deadlift_bundle.npz",
    _EVIDENCE / "clean_and_jerk" / "clean_and_jerk_bundle.npz",
)


@pytest.fixture(scope="module")
def spec() -> dict:
    return json.loads(LIFT_SPEC.read_bytes())


def _hip(spec: dict, side: str) -> dict:
    return next(j for j in spec["joints"] if j["name"] == f"hip_{side}")


def test_bundles_replay_the_lift_body(spec: dict) -> None:
    assert LIFT_SPEC == LIFT_BODY_SPEC
    assert spec["subject"]["stature_m"] == LIFT_BODY_STATURE_M
    assert spec["subject"]["mass_kg"] == LIFT_BODY_MASS_KG


def test_lift_body_has_a_lumbar_joint(spec: dict) -> None:
    for name in ("SpineInputX", "SpineInputY"):
        assert name in spec["coordinate_order"]
        assert tuple(spec["coordinate_ranges_deg"][name]) == HUMAN_RANGES_DEG[name]


def test_hip_spacing_is_de_leva_scaled(spec: dict) -> None:
    left = np.asarray(_hip(spec, "l")["parent_to_base"], dtype=float)[:3, 3]
    right = np.asarray(_hip(spec, "r")["parent_to_base"], dtype=float)[:3, 3]
    expected = 2.0 * HIP_HALF_WIDTH_FRACTION_OF_STATURE * LIFT_BODY_STATURE_M
    assert np.linalg.norm(left - right) == pytest.approx(expected, abs=1e-6)


def test_cited_ranges_become_radian_bounds(spec: dict) -> None:
    bounds = lift_bounds(spec)
    assert bounds["hip_flexion_r"] == pytest.approx(np.radians((-30.0, 120.0)))
    assert set(bounds) == set(spec["coordinate_ranges_deg"])
    assert lift_bounds({}) == {}
    with pytest.raises(ValueError, match="empty"):
        lift_bounds({"coordinate_ranges_deg": {"x": (1.0, 1.0)}})


def test_lift_ik_uses_the_bounded_solver() -> None:
    assert LiftIKOptions().solver == "trf"
    with pytest.raises(ValueError, match="solver"):
        LiftIKOptions(solver="newton")


def test_seed_pose_is_in_spec_order(spec: dict) -> None:
    seed = load_seed_pose()
    assert seed.shape == (len(spec["coordinate_order"]),)
    assert np.isfinite(seed).all()


def test_pelvis_rotation_accepts_the_mirrored_left_hip(spec: dict) -> None:
    rotation = bundle_pelvis_rotation(spec)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    broken = json.loads(json.dumps(spec))
    hip = _hip(broken, "l")
    base = np.asarray(hip["parent_to_base"], dtype=float)
    base[:3, :3] = base[:3, :3] @ np.diag([-1.0, -1.0, 1.0])
    hip["parent_to_base"] = base.tolist()
    with pytest.raises(ValueError, match="pelvis orientation"):
        bundle_pelvis_rotation(broken)


@pytest.mark.parametrize("path", COMMITTED_BUNDLES, ids=lambda p: p.stem)
def test_committed_bundle_stays_inside_cited_ranges(path) -> None:
    bundle = InputBundle.load(path)
    spec = json.loads(bundle.spec_bytes)
    ranges = {k: tuple(v) for k, v in spec["coordinate_ranges_deg"].items()}
    order = spec["coordinate_order"]
    q = np.asarray(bundle.reference_q, dtype=float)[:, : len(order)]
    # The shared flag tolerance (0.5 deg): the IK solution is inside the bounds
    # exactly; the MuJoCo-tracked reference overshoots a bound by <= 0.02 deg.
    assert violations(q, order, ranges) == {}
    for name in ("subtalar_angle_l", "subtalar_angle_r"):
        assert np.abs(q[:, order.index(name)]).max() < np.radians(0.5)


@pytest.mark.parametrize("path", COMMITTED_BUNDLES[:1], ids=lambda p: p.stem)
def test_committed_squat_stands_symmetric(path) -> None:
    """Scapulae and trunk twist are held neutral: shoulders level at the start."""
    bundle = InputBundle.load(path)
    standing = bundle.provenance["ik"]["q_standing"]
    spec = json.loads(bundle.spec_bytes)
    order = spec["coordinate_order"]
    for name in ("LScapInputX", "RScapInputX", "LScapInputY", "RScapInputY"):
        assert standing[order.index(name)] == pytest.approx(0.0, abs=1e-12)
    for side in "lr":
        assert np.degrees(standing[order.index(f"hip_flexion_{side}")]) < 5.0


@pytest.mark.integration
def test_committed_lift_body_regenerates() -> None:
    from src.shared.python.lifting import lift_body

    assert lift_body.main([]) == 0
