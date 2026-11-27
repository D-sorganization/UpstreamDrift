"""Soft stance anchor (#15296): a planted sphere may slide a set allowance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.contact_law import GroundPlane
from src.shared.python.motion_matching.stance_anchor import (
    SoftAnchor,
    soft_anchor_rows,
)

pytestmark = [pytest.mark.unit]
ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"
GROUND = GroundPlane(normal=(0.0, 0.0, 1.0), height_m=0.0)
UP = (0.0, 0.0, 1.0)


def test_rows_are_free_inside_the_allowance_and_stiff_beyond_it() -> None:
    rule = SoftAnchor(normal=UP, normal_weight=100.0, slide_weight=4.0, slide_m=0.005)
    jac = np.eye(3)
    inside, slope = soft_anchor_rows(np.array([0.003, 0.0, 0.002]), jac, rule)
    np.testing.assert_allclose(inside, [10.0 * 0.002, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(slope[1:], 0.0)
    np.testing.assert_allclose(slope[0], [0.0, 0.0, 10.0])
    beyond, _ = soft_anchor_rows(np.array([0.0, 0.009, 0.0]), jac, rule)
    np.testing.assert_allclose(beyond, [0.0, 0.0, 2.0 * 0.004, 0.0], atol=1e-15)
    at_rest, flat = soft_anchor_rows(np.zeros(3), jac, rule)
    assert not at_rest.any() and not flat[1:].any()


def test_the_slope_is_the_derivative_of_the_rows() -> None:
    rule = SoftAnchor(normal=UP, normal_weight=100.0, slide_weight=9.0, slide_m=0.004)
    rng = np.random.default_rng(3)
    jac = rng.normal(size=(3, 5))
    q = rng.normal(size=5)
    gap0 = np.array([0.011, -0.006, 0.002]) - jac @ q

    def rows(x: np.ndarray) -> np.ndarray:
        return soft_anchor_rows(gap0 + jac @ x, jac, rule)[0]

    step = 1e-7
    numeric = np.column_stack(
        [(rows(q + step * e) - rows(q - step * e)) / (2 * step) for e in np.eye(5)]
    )
    np.testing.assert_allclose(soft_anchor_rows(gap0 + jac @ q, jac, rule)[1], numeric)


def test_settings_are_checked() -> None:
    for bad in (
        {"slide_m": -0.001},
        {"slide_weight": -1.0},
        {"normal_weight": float("nan")},
        {"normal": (0.0, 0.0, 0.0)},
    ):
        values = {
            "normal": UP,
            "normal_weight": 1.0,
            "slide_weight": 1.0,
            "slide_m": 0.0,
        }
        with pytest.raises(ValueError, match="SoftAnchor"):
            SoftAnchor(**(values | bad))
    with pytest.raises(ValueError, match="gap"):
        soft_anchor_rows(np.zeros(2), np.eye(3), SoftAnchor(UP, 1.0, 1.0, 0.0))


def _kinematics():  # type: ignore[no-untyped-def]
    pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python import full_body_markers as module
    from src.engines.physics_engines.mujoco.python.full_body_model import (
        NativeMujocoFullBodyModel,
    )

    spec = json.loads(SPEC.read_bytes())
    attachments = {
        label: (a["body"], tuple(a["offset_m"]))
        for label, a in spec["marker_attachments"].items()
        if a["offset_m"] is not None
    }
    attachments["LToeOut"] = ("calcn_l", (0.16, 0.0, -0.04))
    attachments["LAnkleOut"] = ("talus_l", (0.0, 0.0, 0.05))
    return module.FullBodyMarkerKinematics(
        NativeMujocoFullBodyModel(SPEC.read_bytes()), attachments
    )


@pytest.mark.requires_mujoco
def test_a_planted_heel_slides_no_farther_than_the_allowance() -> None:
    """The markers ask for a 5 cm shift: the legacy anchor holds the heel, the
    soft anchor lets it go the allowance and no farther, a zero weight frees
    it, and the heel stays on the ground throughout."""
    kin = _kinematics()
    attachments = kin.labels
    q_up = np.zeros(len(kin.coordinate_order))
    q_up[2] = 1.0
    common = {
        "ground": GROUND,
        "closure_weight": 0.0,
        "prior_weight": 1e-6,
        "flat_feet_per_frame": [("heel_l", "forefoot_l")] * 2,
        "plant_stance": True,
        "anchor_stance_at_init": True,
        "ground_weight": 1e4,
    }
    seen = np.ones((2, len(attachments)), dtype=bool)
    q_a = kin.solve_trajectory(
        np.stack([kin.marker_positions(q_up)] * 2), seen, q_up, **common
    )[0][0]
    q_b = q_a.copy()
    q_b[0] += 0.05
    targets = np.stack([kin.marker_positions(q_b)] * 2)
    start = kin.sphere_ground_points(q_a, GROUND)["heel_l"]

    def slide(**soft: float) -> float:
        q, fits = kin.solve_trajectory(targets, seen, q_a, **common, **soft)
        heel = kin.sphere_ground_points(q[1], GROUND)["heel_l"]
        assert abs(fits[1].lowest_sphere_height_m) < 2e-3
        return float(np.linalg.norm(heel - start))

    assert slide() < 2e-3
    assert 0.007 < slide(anchor_slide_m=0.008) < 0.0095
    assert slide(anchor_weight=0.0) > 0.03
    assert 2e-3 < slide(anchor_weight=1.0) < 0.045
    for bad in ({"anchor_slide_m": -1.0}, {"anchor_weight": -1.0}):
        with pytest.raises(ValueError, match="anchor"):
            kin.solve_trajectory(targets, seen, q_a, **common, **bad)


@pytest.mark.requires_mujoco
def test_a_posture_prior_can_be_given_for_single_frames() -> None:
    """``posture_prior_per_frame`` (#15296) adds its targets on the frames
    that name some, on top of the solve's own posture prior."""
    kin = _kinematics()
    order = list(kin.coordinate_order)
    q0 = np.zeros(len(order))
    q0[2] = 1.0
    targets = np.stack([kin.marker_positions(q0)] * 2)
    seen = np.ones(targets.shape[:2], dtype=bool)
    name = "TorsoInput"
    common = {
        "ground": GROUND,
        "closure_weight": 0.0,
        "ground_weight": 0.0,
        "prior_weight": 1e-9,
    }
    plain, _ = kin.solve_trajectory(targets, seen, q0, **common)
    held, _ = kin.solve_trajectory(
        targets,
        seen,
        q0,
        posture_prior_per_frame=[{name: (0.3, 1e6)}, None],
        **common,
    )
    column = order.index(name)
    assert abs(plain[0, column]) < 0.02
    assert held[0, column] == pytest.approx(0.3, abs=0.02)
    assert abs(held[1, column] - 0.3) > 0.1
    with pytest.raises(ValueError, match="posture_prior_per_frame"):
        kin.solve_trajectory(
            targets, seen, q0, posture_prior_per_frame=[None], **common
        )
