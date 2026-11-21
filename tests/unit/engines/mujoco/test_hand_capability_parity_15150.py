"""MuJoCo hand mobility and capacity agree with the other engines by construction (#15150).

Two properties the cross-engine parity test needs and that hold without the
other engines installed: the grip closure is a rigid weld wherever the hands
are, and torque limits are paired with the degrees of freedom they belong to.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from src.engines.physics_engines.mujoco.python.hand_mobility import (  # noqa: E402
    HandMobilitySource,
)
from src.shared.python.loop_nullspace.capacity import CapacitySubject  # noqa: E402
from src.shared.python.spatial_algebra.operational_space import (  # noqa: E402
    acceleration_capacity,
    force_capacity,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[4]
ANTHRO_SPEC = (
    ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
)
#: closure-site separation above which a pose counts as open
MIN_OPEN_GAP_M = 0.01
RANK_TOL = 1e-10
TWIST_TOL = 1e-9
RADII_RTOL = 1e-9


@pytest.fixture(scope="module")
def source() -> HandMobilitySource:
    subject = CapacitySubject(mass_kg=75.0, stature_m=1.75)
    return HandMobilitySource(ANTHRO_SPEC.read_bytes(), torque_limits=subject)


@pytest.fixture(scope="module")
def q(source) -> np.ndarray:
    return np.zeros(len(source.coordinate_order))


def test_grip_closure_is_a_rigid_weld_where_the_hands_are_apart(source, q) -> None:
    """Motions the closure allows give both welded bodies one twist at any point."""
    mj, model, data = source._mj, source.model, source.data
    coordinates = source._set_state(q)
    first, second = source._closure
    gap = np.linalg.norm(data.site_xpos[first] - data.site_xpos[second])
    assert gap > MIN_OPEN_GAP_M, "the pose must leave the closure open"
    point = source.grip_point_world("lead", coordinates)
    closure = source._task_matrices(point, True)[2]
    _, singular, rows = np.linalg.svd(closure)
    allowed = rows[int((singular > RANK_TOL * singular[0]).sum()) :].T
    assert allowed.shape[1] == model.nv - 6
    twists = []
    for site in (first, second):
        jac = np.zeros((6, model.nv))
        body = int(model.site_bodyid[site])
        mj.mj_jac(model, data, jac[:3], jac[3:], np.zeros(3), body)
        twists.append(jac @ allowed)
    assert np.abs(twists[0] - twists[1]).max() < TWIST_TOL


@pytest.mark.parametrize("hand", ["lead", "trail"])
def test_torque_limits_are_paired_with_their_own_degrees_of_freedom(
    source, q, hand
) -> None:
    """Capacity equals the one computed with everything in ``coordinate_order``."""
    model, order = source.model, list(source.coordinate_order)
    joints = [model.joint(int(j)).name for j in model.dof_jntid]
    assert joints != order, "the model must reorder coordinates for this to bite"
    to_order = [joints.index(name) for name in order]
    limits = source._torque_limits
    point = source.grip_point_world(hand, source._set_state(q))
    jac, mass, _ = source._task_matrices(point, False)
    args = (jac[:, to_order], mass[np.ix_(to_order, to_order)], limits.B)
    bounds = (limits.tau_lo, limits.tau_hi, None)
    expected = {
        "acceleration": acceleration_capacity(*args, *bounds),
        "force": force_capacity(*args, *bounds),
    }
    direct = {
        "acceleration": source.acceleration_capacity(q, hand, mode="unconstrained"),
        "force": source.force_capacity(q, hand, mode="unconstrained"),
    }
    both = source.hand_mobility(q, mode="unconstrained")[hand]
    bundled = {"acceleration": both.acceleration_capacity, "force": both.force_capacity}
    for kind, want in expected.items():
        for got in (direct[kind], bundled[kind]):
            assert np.allclose(got.radii, want.radii, rtol=RADII_RTOL), kind
            assert np.allclose(got.centre, want.centre, atol=1e-9), kind
