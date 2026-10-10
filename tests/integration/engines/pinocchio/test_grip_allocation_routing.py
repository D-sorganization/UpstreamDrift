"""Pinocchio allocation ``lambda_grip`` routed into the grip analysis (GCV-8).

The shared allocator solves one 6-D closure wrench per frame, in the native
constraint LOCAL frame (linear then angular) with ``J.T @ lambda`` the load on
the human.  The adapter must turn it into the net grip wrench on the club in the
world frame through ``PinocchioForceTorqueSource.grip_from_allocation``, with
``split_method="allocation"`` and the per-hand wrenches unavailable.
"""

import json
from pathlib import Path
import sys
from unittest.mock import Mock

import numpy as np
import pytest

if isinstance(sys.modules.get("pinocchio"), Mock):
    pytest.skip("Native Pinocchio required", allow_module_level=True)
pin = pytest.importorskip("pinocchio")

from src.engines.physics_engines.pinocchio.python.native_model import (  # noqa: E402
    FullBodyPinocchioModel,
)
from src.shared.python.biomechanics.grip_wrench import (  # noqa: E402
    to_overlay_wrenches,
)
from src.shared.python.motion_matching.multi_engine_torque_allocator import (  # noqa: E402
    create_engine_force_adapter,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_pinocchio,
    pytest.mark.unit,
]

SPEC = (
    Path(__file__).resolve().parents[4]
    / "docs/development/full_body_models/full_body_spec_v1.json"
)
LAMBDA = np.array([1.0, -2.0, 30.0, 0.1, 0.2, -0.3])


@pytest.fixture(scope="module")
def setup():
    spec = json.loads(SPEC.read_bytes())
    plant = FullBodyPinocchioModel(spec)
    adapter = create_engine_force_adapter("pinocchio", SPEC)
    q = np.linspace(-0.04, 0.06, plant.model.nq)
    return adapter, plant, spec, q


def _closure_frame_in_world(plant, spec, q):
    """Closure frame ``a`` from the specification, independent of the weld model."""
    joint, body_pose = plant._bodies[spec["closure"]["body_a"]]
    pin.forwardKinematics(plant.model, plant.data, q)
    placement = np.asarray(spec["closure"]["placement_a"], dtype=float)
    pose = (
        plant.data.oMi[joint] * body_pose * pin.SE3(placement[:3, :3], placement[:3, 3])
    )
    return np.asarray(pose.rotation), np.asarray(pose.translation)


def test_allocation_is_emitted_as_net_only_in_world_frame(setup) -> None:
    adapter, plant, spec, q = setup
    analysis = adapter.grip_analysis_from_allocation(q, LAMBDA)
    rotation, point = _closure_frame_in_world(plant, spec, q)
    assert analysis.split_method == "allocation"
    assert analysis.left is None and analysis.right is None
    assert "split unavailable" in analysis.unavailable_reason
    # load on the human is -lambda, so the wrench on the club is +(-lambda)
    # sign-flipped back: on-club = -R lambda (allocator convention).
    np.testing.assert_allclose(
        analysis.net_force_n, -rotation @ LAMBDA[:3], rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        analysis.couple_at_midpoint_nm, -rotation @ LAMBDA[3:], rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(analysis.midpoint_m, point, rtol=0, atol=1e-12)


def test_overlay_has_net_and_couple_but_no_per_hand_arrows(setup) -> None:
    adapter, _, _, q = setup
    analysis = adapter.grip_analysis_from_allocation(q, LAMBDA)
    labels = {w.label for w in to_overlay_wrenches(analysis, source="t")}
    assert labels == {"grip:net_midpoint", "grip:couple_midpoint"}


def test_trajectory_routes_every_frame(setup) -> None:
    adapter, _, _, q = setup
    q_traj = np.stack([q, q + 0.01])
    lam_traj = np.stack([LAMBDA, 2.0 * LAMBDA])
    analyses = adapter.grip_analyses_from_allocation(q_traj, lam_traj)
    assert len(analyses) == 2
    single = adapter.grip_analysis_from_allocation(q_traj[1], lam_traj[1])
    assert analyses[1].net_force_n == single.net_force_n
    assert all(a.split_method == "allocation" for a in analyses)


def test_rejects_bad_shapes(setup) -> None:
    adapter, _, _, q = setup
    with pytest.raises(ValueError, match="lambda_grip"):
        adapter.grip_analysis_from_allocation(q, np.zeros(5))
    with pytest.raises(ValueError, match="configuration"):
        adapter.grip_analysis_from_allocation(q[:3], LAMBDA)
    with pytest.raises(ValueError, match="frame count"):
        adapter.grip_analyses_from_allocation(np.stack([q, q]), LAMBDA[None, :])
