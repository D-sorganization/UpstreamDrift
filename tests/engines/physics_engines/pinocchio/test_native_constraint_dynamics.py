"""Native capability test; this does not certify the golfer's physical parity."""

import numpy as np
import pytest

pin = pytest.importorskip("pinocchio")
pytestmark = [pytest.mark.live_simulation, pytest.mark.requires_pinocchio]


def test_rigid_closure_reacts_to_all_six_applied_efforts() -> None:
    """A body welded to world must resist forces and moments in every axis."""
    model = pin.Model()
    joint = model.addJoint(
        0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), "free_body"
    )
    model.appendBodyToJoint(
        joint, pin.Inertia(2.0, np.zeros(3), np.eye(3)), pin.SE3.Identity()
    )
    model.gravity.linear[:] = 0.0
    q, v = pin.neutral(model), np.zeros(model.nv)
    closure = pin.RigidConstraintModel(
        pin.ContactType.CONTACT_6D,
        model,
        joint,
        pin.SE3.Identity(),
        0,
        pin.SE3.Identity(),
        pin.ReferenceFrame.LOCAL,
    )
    closures = [closure]
    closure_data = [closure.createData()]
    data = model.createData()
    pin.initConstraintDynamics(model, data, closures, closure_data)
    for effort in np.eye(model.nv):
        free_acceleration = pin.aba(model, model.createData(), q, v, effort)
        assert np.linalg.norm(free_acceleration) >= 0.5
        acceleration = pin.constraintDynamics(
            model, data, q, v, effort, closures, closure_data
        )
        np.testing.assert_allclose(acceleration, 0.0, atol=1e-10)
        assert np.all(np.isfinite(closure_data[0].contact_force.vector))
