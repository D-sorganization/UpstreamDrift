"""Real-Pinocchio ground truth for the reference adapter base frames (#12605).

A rotated free-flyer with a non-zero WORLD-frame linear velocity is converted by
:class:`PinocchioReferenceAdapter` and handed to Pinocchio. Pinocchio's own
kinematics must then report the canonical twist: ``oMi[1].rotation @ v.linear``
equals the canonical world linear velocity, ``v.angular`` the body angular
velocity, and the world classical acceleration the canonical linear
acceleration. Before the fix the adapter passed ``v[0:3]`` through unchanged, so
Pinocchio read the world vector as a LOCAL one.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.pose_interchange import se3
from src.shared.python.pose_interchange.adapters.pinocchio_reference import (
    CanonicalV2State,
    PinocchioReferenceAdapter,
)

pytestmark = [pytest.mark.integration, pytest.mark.requires_pinocchio]


class _UnusedBackend:
    def fk(self, q, frame_name):  # pragma: no cover - not exercised
        raise AssertionError

    def jacobian(self, q, frame_name):  # pragma: no cover - not exercised
        raise AssertionError

    def rnea(self, q, v, a):  # pragma: no cover - not exercised
        raise AssertionError

    def aba(self, q, v, tau):  # pragma: no cover - not exercised
        raise AssertionError


def test_rotated_base_world_velocity_round_trips_through_pinocchio() -> None:
    pin = pytest.importorskip("pinocchio")
    model = pin.Model()
    base = model.addJoint(0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), "root")
    model.appendBodyToJoint(base, pin.Inertia.Random(), pin.SE3.Identity())
    model.addJoint(base, pin.JointModelRZ(), pin.SE3.Identity(), "hinge")
    data = model.createData()

    rng = np.random.default_rng(12605)
    quat = se3.quat_exp(np.array([0.4, -0.9, 1.3]))
    canonical = CanonicalV2State(
        q=np.concatenate([rng.normal(size=3), quat, [0.3]]),
        v=np.concatenate([[1.5, -0.7, 2.0], rng.normal(size=4)]),
        a=rng.normal(size=7),
    )
    adapter = PinocchioReferenceAdapter(_UnusedBackend())

    native = adapter.from_canonical_v2(canonical)
    pin.forwardKinematics(model, data, native.q, native.v, native.a)

    rot = data.oMi[1].rotation
    np.testing.assert_allclose(rot, se3.quat_to_matrix(quat), atol=1e-12)
    np.testing.assert_allclose(rot @ data.v[1].linear, canonical.v[0:3], atol=1e-12)
    np.testing.assert_allclose(data.v[1].angular, canonical.v[3:6], atol=1e-12)
    classical = pin.getClassicalAcceleration(model, data, 1, pin.ReferenceFrame.LOCAL)
    np.testing.assert_allclose(rot @ classical.linear, canonical.a[0:3], atol=1e-10)
    np.testing.assert_allclose(data.a[1].angular, canonical.a[3:6], atol=1e-12)

    recovered = adapter.to_canonical_v2(native)
    np.testing.assert_allclose(recovered.v, canonical.v, atol=1e-12)
    np.testing.assert_allclose(recovered.a, canonical.a, atol=1e-12)
