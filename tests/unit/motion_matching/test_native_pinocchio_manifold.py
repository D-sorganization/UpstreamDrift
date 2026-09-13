"""Live native scalar/manifold equivalence, away from singular charts."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.pinocchio.python.native_manifold_model import (
    NativeManifoldPinocchioModel,
)
from src.engines.physics_engines.pinocchio.python.native_model import (
    NativePinocchioModel,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def models():
    pin = pytest.importorskip("pinocchio")
    if not isinstance(getattr(pin, "__version__", None), str):
        pytest.skip("Real Pinocchio runtime required; suite supplied a mock")
    root = Path(__file__).resolve().parents[3]
    spec = json.loads(
        (
            root
            / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
        ).read_text()
    )
    return NativePinocchioModel(spec), NativeManifoldPinocchioModel(spec)


def test_native_inventory_frames_energy_velocity_and_power(models):
    scalar, manifold = models
    pin = manifold.pin
    assert manifold.model.nq == 30
    assert manifold.model.nv == 27
    assert len(manifold.adapter.groups) == 3
    assert len(manifold.constraints) == 1
    rng = np.random.default_rng(10043)
    for _ in range(8):
        names = manifold.adapter.coordinate_order
        q, v, tau = (
            dict(zip(names, rng.uniform(-0.4, 0.4, len(names)), strict=True))
            for _ in range(3)
        )
        mq, mv, mtau = manifold.native_state(q, v, tau)
        poses = manifold.frame_poses(mq)
        for name, pose in scalar.frame_poses(q).items():
            np.testing.assert_allclose(poses[name], pose, atol=2e-13)
        sq, sv = scalar.configuration(q), scalar._velocity_vector(v)
        scalar_energy = pin.computeKineticEnergy(scalar.model, scalar.data, sq, sv)
        manifold_energy = pin.computeKineticEnergy(
            manifold.model, manifold.data, mq, mv
        )
        np.testing.assert_allclose(manifold_energy, scalar_energy, atol=2e-12)
        np.testing.assert_allclose(
            np.dot(mv, mtau), sum(v[n] * tau[n] for n in names), atol=2e-13
        )
        pin.forwardKinematics(scalar.model, scalar.data, sq, sv)
        pin.forwardKinematics(manifold.model, manifold.data, mq, mv)
        for name, index in scalar._frames.items():
            actual = manifold.frame_velocities(mq, mv)[name]
            expected = pin.getFrameVelocity(
                scalar.model, scalar.data, index, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            ).vector
            np.testing.assert_allclose(actual, expected, atol=2e-12)
        restored_q, restored_v = manifold.native_coordinates(mq, mv, q)
        np.testing.assert_allclose(
            list(restored_q.values()), list(q.values()), atol=2e-13
        )
        np.testing.assert_allclose(
            list(restored_v.values()), list(v.values()), atol=2e-13
        )


def test_manifold_integrate_uses_body_tangent_and_roundtrips(models):
    _, manifold = models
    names = manifold.adapter.coordinate_order
    q = dict.fromkeys(names, 0.2)
    v = dict.fromkeys(names, 0.3)
    mq, mv, _ = manifold.native_state(q, v, dict.fromkeys(names, 0.0))
    increment = mv * 1e-7
    next_q = manifold.integrate(mq, increment)
    np.testing.assert_allclose(manifold.difference(mq, next_q), increment, atol=2e-15)
    native, _ = manifold.native_coordinates(next_q, mv, q)
    np.testing.assert_allclose(
        np.array(list(native.values())),
        np.array(list(q.values())) + 1e-7 * np.array(list(v.values())),
        atol=2e-14,
    )


def test_manifold_rejects_nonunit_quaternion_and_wrong_shapes(models):
    _, manifold = models
    names = manifold.adapter.coordinate_order
    zeros = dict.fromkeys(names, 0.0)
    mq, mv, _ = manifold.native_state(zeros, zeros, zeros)
    with pytest.raises(ValueError, match="configuration"):
        manifold.integrate(mq[:-1], mv)
    with pytest.raises(ValueError, match="tangent"):
        manifold.integrate(mq, mv[:-1])
    mq[:] = 0
    with pytest.raises(ValueError, match="quaternion"):
        manifold.frame_poses(mq)


def test_constrained_acceleration_and_weld_parity(models):
    scalar, manifold = models
    rng = np.random.default_rng(10044)
    names = manifold.adapter.coordinate_order
    for _ in range(6):
        q, v, tau = (
            dict(zip(names, rng.uniform(-0.3, 0.3, len(names)), strict=True))
            for _ in range(3)
        )
        expected = scalar.accelerations(q, v, tau)
        actual = manifold.native_accelerations(q, v, tau)
        np.testing.assert_allclose(
            [actual[n] for n in names],
            [expected[n] for n in names],
            rtol=2e-9,
            atol=2e-9,
        )
        expected_pose, expected_rate = scalar.closure_errors()
        actual_pose, actual_rate = manifold.closure_errors()
        np.testing.assert_allclose(actual_pose, expected_pose, atol=2e-13)
        np.testing.assert_allclose(actual_rate, expected_rate, atol=2e-13)


def test_native_efforts_are_mapped_at_actual_configuration(models):
    scalar, manifold = models
    names = manifold.adapter.coordinate_order
    q = dict.fromkeys(names, 0.35)
    v = dict.fromkeys(names, -0.2)
    tau = dict.fromkeys(names, 1.2)
    reference = dict.fromkeys(names, -0.1)
    mq, mv, _ = manifold.native_state(q, v, tau)
    expected_qdd = scalar.accelerations(q, v, tau)
    actual_a = manifold.acceleration_from_native_efforts(mq, mv, tau, reference)
    native_state = manifold._native_state(mq, mv, actual_a)
    actual_qdd = manifold.adapter.restore(native_state, q)[2]
    np.testing.assert_allclose(
        [actual_qdd[n] for n in names],
        [expected_qdd[n] for n in names],
        rtol=2e-9,
        atol=2e-9,
    )


def test_actual_engine_difference_rate_matches_centered_tangent_probe(models):
    _, manifold = models
    names = manifold.adapter.coordinate_order
    native = dict.fromkeys(names, 0.2)
    q, v, _ = manifold.native_state(
        native, dict.fromkeys(names, 0.3), dict.fromkeys(names, 0.0)
    )
    anchor = manifold.integrate(q, -0.2 * v)
    epsilon = 1e-6
    expected = (
        manifold.difference(anchor, manifold.integrate(q, epsilon * v))
        - manifold.difference(anchor, manifold.integrate(q, -epsilon * v))
    ) / (2 * epsilon)
    np.testing.assert_allclose(
        manifold.difference_rate(anchor, q, v), expected, atol=2e-10
    )


def test_actual_engine_noncommuting_rotation_integrates_at_fourth_order(models):
    from scipy.spatial.transform import Rotation
    from src.shared.python.motion_matching.manifold_forward import (
        integrate_manifold_forward,
    )

    _, manifold = models
    q0 = manifold.pin.neutral(manifold.model)
    qindex, vindex = next(iter(manifold._tree.spherical.values()))
    v0 = np.zeros(manifold.model.nv)
    v0[vindex] = 1.0
    target = Rotation.from_rotvec([1, 0, 0]) * Rotation.from_rotvec([0, 1, 0])
    errors = []

    def acceleration(t, q, v):
        a = np.zeros(manifold.model.nv)
        a[vindex : vindex + 3] = [-2 * t * np.sin(t * t), 2, 2 * t * np.cos(t * t)]
        return a

    for h in (0.2, 0.1, 0.05):
        result = integrate_manifold_forward(
            q0,
            v0,
            np.array([0.0, 1.0]),
            acceleration,
            integrate=manifold.integrate,
            difference_rate=manifold.difference_rate,
            max_step=h,
        )
        actual = Rotation.from_quat(result.configuration[-1, qindex : qindex + 4])
        errors.append(np.linalg.norm((target.inv() * actual).as_rotvec()))
    assert 12 < errors[0] / errors[1] < 20
    assert 12 < errors[1] / errors[2] < 20
    assert errors[-1] < 3e-7


def test_native_dynamics_preserves_middle_branch_when_nearest_inverse_flips(models):
    _, manifold = models
    names = manifold.adapter.coordinate_order
    q = dict.fromkeys(names, 0.2)
    reference = dict(q)
    group = next(g for g in manifold.adapter.groups if g.coordinates[1] == "LSInputY")
    for name, target, prior in zip(
        group.coordinates, [2.8, -1.6, 2.8], [0, -2, 0], strict=True
    ):
        q[name], reference[name] = target, prior
    rates = dict.fromkeys(names, 0.3)
    efforts = dict.fromkeys(names, 1.5)
    mq, mv, mapped_effort = manifold.native_state(q, rates, efforts)
    restored_q, restored_v = manifold.native_coordinates(mq, mv, reference)
    np.testing.assert_allclose(list(restored_q.values()), list(q.values()), atol=2e-13)
    np.testing.assert_allclose(
        list(restored_v.values()), list(rates.values()), atol=2e-12
    )
    expected = manifold.acceleration(mq, mv, mapped_effort)
    actual = manifold.acceleration_from_native_efforts(mq, mv, efforts, reference)
    np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-9)


def test_actual_engine_adaptive_rotation_matches_exact_noncommuting_motion(models):
    from scipy.spatial.transform import Rotation
    from src.shared.python.motion_matching.manifold_forward import (
        integrate_manifold_adaptive,
    )

    _, manifold = models
    q0 = manifold.pin.neutral(manifold.model)
    qi, vi = next(iter(manifold._tree.spherical.values()))
    v0 = np.zeros(manifold.model.nv)
    v0[vi] = 1.0

    def acceleration(t, q, v):
        a = np.zeros(manifold.model.nv)
        a[vi : vi + 3] = [-2 * t * np.sin(t * t), 2, 2 * t * np.cos(t * t)]
        return a

    result = integrate_manifold_adaptive(
        q0,
        v0,
        np.array([0.0, 1.0]),
        acceleration,
        integrate=manifold.integrate,
        difference_rate=manifold.difference_rate,
        difference=manifold.difference,
        rtol=1e-9,
        atol=1e-11,
        max_step=0.2,
    )
    target = Rotation.from_rotvec([1, 0, 0]) * Rotation.from_rotvec([0, 1, 0])
    actual = Rotation.from_quat(result.configuration[-1, qi : qi + 4])
    assert np.linalg.norm((target.inv() * actual).as_rotvec()) < 2e-8
    np.testing.assert_allclose(
        result.velocity[-1, vi : vi + 3], [np.cos(1), 2, np.sin(1)], atol=2e-8
    )
