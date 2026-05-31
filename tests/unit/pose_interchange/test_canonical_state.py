"""Tests for the canonical-v2 ``CanonicalState`` value type (CC-2).

Contract: docs/conventions/canonical-v2.md (ADR-0026).
q = [base_xyz(3), base_quat_wxyz(4), joints(n_j)]  -> nq = 7 + n_j
v = a = [base_lin(3), base_ang(3), joints(n_j)]     -> nv = 6 + n_j  (nq = nv + 1)
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.shared.python.pose_interchange import se3
from src.shared.python.pose_interchange.canonical_state import (
    CONVENTION_TAG_V2,
    CanonicalState,
    canonical_state_zero,
)

pytestmark = pytest.mark.unit


def _make(n_joints: int = 2) -> CanonicalState:
    q = np.concatenate(
        [np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(n_joints)]
    )
    v = np.zeros(6 + n_joints)
    return CanonicalState(q=q, v=v, a=v.copy(), t=0.0)


def _random_state(rng: np.random.Generator, n_joints: int = 3) -> CanonicalState:
    pos = rng.uniform(-1, 1, 3)
    quat = se3.quat_exp(rng.uniform(-1.5, 1.5, 3))
    joints = rng.uniform(-2, 2, n_joints)
    q = np.concatenate([pos, quat, joints])
    v = rng.uniform(-1, 1, 6 + n_joints)
    a = rng.uniform(-1, 1, 6 + n_joints)
    return CanonicalState(q=q, v=v, a=a, t=float(rng.uniform(0, 5)))


# ---- construction / validation ----------------------------------------------


def test_construction_and_shapes() -> None:
    s = _make(n_joints=4)
    assert s.n_joints == 4
    assert s.nq == 11
    assert s.nv == 10
    assert s.convention == CONVENTION_TAG_V2
    assert s.frame == "world_Zup"
    assert s.units == "SI"


def test_accessors() -> None:
    s = _make(n_joints=2)
    assert np.allclose(s.base_position, np.zeros(3))
    assert np.allclose(s.base_quat_wxyz, [1.0, 0.0, 0.0, 0.0])
    assert s.joint_q.shape == (2,)


def test_rejects_non_unit_quaternion() -> None:
    q = np.concatenate([np.zeros(3), np.array([2.0, 0.0, 0.0, 0.0]), np.zeros(2)])
    with pytest.raises(ValueError, match="unit norm"):
        CanonicalState(q=q, v=np.zeros(8), a=np.zeros(8), t=0.0)


def test_rejects_nq_nv_mismatch() -> None:
    q = np.concatenate([np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(2)])
    with pytest.raises(ValueError, match="nq must equal nv"):
        CanonicalState(q=q, v=np.zeros(9), a=np.zeros(9), t=0.0)  # nv should be 8


def test_rejects_short_q() -> None:
    with pytest.raises(ValueError, match="at least 7"):
        CanonicalState(q=np.zeros(5), v=np.zeros(4), a=np.zeros(4), t=0.0)


def test_rejects_non_finite() -> None:
    s = _make()
    bad = s.q.copy()
    bad[0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        CanonicalState(q=bad, v=s.v, a=s.a, t=0.0)


def test_rejects_unknown_convention() -> None:
    s = _make()
    with pytest.raises(ValueError, match="convention"):
        CanonicalState(q=s.q, v=s.v, a=s.a, t=0.0, convention="canonical-v1")


def test_is_frozen_and_arrays_readonly() -> None:
    s = _make()
    with pytest.raises((AttributeError, TypeError)):
        s.t = 1.0  # type: ignore[misc]
    with pytest.raises(ValueError):
        s.q[0] = 5.0  # arrays are write-protected


# ---- manifold ops: integrate / difference -----------------------------------


def test_integrate_zero_is_identity() -> None:
    s = _make(n_joints=3)
    out = s.integrate(np.zeros(s.nv))
    assert np.allclose(out.q, s.q)


def test_integrate_translation_and_joints_add() -> None:
    s = _make(n_joints=2)
    dq = np.zeros(s.nv)
    dq[0] = 0.5  # base x
    dq[6] = 0.25  # joint 0
    out = s.integrate(dq)
    assert np.isclose(out.base_position[0], 0.5)
    assert np.isclose(out.joint_q[0], 0.25)


def test_integrate_rotation_uses_manifold() -> None:
    s = _make()
    dq = np.zeros(s.nv)
    dq[3:6] = np.array([0.0, 0.0, np.pi / 2])  # body-frame yaw
    out = s.integrate(dq)
    expected = se3.quat_exp(np.array([0.0, 0.0, np.pi / 2]))
    assert np.allclose(out.base_quat_wxyz, expected, atol=1e-12)
    assert np.isclose(np.linalg.norm(out.base_quat_wxyz), 1.0)


@given(st.integers(min_value=0, max_value=400))
@settings(max_examples=60, deadline=None)
def test_integrate_difference_roundtrip(seed: int) -> None:
    """a.integrate(a.difference(b)) == b  (the manifold round-trip invariant)."""
    rng = np.random.default_rng(seed)
    a = _random_state(rng, n_joints=3)
    b = _random_state(rng, n_joints=3)
    dq = a.difference(b)
    recovered = a.integrate(dq)
    assert np.allclose(recovered.base_position, b.base_position, atol=1e-9)
    assert np.allclose(
        se3.quat_to_matrix(recovered.base_quat_wxyz),
        se3.quat_to_matrix(b.base_quat_wxyz),
        atol=1e-9,
    )
    assert np.allclose(recovered.joint_q, b.joint_q, atol=1e-9)


def test_difference_rejects_shape_mismatch() -> None:
    a = _make(n_joints=2)
    b = _make(n_joints=3)
    with pytest.raises(ValueError, match="same n_joints"):
        a.difference(b)


# ---- v1 -> v2 migration ------------------------------------------------------


def test_from_canonical_pose_zero_velocity() -> None:
    pose_mod = pytest.importorskip("src.shared.python.pose_interchange.canonical")
    pose = pose_mod.canonical_from_reference_setup()
    s = CanonicalState.from_canonical_pose(pose)
    assert np.allclose(s.v, 0.0)
    assert np.allclose(s.a, 0.0)
    assert np.isclose(np.linalg.norm(s.base_quat_wxyz), 1.0)
    # joints are the reference angles in radians, in field order
    assert s.n_joints == len(pose.angles_full_dict_rad())


def test_zero_constructor() -> None:
    s = canonical_state_zero(n_joints=5)
    assert s.nq == 12
    assert np.allclose(s.base_quat_wxyz, [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(s.v, 0.0)
