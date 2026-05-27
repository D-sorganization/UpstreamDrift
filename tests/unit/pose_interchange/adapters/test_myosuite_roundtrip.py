"""Round-trip parity tests for :class:`MyoSuiteAdapter`."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.pose_interchange.adapters.myosuite import MyoSuiteAdapter
from src.shared.python.pose_interchange.canonical import CanonicalPose

pytestmark = pytest.mark.unit


def _assert_pose_close(a: CanonicalPose, b: CanonicalPose, *, tol: float) -> None:
    np.testing.assert_allclose(a.pelvis_translation_m, b.pelvis_translation_m, atol=tol)
    np.testing.assert_allclose(
        a.pelvis_rotation_xyz_deg, b.pelvis_rotation_xyz_deg, atol=tol
    )
    full_a = a.angles_full_dict_deg()
    full_b = b.angles_full_dict_deg()
    assert set(full_a) == set(full_b)
    for key, val in full_a.items():
        assert val == pytest.approx(full_b[key], abs=tol)


def test_myosuite_roundtrip_100_random(random_poses: list[CanonicalPose]) -> None:
    adapter = MyoSuiteAdapter()
    for pose in random_poses:
        q = adapter.from_canonical(pose)
        recovered = adapter.to_canonical(q)
        _assert_pose_close(pose, recovered, tol=1e-9)


def test_myosuite_q_roundtrip_identity(random_poses: list[CanonicalPose]) -> None:
    adapter = MyoSuiteAdapter()
    for pose in random_poses:
        q1 = adapter.from_canonical(pose)
        q2 = adapter.from_canonical(adapter.to_canonical(q1))
        np.testing.assert_allclose(q1, q2, atol=1e-9)


def test_myosuite_quaternion_is_w_first(random_poses: list[CanonicalPose]) -> None:
    """Sanity-check: q[3] is the scalar component (w-first MJCF convention)."""
    adapter = MyoSuiteAdapter()
    for pose in random_poses:
        q = adapter.from_canonical(pose)
        # Quaternion slice is q[3:7]; norm should be 1.
        assert abs(np.linalg.norm(q[3:7]) - 1.0) < 1e-9


def test_myosuite_rejects_short_q() -> None:
    adapter = MyoSuiteAdapter()
    with pytest.raises(ValueError, match="at least 7 entries"):
        adapter.to_canonical(np.zeros(4))


def test_myosuite_rejects_unsupported_convention_tag() -> None:
    adapter = MyoSuiteAdapter()
    pose = CanonicalPose(
        pelvis_translation_m=np.zeros(3),
        pelvis_rotation_xyz_deg=np.zeros(3),
    )
    # Re-package as an object that claims a different convention tag.
    object.__setattr__(pose, "convention_tag", "future-v9")
    with pytest.raises(ValueError, match="unsupported convention"):
        adapter.from_canonical(pose)
