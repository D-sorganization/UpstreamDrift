"""DbC-gap regression tests for pose_interchange adapters (issue #7145)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.pose_interchange.adapters import _base
from src.shared.python.pose_interchange.adapters.drake import (
    _select_base_quaternion,
)
from src.shared.python.pose_interchange.adapters.pinocchio import PinocchioAdapter
from src.shared.python.pose_interchange.protocol import JointSlot

pytestmark = pytest.mark.unit


# --- Defect 1: symmetric quaternion-layout discrimination (Drake) ----------


def _q_with_quat(quat: np.ndarray, *, at_canonical: bool) -> np.ndarray:
    # The q[0:4] and q[3:7] blocks overlap at index 3, so fill the decoy block
    # first and write the real (unit) quaternion last. Decoy norm is ~1.6 (far
    # from unit) so only the real block passes the unit-norm test.
    q = np.zeros(8, dtype=np.float64)
    if at_canonical:
        q[0:4] = [0.8, 0.8, 0.8, 0.8]
        q[3:7] = quat
    else:
        q[3:7] = [0.8, 0.8, 0.8, 0.8]
        q[0:4] = quat
    return q


def test_select_quaternion_picks_canonical_block_when_only_it_is_unit() -> None:
    q = _q_with_quat(np.array([1.0, 0.0, 0.0, 0.0]), at_canonical=True)
    np.testing.assert_allclose(_select_base_quaternion(q), [1.0, 0.0, 0.0, 0.0])


def test_select_quaternion_picks_legacy_block_when_only_it_is_unit() -> None:
    q = _q_with_quat(np.array([0.0, 1.0, 0.0, 0.0]), at_canonical=False)
    np.testing.assert_allclose(_select_base_quaternion(q), [0.0, 1.0, 0.0, 0.0])


def test_select_quaternion_raises_when_both_blocks_unit_norm() -> None:
    # Both candidate blocks are unit-norm -> ambiguous -> must raise, not guess.
    q = np.zeros(8, dtype=np.float64)
    q[0:4] = [1.0, 0.0, 0.0, 0.0]
    q[3:7] = [0.0, 0.0, 0.0, 1.0]
    with pytest.raises(ValueError, match="ambiguous"):
        _select_base_quaternion(q)


def test_select_quaternion_raises_when_neither_block_unit_norm() -> None:
    q = np.full(8, 0.3, dtype=np.float64)
    with pytest.raises(ValueError, match="could not be determined"):
        _select_base_quaternion(q)


@pytest.mark.parametrize("delta", [5e-7, -5e-7, 1.5e-6, -1.5e-6])
def test_select_quaternion_boundary_norms(delta: float) -> None:
    # Canonical block at norm 1+delta; legacy block clearly non-unit.
    quat = np.array([1.0 + delta, 0.0, 0.0, 0.0])
    q = _q_with_quat(quat, at_canonical=True)
    if abs(delta) < 1e-6:
        np.testing.assert_allclose(_select_base_quaternion(q), quat)
    else:
        with pytest.raises(ValueError):
            _select_base_quaternion(q)


# --- Defect 2: encode/decode reject multi-DOF slots ------------------------


def _multi_dof_layout() -> dict[str, JointSlot]:
    return {
        "spine": JointSlot(
            canonical_name="spine",
            engine_name="spine",
            start_index=7,
            length=2,
            units="rad",
            sign=1,
        )
    }


def test_encode_joint_angles_rejects_multi_dof_slot() -> None:
    q = np.zeros(10, dtype=np.float64)
    with pytest.raises(NotImplementedError, match="length=2"):
        _base.encode_joint_angles({"spine": 1.0}, _multi_dof_layout(), q)


def test_decode_joint_angles_rejects_multi_dof_slot() -> None:
    q = np.zeros(10, dtype=np.float64)
    with pytest.raises(NotImplementedError, match="length=2"):
        _base.decode_joint_angles(q, _multi_dof_layout())


# --- Defect 3: actionable error when a real Model lacks a layout -----------


class _FakePinocchioModel:
    """Stands in for a real pinocchio.Model: no joint_layout attribute."""

    name = "humanoid"


def test_pinocchio_layout_error_states_remediation() -> None:
    adapter = PinocchioAdapter()
    with pytest.raises(TypeError) as excinfo:
        adapter.joint_layout(_FakePinocchioModel())
    message = str(excinfo.value)
    assert "joint_layout" in message
    assert "build_default_joint_layout" in message
