"""Tests for src.shared.python.spatial_algebra.joints (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.spatial_algebra.joints import (
    JOINT_AXIS_INDICES,
    S_PX,
    S_RX,
    S_RY,
    S_RZ,
    jcalc,
)


class TestMotionSubspaces:
    def test_s_rx_shape(self) -> None:
        assert S_RX.shape == (6,)

    def test_s_rx_is_unit_at_index_0(self) -> None:
        assert S_RX[0] == pytest.approx(1.0)
        assert np.all(S_RX[1:] == 0.0)

    def test_s_rz_is_unit_at_index_2(self) -> None:
        assert S_RZ[2] == pytest.approx(1.0)
        assert S_RZ[0] == pytest.approx(0.0)

    def test_s_px_is_unit_at_index_3(self) -> None:
        assert S_PX[3] == pytest.approx(1.0)
        assert np.all(S_PX[:3] == 0.0)

    def test_subspaces_are_read_only(self) -> None:
        with pytest.raises(ValueError):
            S_RX[0] = 0.0


class TestJointAxisIndices:
    def test_rx_index_is_0(self) -> None:
        assert JOINT_AXIS_INDICES["Rx"] == 0

    def test_rz_index_is_2(self) -> None:
        assert JOINT_AXIS_INDICES["Rz"] == 2

    def test_px_index_is_3(self) -> None:
        assert JOINT_AXIS_INDICES["Px"] == 3

    def test_pz_index_is_5(self) -> None:
        assert JOINT_AXIS_INDICES["Pz"] == 5


class TestJcalc:
    def test_rx_returns_tuple(self) -> None:
        X, S, dof = jcalc("Rx", 0.0)
        assert X.shape == (6, 6)
        assert S.shape == (6,)
        assert dof == 0

    def test_rx_at_zero_is_identity(self) -> None:
        X, S, dof = jcalc("Rx", 0.0)
        np.testing.assert_allclose(X, np.eye(6), atol=1e-12)

    def test_ry_at_zero_is_identity(self) -> None:
        X, S, dof = jcalc("Ry", 0.0)
        np.testing.assert_allclose(X, np.eye(6), atol=1e-12)

    def test_rz_at_zero_is_identity(self) -> None:
        X, S, dof = jcalc("Rz", 0.0)
        np.testing.assert_allclose(X, np.eye(6), atol=1e-12)

    def test_px_at_zero_is_identity(self) -> None:
        X, S, dof = jcalc("Px", 0.0)
        np.testing.assert_allclose(X, np.eye(6), atol=1e-12)

    def test_rx_subspace_is_s_rx(self) -> None:
        _, S, _ = jcalc("Rx", 0.5)
        np.testing.assert_array_equal(S, S_RX)

    def test_ry_subspace_is_s_ry(self) -> None:
        _, S, _ = jcalc("Ry", 0.3)
        np.testing.assert_array_equal(S, S_RY)

    def test_rz_subspace_is_s_rz(self) -> None:
        _, S, _ = jcalc("Rz", 1.0)
        np.testing.assert_array_equal(S, S_RZ)

    def test_px_subspace_is_s_px(self) -> None:
        _, S, _ = jcalc("Px", 0.5)
        np.testing.assert_array_equal(S, S_PX)

    def test_rx_nonzero_angle_changes_transform(self) -> None:
        X_zero, _, _ = jcalc("Rx", 0.0)
        X_nonzero, _, _ = jcalc("Rx", np.pi / 4)
        assert not np.allclose(X_zero, X_nonzero)

    def test_all_joint_types_return_valid_shapes(self) -> None:
        for jtype in ["Rx", "Ry", "Rz", "Px", "Py", "Pz"]:
            X, S, _ = jcalc(jtype, 0.1)
            assert X.shape == (6, 6), f"Wrong shape for {jtype}"
            assert S.shape == (6,), f"Wrong subspace for {jtype}"
