"""Tests for src.shared.python.pendulum_simulator.jacobians_golfer (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
from src.shared.python.pendulum_simulator.jacobians_golfer import (
    ellipsoids_golfer,
    jacobian_golfer,
)
from src.shared.python.pendulum_simulator.physics_golfer import N_DOF, GolferParams


def _make_params() -> GolferParams:
    return GolferParams(
        m_hub=10.0,
        m_r_upper=2.0,
        m_r_fore=1.5,
        m_l_upper=2.0,
        m_l_fore=1.5,
        m_club=0.4,
        L_hub=0.2,
        L_r_upper=0.3,
        L_r_fore=0.25,
        L_l_upper=0.3,
        L_l_fore=0.25,
        L_club=1.0,
        d_rs=0.2,
        d_ls=0.2,
        grip_right=0.05,
        grip_left=0.15,
    )


_P = _make_params()
_Q = np.zeros(N_DOF)

_EXPECTED_KEYS = {"rh", "lh", "club_tip", "re", "le", "hub"}


class TestJacobianGolfer:
    def test_jacobians_golfer_returns_dict(self) -> None:
        result = jacobian_golfer(_Q, _P)
        assert isinstance(result, dict)

    def test_jacobians_golfer_has_expected_keys(self) -> None:
        result = jacobian_golfer(_Q, _P)
        assert set(result.keys()) == _EXPECTED_KEYS

    def test_shapes_are_2_by_ndof(self) -> None:
        result = jacobian_golfer(_Q, _P)
        for key, J in result.items():
            assert J.shape == (
                2,
                N_DOF,
            ), f"Key {key}: expected (2, {N_DOF}), got {J.shape}"

    def test_jacobians_golfer_finite_values(self) -> None:
        result = jacobian_golfer(_Q, _P)
        for key, J in result.items():
            assert np.all(np.isfinite(J)), f"Non-finite values in Jacobian for {key}"

    def test_nonzero_angles(self) -> None:
        q = np.zeros(N_DOF)
        q[0] = 0.3
        q[1] = 0.2
        result = jacobian_golfer(q, _P)
        for key, J in result.items():
            assert np.all(
                np.isfinite(J)
            ), f"Non-finite values for {key} at nonzero angles"

    def test_club_tip_different_from_hub(self) -> None:
        result = jacobian_golfer(_Q, _P)
        assert not np.allclose(result["club_tip"], result["hub"])


class TestEllipsoidsGolfer:
    def test_jacobians_golfer_returns_dict(self) -> None:
        result = ellipsoids_golfer(_Q, _P)
        assert isinstance(result, dict)

    def test_jacobians_golfer_has_expected_keys(self) -> None:
        result = ellipsoids_golfer(_Q, _P)
        assert set(result.keys()) == _EXPECTED_KEYS

    def test_each_entry_has_jacobian_key(self) -> None:
        result = ellipsoids_golfer(_Q, _P)
        for key, entry in result.items():
            assert "jacobian" in entry, f"Missing 'jacobian' in entry for {key}"

    def test_each_entry_has_mob_semi_axes(self) -> None:
        result = ellipsoids_golfer(_Q, _P)
        for key, entry in result.items():
            assert "mob_semi_axes" in entry, f"Missing 'mob_semi_axes' for {key}"

    def test_mob_semi_axes_shape(self) -> None:
        result = ellipsoids_golfer(_Q, _P)
        for key, entry in result.items():
            axes = entry["mob_semi_axes"]
            assert axes.shape == (
                2,
            ), f"Key {key}: expected shape (2,), got {axes.shape}"

    def test_mob_semi_axes_non_negative(self) -> None:
        result = ellipsoids_golfer(_Q, _P)
        for key, entry in result.items():
            axes = entry["mob_semi_axes"]
            assert np.all(axes >= 0), f"Negative semi-axes for {key}"

    def test_club_tip_mob_semi_axes_positive(self) -> None:
        result = ellipsoids_golfer(_Q, _P)
        # club_tip is not degenerate
        assert np.all(result["club_tip"]["mob_semi_axes"] > 0)
