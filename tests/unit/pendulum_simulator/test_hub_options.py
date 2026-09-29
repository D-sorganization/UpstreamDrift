"""Tests for src.shared.python.pendulum_simulator.hub_options (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.pendulum_simulator.hub_options import (
    compute_system_com,
    effective_hub_mass,
    hub_offset_for_com,
    make_massless_hub_params,
)
from src.shared.python.pendulum_simulator.physics_golfer import GolferParams


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
_Q0 = np.zeros(8)


class TestEffectiveHubMass:
    def test_normal_mass_returned_when_not_massless(self) -> None:
        assert effective_hub_mass(5.0) == pytest.approx(5.0)

    def test_massless_returns_epsilon(self) -> None:
        result = effective_hub_mass(10.0, massless=True)
        assert result > 0.0
        assert result < 1e-4  # very small

    def test_normal_mass_positive(self) -> None:
        assert effective_hub_mass(1.0) > 0.0

    def test_massless_positive(self) -> None:
        assert effective_hub_mass(10.0, massless=True) > 0.0


class TestMakeMasslessHubParams:
    def test_returns_golfer_params(self) -> None:
        result = make_massless_hub_params(_P)
        assert isinstance(result, GolferParams)

    def test_hub_mass_nearly_zero(self) -> None:
        result = make_massless_hub_params(_P)
        assert result.m_hub < 1e-4
        assert result.m_hub > 0.0

    def test_other_params_unchanged(self) -> None:
        result = make_massless_hub_params(_P)
        assert result.m_r_upper == _P.m_r_upper
        assert result.L_hub == _P.L_hub
        assert result.L_club == _P.L_club

    def test_original_unchanged(self) -> None:
        original_mass = _P.m_hub
        make_massless_hub_params(_P)
        assert _P.m_hub == original_mass


class TestComputeSystemCom:
    def test_hub_options_returns_shape_2(self) -> None:
        result = compute_system_com(_Q0, _P)
        assert result.shape == (2,)

    def test_hub_options_values_are_finite(self) -> None:
        result = compute_system_com(_Q0, _P)
        assert np.all(np.isfinite(result))

    def test_hub_options_returns_ndarray(self) -> None:
        result = compute_system_com(_Q0, _P)
        assert isinstance(result, np.ndarray)

    def test_heavier_hub_pulls_com_upward(self) -> None:
        # Hub is above origin → heavier hub → COM higher
        p_heavy_hub = GolferParams(
            m_hub=100.0, **{k: v for k, v in _P.__dict__.items() if k != "m_hub"}
        )
        com_heavy = compute_system_com(_Q0, p_heavy_hub)
        com_light = compute_system_com(_Q0, _P)
        # Heavy hub → COM closer to hub position (higher y)
        assert com_heavy[1] > com_light[1]


class TestHubOffsetForCom:
    def test_hub_options_returns_tuple(self) -> None:
        result = hub_offset_for_com(_Q0, _P)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_values_are_floats(self) -> None:
        dx, dy = hub_offset_for_com(_Q0, _P)
        assert isinstance(dx, float)
        assert isinstance(dy, float)

    def test_hub_options_values_are_finite(self) -> None:
        dx, dy = hub_offset_for_com(_Q0, _P)
        assert np.isfinite(dx) and np.isfinite(dy)
