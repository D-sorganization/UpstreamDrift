"""Tests for src.shared.python.pendulum_simulator.constants (Issues #1949, #1744)."""

from __future__ import annotations

import pytest

from src.shared.python.pendulum_simulator.constants import (
    GRAVITY_MSS,
    GRAVITY_STANDARD,
    INCHES_PER_M,
    LBF_PER_N,
    M_PER_INCH,
    NM_PER_KGFM,
)


class TestGravityConstants:
    def test_gravity_mss_approx_9_81(self) -> None:
        assert pytest.approx(9.81) == GRAVITY_MSS

    def test_gravity_standard_approx_9_80665(self) -> None:
        assert pytest.approx(9.80665) == GRAVITY_STANDARD

    def test_gravity_mss_positive(self) -> None:
        assert GRAVITY_MSS > 0.0

    def test_gravity_standard_positive(self) -> None:
        assert GRAVITY_STANDARD > 0.0

    def test_gravity_mss_close_to_standard(self) -> None:
        # Both within 1% of each other
        assert abs(GRAVITY_MSS - GRAVITY_STANDARD) < 0.1


class TestConversionFactors:
    def test_nm_per_kgfm_equals_gravity_standard(self) -> None:
        assert pytest.approx(GRAVITY_STANDARD) == NM_PER_KGFM

    def test_lbf_per_n_positive(self) -> None:
        assert LBF_PER_N > 0.0

    def test_lbf_per_n_approx(self) -> None:
        # 1 N ≈ 0.2248 lbf
        assert pytest.approx(0.224809, rel=1e-3) == LBF_PER_N

    def test_inches_per_m_approx(self) -> None:
        assert pytest.approx(39.3701, rel=1e-3) == INCHES_PER_M

    def test_m_per_inch_approx(self) -> None:
        assert pytest.approx(0.0254, rel=1e-3) == M_PER_INCH

    def test_inches_per_m_and_m_per_inch_are_reciprocals(self) -> None:
        assert pytest.approx(1.0, rel=1e-3) == INCHES_PER_M * M_PER_INCH

    def test_all_conversion_factors_positive(self) -> None:
        for val in (NM_PER_KGFM, LBF_PER_N, INCHES_PER_M, M_PER_INCH):
            assert val > 0.0
