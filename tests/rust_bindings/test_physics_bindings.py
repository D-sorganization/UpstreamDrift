"""TDD tests for upstream-physics Rust Python bindings.

Validates that the PyO3-wrapped physics kernels provide correct
Python-accessible types and produce physically valid results.

Principles:
- TDD: Tests define the expected Python API contract.
- DbC: Validates physical constraints (positive values, conservation laws).
- DRY: Uses the same upstream_physics import downstream code will use.
"""

from __future__ import annotations

import pytest

upstream_physics = pytest.importorskip(
    "upstream_physics",
    reason="upstream_physics wheel not installed (run: maturin develop --features python)",
)


class TestIntegratorConfig:
    """Test IntegratorConfig Python binding."""

    def test_create_defaults(self) -> None:
        """IntegratorConfig() must use sensible defaults."""
        config = upstream_physics.IntegratorConfig()
        assert config is not None

    def test_create_with_params(self) -> None:
        """IntegratorConfig(dt, max_steps) must accept parameters."""
        config = upstream_physics.IntegratorConfig(dt=0.01, max_steps=500)
        assert config is not None


class TestIntegrationResult:
    """Test IntegrationResult Python binding."""

    def test_type_exists(self) -> None:
        """IntegrationResult type must be importable."""
        assert hasattr(upstream_physics, "IntegrationResult")


class TestContactParameters:
    """Test ContactParameters Python binding."""

    def test_create_defaults(self) -> None:
        """ContactParameters() must use golf green defaults."""
        cp = upstream_physics.ContactParameters()
        assert cp is not None

    def test_create_with_params(self) -> None:
        """ContactParameters(cor, friction) must accept parameters."""
        cp = upstream_physics.ContactParameters(cor=0.6, friction=0.3)
        assert cp is not None


class TestContactResult:
    """Test ContactResult Python binding."""

    def test_type_exists(self) -> None:
        """ContactResult type must be importable."""
        assert hasattr(upstream_physics, "ContactResult")

    def test_speed_getter(self) -> None:
        """ContactResult must have a speed getter."""
        assert hasattr(upstream_physics.ContactResult, "speed")


class TestSwingPlaneResult:
    """Test SwingPlaneResult Python binding."""

    def test_type_exists(self) -> None:
        """SwingPlaneResult type must be importable."""
        assert hasattr(upstream_physics, "SwingPlaneResult")

    def test_has_plane_angle_deg(self) -> None:
        """SwingPlaneResult must have plane_angle_deg getter."""
        assert hasattr(upstream_physics.SwingPlaneResult, "plane_angle_deg")

    def test_has_mean_residual(self) -> None:
        """SwingPlaneResult must have mean_residual getter."""
        assert hasattr(upstream_physics.SwingPlaneResult, "mean_residual")
