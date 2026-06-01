"""Tests for the Rust kernel Python adapter module.

Validates:
- Graceful fallback when Rust wheel is not installed.
- API contract for create_integrator_config / create_contact_parameters.
- Positive-precondition guards (#6934).
- Pure-Python fallback math, asserted on both backends (#6940).
- Diagnostic info correctness on both forced backends (#6942).
"""

from __future__ import annotations

import math

import pytest
from src.shared.python.physics import rust_kernel
from src.shared.python.physics.rust_kernel import (
    create_air_properties,
    create_ball_properties,
    create_contact_parameters,
    create_integrator_config,
    get_kernel_info,
    is_rust_available,
)


@pytest.fixture
def force_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the pure-Python fallback path regardless of wheel presence."""
    monkeypatch.setattr(rust_kernel, "_RUST_AVAILABLE", False)
    monkeypatch.setattr(rust_kernel, "_rust", None)


@pytest.fixture
def force_rust(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the Rust path; skip if the wheel is not installed."""
    if not rust_kernel.is_rust_available():
        pytest.skip("upstream_physics Rust wheel not installed")
    monkeypatch.setattr(rust_kernel, "_RUST_AVAILABLE", True)


class TestRustKernelAvailability:
    """Test kernel availability detection."""

    def test_is_rust_available_returns_bool(self) -> None:
        """is_rust_available() must return a boolean."""
        result = is_rust_available()
        assert isinstance(result, bool)


class TestIntegratorConfig:
    """Test create_integrator_config adapter."""

    def test_rust_kernel_adapter_default_config(self) -> None:
        """Default integrator config must be constructible."""
        config = create_integrator_config()
        assert config is not None

    def test_rust_kernel_adapter_custom_config(self) -> None:
        """Custom config with specified parameters."""
        config = create_integrator_config(dt=0.01, max_steps=500)
        assert config is not None

    def test_fallback_returns_dict(self, force_fallback: None) -> None:
        """Forced fallback returns a dict with the expected key/values."""
        config = create_integrator_config(dt=0.005, max_steps=1000)
        assert isinstance(config, dict)
        assert config["dt"] == 0.005
        assert config["max_steps"] == 1000


class TestContactParameters:
    """Test create_contact_parameters adapter."""

    def test_rust_kernel_adapter_default_params(self) -> None:
        """Default contact params must be constructible."""
        params = create_contact_parameters()
        assert params is not None

    def test_rust_kernel_adapter_custom_params(self) -> None:
        """Custom contact params with specified COR and friction."""
        params = create_contact_parameters(cor=0.6, friction=0.3)
        assert params is not None

    def test_fallback_returns_dict(self, force_fallback: None) -> None:
        """Forced fallback returns a dict with the expected key/values."""
        params = create_contact_parameters(cor=0.75, friction=0.2)
        assert isinstance(params, dict)
        assert params["cor"] == 0.75
        assert params["friction"] == 0.2


class TestAirPropertiesPreconditions:
    """Positive-precondition guards for create_air_properties (#6934)."""

    def test_zero_density_raises(self) -> None:
        with pytest.raises(ValueError, match="density"):
            create_air_properties(density=0.0)

    def test_negative_density_raises(self) -> None:
        with pytest.raises(ValueError, match="density"):
            create_air_properties(density=-1.0)

    def test_non_positive_viscosity_raises(self) -> None:
        with pytest.raises(ValueError, match="viscosity"):
            create_air_properties(viscosity=0.0)

    def test_non_positive_temperature_raises(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            create_air_properties(temperature=-1.0)


class TestBallPropertiesPreconditions:
    """Positive-precondition guards for create_ball_properties (#6934)."""

    def test_zero_mass_raises(self) -> None:
        with pytest.raises(ValueError, match="mass"):
            create_ball_properties(mass=0.0)

    def test_negative_mass_raises(self) -> None:
        with pytest.raises(ValueError, match="mass"):
            create_ball_properties(mass=-0.04)

    def test_zero_radius_raises(self) -> None:
        with pytest.raises(ValueError, match="radius"):
            create_ball_properties(radius=0.0)

    def test_negative_radius_raises(self) -> None:
        with pytest.raises(ValueError, match="radius"):
            create_ball_properties(radius=-0.02)


class TestFallbackMath:
    """Value-assert the pure-Python fallback math unconditionally (#6940)."""

    def test_ball_fallback_area_is_pi_r_squared(self, force_fallback: None) -> None:
        radius = 0.025
        props = create_ball_properties(mass=0.05, radius=radius)
        assert isinstance(props, dict)
        assert set(props) == {
            "mass",
            "radius",
            "area",
            "drag_coefficient",
            "spin_decay_rate",
        }
        assert props["area"] == pytest.approx(math.pi * radius**2)

    def test_air_fallback_dict_shape(self, force_fallback: None) -> None:
        props = create_air_properties(density=1.2, viscosity=2e-5, temperature=300.0)
        assert isinstance(props, dict)
        assert set(props) == {"density", "viscosity", "temperature", "pressure"}
        assert props["density"] == 1.2


class TestKernelDiagnostics:
    """get_kernel_info on each forced backend (#6942)."""

    def test_kernel_info_has_required_keys(self) -> None:
        info = get_kernel_info()
        assert "rust_available" in info
        assert "backend" in info
        assert info["backend"] in ("rust", "python-fallback")

    def test_kernel_info_fallback_backend(self, force_fallback: None) -> None:
        info = get_kernel_info()
        assert info["rust_available"] is False
        assert info["backend"] == "python-fallback"
        assert "types" not in info

    def test_kernel_info_rust_backend(self, force_rust: None) -> None:
        info = get_kernel_info()
        assert info["rust_available"] is True
        assert info["backend"] == "rust"
        assert "types" in info
        assert info["types"]["IntegratorConfig"] is True
        assert info["types"]["ContactParameters"] is True


class TestMathUtilities:
    """Test clamp and lerp delegation."""

    def test_clamp_in_range(self) -> None:
        from src.shared.python.physics.rust_kernel import clamp

        assert clamp(5.0, 0.0, 10.0) == 5.0

    def test_clamp_below_min(self) -> None:
        from src.shared.python.physics.rust_kernel import clamp

        assert clamp(-3.0, 0.0, 10.0) == 0.0

    def test_clamp_above_max(self) -> None:
        from src.shared.python.physics.rust_kernel import clamp

        assert clamp(15.0, 0.0, 10.0) == 10.0

    def test_lerp_endpoints(self) -> None:
        from src.shared.python.physics.rust_kernel import lerp

        assert abs(lerp(10.0, 20.0, 0.0) - 10.0) < 1e-10
        assert abs(lerp(10.0, 20.0, 1.0) - 20.0) < 1e-10

    def test_lerp_midpoint(self) -> None:
        from src.shared.python.physics.rust_kernel import lerp

        assert abs(lerp(0.0, 100.0, 0.5) - 50.0) < 1e-10


class TestDeprecationHelpers:
    """Test mark_legacy deprecation helper."""

    def test_mark_legacy_no_crash(self) -> None:
        from src.shared.python.physics.rust_kernel import mark_legacy

        mark_legacy("test_func", "test_module")

    def test_mark_legacy_deduplication(self) -> None:
        from src.shared.python.physics.rust_kernel import mark_legacy

        mark_legacy("test_dedup", "test_module")
        mark_legacy("test_dedup", "test_module")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
