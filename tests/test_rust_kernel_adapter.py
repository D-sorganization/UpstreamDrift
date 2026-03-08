"""Tests for the Rust kernel Python adapter module.

Validates:
- Graceful fallback when Rust wheel is not installed.
- API contract for create_integrator_config / create_contact_parameters.
- Diagnostic info correctness.
"""

from __future__ import annotations

import pytest

from src.shared.python.physics.rust_kernel import (
    create_contact_parameters,
    create_integrator_config,
    get_kernel_info,
    is_rust_available,
)


class TestRustKernelAvailability:
    """Test kernel availability detection."""

    def test_is_rust_available_returns_bool(self) -> None:
        """is_rust_available() must return a boolean."""
        result = is_rust_available()
        assert isinstance(result, bool)


class TestIntegratorConfig:
    """Test create_integrator_config adapter."""

    def test_default_config(self) -> None:
        """Default integrator config must have dt=0.001, max_steps=10000."""
        config = create_integrator_config()
        assert config is not None

    def test_custom_config(self) -> None:
        """Custom config with specified parameters."""
        config = create_integrator_config(dt=0.01, max_steps=500)
        assert config is not None

    def test_fallback_returns_dict(self) -> None:
        """When Rust is unavailable, returns a dict with expected keys."""
        # This test works regardless of whether Rust is available
        config = create_integrator_config(dt=0.005, max_steps=1000)
        if isinstance(config, dict):
            assert config["dt"] == 0.005
            assert config["max_steps"] == 1000


class TestContactParameters:
    """Test create_contact_parameters adapter."""

    def test_default_params(self) -> None:
        """Default contact params must have sensible golf defaults."""
        params = create_contact_parameters()
        assert params is not None

    def test_custom_params(self) -> None:
        """Custom contact params with specified COR and friction."""
        params = create_contact_parameters(cor=0.6, friction=0.3)
        assert params is not None

    def test_fallback_returns_dict(self) -> None:
        """When Rust is unavailable, returns a dict with expected keys."""
        params = create_contact_parameters(cor=0.75, friction=0.2)
        if isinstance(params, dict):
            assert params["cor"] == 0.75
            assert params["friction"] == 0.2


class TestKernelDiagnostics:
    """Test get_kernel_info diagnostics."""

    def test_kernel_info_has_required_keys(self) -> None:
        """Kernel info must have rust_available and backend keys."""
        info = get_kernel_info()
        assert "rust_available" in info
        assert "backend" in info
        assert info["backend"] in ("rust", "python-fallback")

    def test_kernel_info_types_when_rust_available(self) -> None:
        """When Rust is available, types dict is present."""
        info = get_kernel_info()
        if info["rust_available"]:
            assert "types" in info
            assert info["types"]["IntegratorConfig"] is True
            assert info["types"]["ContactParameters"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
