"""CI tripwire for Rust-backed Python suites."""

from __future__ import annotations

import importlib
import os

import pytest

pytestmark = pytest.mark.unit

# Python extension modules produced by the maturin wheel build. Mirrors
# ``scripts/ci/import_built_rust_wheels.py`` and the rust-wheel-parity CI lane.
RUST_WHEEL_MODULES: tuple[str, ...] = (
    "upstream_physics",
    "upstream_mocap_preproc",
    "upstream_mocap_io",
    "upstream_muscle",
    "upstream_motion_matching",
    "ai_backend",
)


def test_ci_rust_wheels_expected_makes_rust_kernel_available() -> None:
    """Wheel-installing CI lanes must fail before skipif can hide regressions."""
    if os.environ.get("CI_RUST_WHEELS_EXPECTED") != "1":
        pytest.skip("Rust wheels are only mandatory in the Rust wheel CI lane")

    from src.shared.python.physics.rust_kernel import is_rust_available

    assert is_rust_available() is True


@pytest.mark.parametrize("module_name", RUST_WHEEL_MODULES)
def test_ci_rust_wheels_expected_imports_every_wheel(module_name: str) -> None:
    """Every PyO3 wheel must be importable when wheels are expected.

    A missing wheel here means the maturin build or install regressed; the
    lane must fail rather than let parity tests skip silently (issue #7601).
    """
    if os.environ.get("CI_RUST_WHEELS_EXPECTED") != "1":
        pytest.skip("Rust wheels are only mandatory in the Rust wheel CI lane")

    importlib.import_module(module_name)
