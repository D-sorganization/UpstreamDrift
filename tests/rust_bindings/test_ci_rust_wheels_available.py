"""CI tripwire for Rust-backed Python suites."""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.unit


def test_ci_rust_wheels_expected_makes_rust_kernel_available() -> None:
    """Wheel-installing CI lanes must fail before skipif can hide regressions."""
    if os.environ.get("CI_RUST_WHEELS_EXPECTED") != "1":
        pytest.skip("Rust wheels are only mandatory in the Rust wheel CI lane")

    from src.shared.python.physics.rust_kernel import is_rust_available

    assert is_rust_available() is True
