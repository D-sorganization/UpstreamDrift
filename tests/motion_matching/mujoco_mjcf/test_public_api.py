"""Tests for the public API of the MuJoCo motion-matching module.

Verifies that all exported functions and classes are accessible via the
canonical import paths defined in __init__.py, per issue #4247 and the
cross-engine parity spec (CROSS_ENGINE_PARITY_SPEC.md §2.2).
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.requires_mujoco, pytest.mark.unit]


def test_synthesize_target_from_coefficients_is_exported() -> None:
    """The synthesize_target_from_coefficients function is public."""
    from src.engines.physics_engines.mujoco.python import motion_matching
    from src.engines.physics_engines.mujoco.python.motion_matching import (
        synthesize_target_from_coefficients,
    )
    from src.engines.physics_engines.mujoco.python.motion_matching.synthesize import (
        synthesize_target_from_coefficients as synthesize_impl,
    )

    assert "synthesize_target_from_coefficients" in motion_matching.__all__
    assert synthesize_target_from_coefficients is synthesize_impl


def test_synthesize_target_from_coefficients_in_all() -> None:
    """The function appears in __all__ for proper public API exposure."""
    from src.engines.physics_engines.mujoco.python import motion_matching

    assert "synthesize_target_from_coefficients" in motion_matching.__all__
