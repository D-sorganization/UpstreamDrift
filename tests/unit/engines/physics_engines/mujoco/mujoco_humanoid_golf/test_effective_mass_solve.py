"""Effective-mass regression tests for MuJoCo humanoid golf helpers."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf._kfa_effective_mass import (
    _KFAEffectiveMassMixin,
)
from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.jacobian_utils import (
    compute_effective_mass_value,
)
from src.shared.python.core.numerical_constants import EPSILON_SINGULARITY_DETECTION

pytestmark = pytest.mark.unit


def _sample_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    direction = np.array([0.3, -0.4, 0.5], dtype=float)
    direction = direction / np.linalg.norm(direction)
    jacp = np.array(
        [
            [0.4, -0.2, 0.1, 0.0],
            [0.1, 0.3, -0.5, 0.2],
            [-0.2, 0.6, 0.4, -0.3],
        ],
        dtype=float,
    )
    transform = np.array(
        [
            [2.0, 0.1, -0.2, 0.0],
            [0.0, 1.5, 0.3, -0.1],
            [0.4, -0.2, 1.7, 0.2],
            [0.1, 0.3, -0.1, 1.2],
        ],
        dtype=float,
    )
    mass_matrix = transform.T @ transform + np.eye(4) * 0.25
    return direction, jacp, mass_matrix


def _legacy_inverse_effective_mass(
    direction: np.ndarray, jacp: np.ndarray, mass_matrix: np.ndarray
) -> float:
    j_dir = direction @ jacp
    denominator = (
        j_dir @ np.linalg.inv(mass_matrix) @ j_dir.T + EPSILON_SINGULARITY_DETECTION
    )
    return float(1.0 / denominator)


def test_effective_mass_helpers_match_legacy_inverse_formula() -> None:
    """Pin current numeric output before replacing the explicit inverse."""
    direction, jacp, mass_matrix = _sample_inputs()
    expected = _legacy_inverse_effective_mass(direction, jacp, mass_matrix)

    assert compute_effective_mass_value(direction, jacp, mass_matrix) == pytest.approx(
        expected, rel=1e-12, abs=1e-12
    )
    assert _KFAEffectiveMassMixin()._compute_effective_mass_value(
        direction, jacp, mass_matrix
    ) == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_effective_mass_helpers_solve_without_forming_inverse() -> None:
    """Both duplicate call sites must avoid explicitly forming M^-1."""
    direction, jacp, mass_matrix = _sample_inputs()

    with patch("numpy.linalg.inv", side_effect=AssertionError("no inverse")):
        standalone = compute_effective_mass_value(direction, jacp, mass_matrix)
        mixin = _KFAEffectiveMassMixin()._compute_effective_mass_value(
            direction, jacp, mass_matrix
        )

    expected = _legacy_inverse_effective_mass(direction, jacp, mass_matrix)
    assert standalone == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert mixin == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_effective_mass_helper_rejects_dimension_mismatch() -> None:
    """Guard the solve boundary so bad Jacobian/M dimensions fail clearly."""
    direction, jacp, mass_matrix = _sample_inputs()

    with pytest.raises(ValueError, match="velocity dimension"):
        compute_effective_mass_value(direction, jacp[:, :3], mass_matrix)
