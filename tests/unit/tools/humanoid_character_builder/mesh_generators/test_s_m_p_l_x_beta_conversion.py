"""
Unit tests for SMPL-X and MakeHuman mesh generators.

Tests use mocked external dependencies (smplx, trimesh, subprocess) so that
the full pipeline logic can be validated without installing heavy optional
packages.

See issues #979 (MakeHuman) and #980 (SMPL-X).
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from humanoid_character_builder.core.body_parameters import (
    BodyParameters,
    GenderModel,
)
from humanoid_character_builder.generators.mesh_generator import (
    GeneratedMeshResult,
    MakeHumanMeshGenerator,
    MeshGenerator,
    MeshGeneratorBackend,
    MeshGeneratorInterface,
    SMPLXMeshGenerator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_params(**overrides: Any) -> BodyParameters:
    """Create default BodyParameters with optional overrides."""
    kwargs: dict[str, Any] = {
        "height_m": 1.80,
        "mass_kg": 80.0,
    }
    kwargs.update(overrides)
    return BodyParameters(**kwargs)


# ---------------------------------------------------------------------------
# SMPL-X Generator Tests  (See issue #980)
# ---------------------------------------------------------------------------


class TestSMPLXBetaConversion:
    """Test BodyParameters -> SMPL-X beta parameter conversion."""

    def test_default_params_produce_near_zero_betas(self) -> None:
        params = BodyParameters()  # 1.75 m, 75 kg, average
        betas = SMPLXMeshGenerator._convert_params_to_betas(params)
        assert betas.shape == (SMPLXMeshGenerator.NUM_BETAS,)
        # Default body close to SMPL-X mean -> betas should be small
        assert np.abs(betas).max() < 5.0

    def test_tall_heavy_person(self) -> None:
        params = _default_params(height_m=2.00, mass_kg=110.0)
        betas = SMPLXMeshGenerator._convert_params_to_betas(params)
        # beta[0] should be positive (tall)
        assert betas[0] > 0
        # beta[1] should be positive (high BMI)
        assert betas[1] > 0

    def test_short_light_person(self) -> None:
        params = _default_params(height_m=1.50, mass_kg=45.0)
        betas = SMPLXMeshGenerator._convert_params_to_betas(params)
        assert betas[0] < 0  # shorter than mean
        # BMI = 45 / 1.50^2 = 20.0, below mean of 22 -> negative beta
        assert betas[1] < 0  # lower BMI

    def test_proportion_factors_map(self) -> None:
        params = _default_params(
            shoulder_width_factor=1.2,
            hip_width_factor=0.9,
            arm_length_factor=1.1,
            leg_length_factor=1.1,
            torso_length_factor=1.05,
        )
        betas = SMPLXMeshGenerator._convert_params_to_betas(params)
        assert betas[2] > 0  # wider shoulders
        assert betas[3] < 0  # narrower hips
        assert betas[4] > 0  # longer arms
        assert betas[5] > 0  # longer legs
        assert betas[6] > 0  # longer torso

    def test_muscularity_mapping(self) -> None:
        lean = _default_params(muscularity=0.1)
        buff = _default_params(muscularity=0.9)
        b_lean = SMPLXMeshGenerator._convert_params_to_betas(lean)
        b_buff = SMPLXMeshGenerator._convert_params_to_betas(buff)
        assert b_buff[7] > b_lean[7]


# ---------------------------------------------------------------------------
# MakeHuman Generator Tests  (See issue #979)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Factory Tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# GeneratedMeshResult Tests
# ---------------------------------------------------------------------------
