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


# ---------------------------------------------------------------------------
# MakeHuman Generator Tests  (See issue #979)
# ---------------------------------------------------------------------------


class TestMakeHumanParamConversion:
    """Test BodyParameters -> MakeHuman modifier conversion."""

    def test_default_params_conversion(self) -> None:
        params = BodyParameters()
        modifiers = MakeHumanMeshGenerator._convert_params_to_makehuman(params)

        assert "macrodetails/Gender" in modifiers
        assert "macrodetails/Age" in modifiers
        assert "macrodetails-universal/Muscle" in modifiers
        assert "macrodetails-universal/Weight" in modifiers
        assert "__height_scale__" in modifiers

    def test_gender_mapping(self) -> None:
        male = BodyParameters(gender_model=GenderModel.MALE)
        female = BodyParameters(gender_model=GenderModel.FEMALE)

        m = MakeHumanMeshGenerator._convert_params_to_makehuman(male)
        f = MakeHumanMeshGenerator._convert_params_to_makehuman(female)

        assert m["macrodetails/Gender"] == 1.0
        assert f["macrodetails/Gender"] == 0.0

    def test_age_normalisation(self) -> None:
        young = BodyParameters()
        young.appearance.age_years = 20.0
        old = BodyParameters()
        old.appearance.age_years = 60.0

        m_young = MakeHumanMeshGenerator._convert_params_to_makehuman(young)
        m_old = MakeHumanMeshGenerator._convert_params_to_makehuman(old)

        assert m_young["macrodetails/Age"] < m_old["macrodetails/Age"]
        assert 0.0 <= m_young["macrodetails/Age"] <= 1.0

    def test_height_scale(self) -> None:
        tall = _default_params(height_m=1.90)
        modifiers = MakeHumanMeshGenerator._convert_params_to_makehuman(tall)
        assert modifiers["__height_scale__"] > 1.0

    def test_proportion_modifiers(self) -> None:
        params = _default_params(
            shoulder_width_factor=1.2,
            hip_width_factor=0.9,
            arm_length_factor=1.1,
            leg_length_factor=1.1,
        )
        modifiers = MakeHumanMeshGenerator._convert_params_to_makehuman(params)
        assert modifiers["macrodetails-proportions/ShoulderWidth"] > 0
        assert modifiers["macrodetails-proportions/HipWidth"] < 0
        assert modifiers["macrodetails-proportions/ArmLength"] > 0
        assert modifiers["macrodetails-proportions/LegLength"] > 0


# ---------------------------------------------------------------------------
# Factory Tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# GeneratedMeshResult Tests
# ---------------------------------------------------------------------------
