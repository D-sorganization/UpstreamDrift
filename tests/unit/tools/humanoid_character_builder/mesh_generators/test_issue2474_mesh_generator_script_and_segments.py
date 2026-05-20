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


# ---------------------------------------------------------------------------
# Factory Tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# GeneratedMeshResult Tests
# ---------------------------------------------------------------------------


class TestIssue2474MeshGeneratorScriptAndSegments:
    """Issue #2474: generated MakeHuman script must not use undefined logger; segments must not overlap."""

    def test_generated_script_does_not_use_logger(self) -> None:
        """_build_mh_script must not emit 'logger' without defining it."""

        script = MakeHumanMeshGenerator._build_mh_script(
            modifiers={"face/age": 0.5},
            body_obj_path=Path("/tmp/body.obj"),
            groups_json_path=Path("/tmp/groups.json"),
        )
        # Find every reference to 'logger' that isn't a definition
        import re

        uses = re.findall(r"\blogger\b", script)
        definitions = re.findall(
            r"(?:logger\s*=|import\s+\w+\s+as\s+logger|logger\s*:)", script
        )
        undefined_uses = len(uses) - len(definitions)
        assert undefined_uses == 0, (
            f"Generated script references 'logger' {undefined_uses} time(s) without defining it. "
            "Use print() or add a logger definition to the generated script."
        )

    def test_generated_script_is_valid_python_syntax(self) -> None:
        """_build_mh_script output must be syntactically valid Python."""
        import ast

        script = MakeHumanMeshGenerator._build_mh_script(
            modifiers={"face/age": 0.3, "body/weight": 0.6},
            body_obj_path=Path("/tmp/body.obj"),
            groups_json_path=Path("/tmp/groups.json"),
        )
        try:
            ast.parse(script)
        except SyntaxError as e:
            pytest.fail(f"Generated script has syntax error: {e}")

    def test_smplx_segment_ranges_do_not_overlap(self) -> None:
        """SMPLX_SEGMENT_VERTEX_RANGES must not have overlapping vertex ranges."""
        ranges = list(SMPLXMeshGenerator.SMPLX_SEGMENT_VERTEX_RANGES.items())
        for i, (name_a, (start_a, end_a)) in enumerate(ranges):
            for name_b, (start_b, end_b) in ranges[i + 1 :]:
                overlap_start = max(start_a, start_b)
                overlap_end = min(end_a, end_b)
                assert overlap_start >= overlap_end, (
                    f"Segments '{name_a}' [{start_a}, {end_a}) and "
                    f"'{name_b}' [{start_b}, {end_b}) overlap at [{overlap_start}, {overlap_end}). "
                    "Overlapping segments contaminate mass/inertia calculations."
                )

    def test_validate_vertex_ranges_detects_overlap(self) -> None:
        """validate_vertex_ranges must return False when segments overlap."""
        # Temporarily inject an overlapping range
        original = SMPLXMeshGenerator.SMPLX_SEGMENT_VERTEX_RANGES.copy()
        try:
            SMPLXMeshGenerator.SMPLX_SEGMENT_VERTEX_RANGES = dict(original)
            SMPLXMeshGenerator.SMPLX_SEGMENT_VERTEX_RANGES["_test_overlap_a"] = (
                100,
                300,
            )
            SMPLXMeshGenerator.SMPLX_SEGMENT_VERTEX_RANGES["_test_overlap_b"] = (
                200,
                400,
            )
            result = SMPLXMeshGenerator.validate_vertex_ranges(
                SMPLXMeshGenerator.SMPLX_EXPECTED_VERTEX_COUNT
            )
            assert (
                result is False
            ), "validate_vertex_ranges must return False when segments overlap"
        finally:
            SMPLXMeshGenerator.SMPLX_SEGMENT_VERTEX_RANGES = original
