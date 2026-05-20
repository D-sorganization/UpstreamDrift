"""Contract tests for #2486: mesh_generator.py split.

Tests run red before the split and green after.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
GENERATORS_DIR = REPO / "src/shared/python/humanoid_character_builder/generators"
LOC_BUDGET_TYPES = 150
LOC_BUDGET_PRIMITIVES = 250
LOC_BUDGET_MAKEHUMAN = 750
LOC_BUDGET_SMPLX = 750
LOC_BUDGET_COORDINATOR = 200


def _count_lines(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


class TestMeshGeneratorSplitStructure:
    """Split modules must exist after refactor."""

    @pytest.mark.unit
    def test_types_module_exists(self) -> None:
        assert (GENERATORS_DIR / "_mesh_types.py").exists()

    @pytest.mark.unit
    def test_primitives_module_exists(self) -> None:
        assert (GENERATORS_DIR / "_mesh_primitives.py").exists()

    @pytest.mark.unit
    def test_smplx_module_exists(self) -> None:
        assert (GENERATORS_DIR / "_mesh_smplx.py").exists()


class TestMeshGeneratorFileSizes:
    """Each file must be within LOC budget after split."""

    @pytest.mark.unit
    def test_mesh_generator_split_2486_coordinator_loc(self) -> None:
        loc = _count_lines(GENERATORS_DIR / "mesh_generator.py")
        assert (
            loc <= LOC_BUDGET_COORDINATOR
        ), f"mesh_generator.py has {loc} LOC; budget {LOC_BUDGET_COORDINATOR}"

    @pytest.mark.unit
    def test_types_loc(self) -> None:
        loc = _count_lines(GENERATORS_DIR / "_mesh_types.py")
        assert (
            loc <= LOC_BUDGET_TYPES
        ), f"_mesh_types.py has {loc} LOC; budget {LOC_BUDGET_TYPES}"

    @pytest.mark.unit
    def test_primitives_loc(self) -> None:
        loc = _count_lines(GENERATORS_DIR / "_mesh_primitives.py")
        assert (
            loc <= LOC_BUDGET_PRIMITIVES
        ), f"_mesh_primitives.py has {loc} LOC; budget {LOC_BUDGET_PRIMITIVES}"

    @pytest.mark.unit
    def test_smplx_loc(self) -> None:
        loc = _count_lines(GENERATORS_DIR / "_mesh_smplx.py")
        assert (
            loc <= LOC_BUDGET_SMPLX
        ), f"_mesh_smplx.py has {loc} LOC; budget {LOC_BUDGET_SMPLX}"


_smplx_available = importlib.util.find_spec("smplx") is not None


class TestMeshGeneratorPublicAPI:
    """Public API must remain importable from mesh_generator (backward compat)."""

    @pytest.mark.unit
    def test_import_mesh_generator_backend(self) -> None:
        from humanoid_character_builder.generators.mesh_generator import (
            MeshGeneratorBackend,
        )

        assert MeshGeneratorBackend is not None

    @pytest.mark.unit
    def test_import_generated_mesh_result(self) -> None:
        from humanoid_character_builder.generators.mesh_generator import (
            GeneratedMeshResult,
        )

        assert GeneratedMeshResult is not None

    @pytest.mark.unit
    def test_import_mesh_generator_interface(self) -> None:
        from humanoid_character_builder.generators.mesh_generator import (
            MeshGeneratorInterface,
        )

        assert MeshGeneratorInterface is not None

    @pytest.mark.unit
    def test_import_primitive_mesh_generator(self) -> None:
        from humanoid_character_builder.generators.mesh_generator import (
            PrimitiveMeshGenerator,
        )

        assert PrimitiveMeshGenerator is not None

    @pytest.mark.unit
    def test_import_mesh_generator(self) -> None:
        from humanoid_character_builder.generators.mesh_generator import (
            MeshGenerator,
        )

        assert MeshGenerator is not None
