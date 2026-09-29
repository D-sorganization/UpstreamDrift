"""Tests for API architecture improvements (#1485, #1488).

Tests:
- Route registry auto-discovery and registration
- Task manager with TTL, concurrency, and lifecycle
- API versioning (routes available under /api/v1/)
- Linkage mechanisms decomposition (imports still work)
"""

from __future__ import annotations


import pytest

# ── Route Registry Tests ─────────────────────────────────────────


# ── Task Manager Tests ────────────────────────────────────────────


# ── Dict-like Compatibility Tests (#4843) ────────────────────────


# ── API Versioning Tests ──────────────────────────────────────────


# ── Linkage Decomposition Tests ──────────────────────────────────


class TestLinkageMechanismsDecomposition:
    """Tests that linkage_mechanisms decomposition preserves the public API (#1485)."""

    def test_imports_from_init(self) -> None:
        """All public symbols are importable from the package."""
        from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.linkage_mechanisms import (
            LINKAGE_CATALOG,
            generate_chebyshev_linkage_xml,
            generate_delta_robot_xml,
            generate_five_bar_parallel_xml,
            generate_four_bar_linkage_xml,
            generate_geneva_mechanism_xml,
            generate_oldham_coupling_xml,
            generate_pantograph_xml,
            generate_peaucellier_linkage_xml,
            generate_scotch_yoke_xml,
            generate_slider_crank_xml,
            generate_stewart_platform_xml,
            generate_watt_linkage_xml,
        )

        # All should be callable
        assert callable(generate_four_bar_linkage_xml)
        assert callable(generate_slider_crank_xml)
        assert callable(generate_scotch_yoke_xml)
        assert callable(generate_geneva_mechanism_xml)
        assert callable(generate_oldham_coupling_xml)
        assert callable(generate_peaucellier_linkage_xml)
        assert callable(generate_chebyshev_linkage_xml)
        assert callable(generate_watt_linkage_xml)
        assert callable(generate_pantograph_xml)
        assert callable(generate_delta_robot_xml)
        assert callable(generate_five_bar_parallel_xml)
        assert callable(generate_stewart_platform_xml)
        # Catalog should be a dict
        assert isinstance(LINKAGE_CATALOG, dict)

    def test_catalog_has_all_mechanisms(self) -> None:
        """LINKAGE_CATALOG contains all expected mechanisms."""
        from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.linkage_mechanisms import (
            LINKAGE_CATALOG,
        )

        assert len(LINKAGE_CATALOG) == 16
        for name, entry in LINKAGE_CATALOG.items():
            assert "xml" in entry, f"Missing 'xml' in catalog entry: {name}"
            assert "actuators" in entry, f"Missing 'actuators' in catalog entry: {name}"
            assert "category" in entry, f"Missing 'category' in catalog entry: {name}"
            assert "description" in entry, (
                f"Missing 'description' in catalog entry: {name}"
            )

    def test_four_bar_generates_valid_xml(self) -> None:
        """Four-bar linkage XML contains expected MuJoCo elements."""
        from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.linkage_mechanisms import (
            generate_four_bar_linkage_xml,
        )

        xml = generate_four_bar_linkage_xml()
        assert "<mujoco" in xml
        assert "crank_joint" in xml
        assert "actuator" in xml

    def test_slider_crank_generates_valid_xml(self) -> None:
        """Slider-crank XML contains expected elements."""
        from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.linkage_mechanisms import (
            generate_slider_crank_xml,
        )

        xml = generate_slider_crank_xml(orientation="horizontal")
        assert "<mujoco" in xml
        assert "slider_crank" in xml

    def test_init_has_no_logic(self) -> None:
        """__init__.py contains only imports and __all__, no function/class definitions."""
        from pathlib import Path

        init_path = Path(
            "src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/"
            "linkage_mechanisms/__init__.py"
        )
        content = init_path.read_text(encoding="utf-8")
        # Should not contain any function or class definitions
        lines = content.split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("def ", "class ")):
                pytest.fail(
                    f"__init__.py should contain only imports, found: {stripped}"
                )

    def test_submodules_exist(self) -> None:
        """All decomposed sub-modules are importable."""
        import importlib

        submodules = [
            "four_bar",
            "slider_mechanisms",
            "special_mechanisms",
            "straight_line",
            "parallel_mechanisms",
            "catalog",
        ]
        base = "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.linkage_mechanisms"
        for name in submodules:
            mod = importlib.import_module(f"{base}.{name}")
            assert mod is not None, f"Failed to import {name}"


# ── OpenAPI Enhancement Tests ─────────────────────────────────────
