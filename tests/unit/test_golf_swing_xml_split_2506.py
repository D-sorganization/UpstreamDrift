"""Contract tests for issue #2506: golf_swing_models_xml.py split.

Tests run red before the split and green after.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
XML_DIR = REPO / "src/engines/physics_engines/mujoco"
LOC_BUDGET = 500  # Each split file should be under 500 LOC


def _count_lines(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


class TestGolfSwingXmlSplitStructure:
    """Split XML modules must exist after refactor."""

    @pytest.mark.unit
    def test_upper_body_xml_module_exists(self) -> None:
        assert (XML_DIR / "_golf_swing_upper_body_xml.py").exists()

    @pytest.mark.unit
    def test_full_body_xml_module_exists(self) -> None:
        assert (XML_DIR / "_golf_swing_full_body_xml.py").exists()

    @pytest.mark.unit
    def test_advanced_xml_module_exists(self) -> None:
        assert (XML_DIR / "_golf_swing_advanced_xml.py").exists()


class TestGolfSwingXmlFileSizes:
    """golf_swing_models_xml.py coordinator and split files must be <= 500 LOC."""

    @pytest.mark.unit
    def test_golf_swing_xml_split_2506_coordinator_loc(self) -> None:
        loc = _count_lines(XML_DIR / "golf_swing_models_xml.py")
        assert loc <= LOC_BUDGET, (
            f"golf_swing_models_xml.py has {loc} LOC; budget {LOC_BUDGET}"
        )

    @pytest.mark.unit
    def test_upper_body_loc(self) -> None:
        loc = _count_lines(XML_DIR / "_golf_swing_upper_body_xml.py")
        assert loc <= LOC_BUDGET, (
            f"_golf_swing_upper_body_xml.py has {loc} LOC; budget {LOC_BUDGET}"
        )

    @pytest.mark.unit
    def test_full_body_loc(self) -> None:
        loc = _count_lines(XML_DIR / "_golf_swing_full_body_xml.py")
        assert loc <= LOC_BUDGET, (
            f"_golf_swing_full_body_xml.py has {loc} LOC; budget {LOC_BUDGET}"
        )

    @pytest.mark.unit
    def test_advanced_loc(self) -> None:
        loc = _count_lines(XML_DIR / "_golf_swing_advanced_xml.py")
        assert loc <= LOC_BUDGET, (
            f"_golf_swing_advanced_xml.py has {loc} LOC; budget {LOC_BUDGET}"
        )


class TestGolfSwingXmlPublicAPI:
    """Public API must be importable from golf_swing_models_xml (backward compat)."""

    @pytest.mark.unit
    def test_upper_body_xml_importable(self) -> None:
        from src.engines.physics_engines.mujoco.golf_swing_models_xml import (
            UPPER_BODY_GOLF_SWING_XML,
        )

        assert UPPER_BODY_GOLF_SWING_XML is not None
        assert "golf_upper_body_swing" in UPPER_BODY_GOLF_SWING_XML

    @pytest.mark.unit
    def test_full_body_xml_importable(self) -> None:
        from src.engines.physics_engines.mujoco.golf_swing_models_xml import (
            FULL_BODY_GOLF_SWING_XML,
        )

        assert FULL_BODY_GOLF_SWING_XML is not None
        assert "golf_full_body_swing" in FULL_BODY_GOLF_SWING_XML

    @pytest.mark.unit
    def test_advanced_xml_importable(self) -> None:
        from src.engines.physics_engines.mujoco.golf_swing_models_xml import (
            ADVANCED_BIOMECHANICAL_GOLF_SWING_XML,
        )

        assert ADVANCED_BIOMECHANICAL_GOLF_SWING_XML is not None
        assert (
            "advanced_biomechanical_golf_swing" in ADVANCED_BIOMECHANICAL_GOLF_SWING_XML
        )

    @pytest.mark.unit
    def test_club_configs_importable(self) -> None:
        from src.engines.physics_engines.mujoco.golf_swing_models_xml import (
            CLUB_CONFIGS,
        )

        assert "driver" in CLUB_CONFIGS
        assert "iron_7" in CLUB_CONFIGS
        assert "wedge" in CLUB_CONFIGS

    @pytest.mark.unit
    def test_myo_paths_importable(self) -> None:
        from src.engines.physics_engines.mujoco.golf_swing_models_xml import (
            MYOUPPERBODY_PATH,
        )

        assert MYOUPPERBODY_PATH is not None
