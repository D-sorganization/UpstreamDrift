"""Quick test script to verify linkage mechanisms are properly configured."""

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Fix path to include local python package
CURRENT_DIR = Path(__file__).resolve().parent
PYTHON_DIR = CURRENT_DIR / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

# Mock mujoco to prevent DLL errors on systems where it's not installed/working
# This is necessary because importing mujoco_golf_pendulum triggers imports of modules
# that depend on mujoco, even if we only want linkage_mechanisms (which uses numpy).
if "mujoco" not in sys.modules:
    sys.modules["mujoco"] = MagicMock()

logger = logging.getLogger(__name__)


def get_linkage_mechanisms():
    """Import and return the linkage_mechanisms module."""
    try:
        from mujoco_golf_pendulum import linkage_mechanisms

        return linkage_mechanisms
    except ImportError as e:
        pytest.fail(f"Failed to import mechanisms: {e}")


def test_catalog_structure():
    """Test that the catalog is properly structured."""
    lm = get_linkage_mechanisms()
    catalog = lm.LINKAGE_CATALOG

    assert len(catalog) > 0, "Catalog is empty"

    for _, config in catalog.items():
        assert "category" in config
        assert "actuators" in config
        assert "xml" in config
        assert len(config["xml"]) > 0


def test_xml_generation():
    """Test XML generation for each mechanism type."""
    lm = get_linkage_mechanisms()

    test_cases = [
        ("Four-bar linkage", lm.generate_four_bar_linkage_xml),
        ("Slider-crank", lm.generate_slider_crank_xml),
        ("Scotch yoke", lm.generate_scotch_yoke_xml),
        ("Geneva mechanism", lm.generate_geneva_mechanism_xml),
        ("Peaucellier linkage", lm.generate_peaucellier_linkage_xml),
        ("Chebyshev linkage", lm.generate_chebyshev_linkage_xml),
        ("Pantograph", lm.generate_pantograph_xml),
        ("Delta robot", lm.generate_delta_robot_xml),
        ("5-bar parallel", lm.generate_five_bar_parallel_xml),
        ("Stewart platform", lm.generate_stewart_platform_xml),
        ("Watt linkage", lm.generate_watt_linkage_xml),
        ("Oldham coupling", lm.generate_oldham_coupling_xml),
    ]

    for name, generator in test_cases:
        try:
            xml = generator()
            assert "<mujoco" in xml, f"{name}: Missing mujoco tag"
            assert "<worldbody>" in xml, f"{name}: Missing worldbody"
            assert "</mujoco>" in xml, f"{name}: Missing closing tag"
        except Exception as e:
            pytest.fail(f"{name} generation failed: {e}")


if __name__ == "__main__":
    # Allow running as script
    logging.basicConfig(level=logging.INFO)
    test_catalog_structure()
    test_xml_generation()
    logger.info("All tests passed!")
