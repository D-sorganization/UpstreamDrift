"""Unit tests for linkage mechanisms catalog."""

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.linkage_mechanisms.catalog import (
    LINKAGE_CATALOG,
)


def test_linkage_catalog_structure():
    """Test the structure of the LINKAGE_CATALOG dictionary."""
    assert isinstance(LINKAGE_CATALOG, dict)
    assert len(LINKAGE_CATALOG) > 0

    for name, metadata in LINKAGE_CATALOG.items():
        assert isinstance(name, str)
        assert isinstance(metadata, dict)

        # Check required keys
        assert "xml" in metadata
        assert "actuators" in metadata
        assert "category" in metadata
        assert "description" in metadata

        # Check types
        assert isinstance(metadata["xml"], str)
        assert isinstance(metadata["actuators"], list)
        assert isinstance(metadata["category"], str)
        assert isinstance(metadata["description"], str)

        # Verify XML is not empty
        assert len(metadata["xml"]) > 0
        assert "<mujoco" in metadata["xml"]


def test_catalog_categories():
    """Test that categories match expected values."""
    categories = {meta["category"] for meta in LINKAGE_CATALOG.values()}
    assert "Four-Bar Linkages" in categories
    assert "Slider Mechanisms" in categories
    assert "Straight-Line Mechanisms" in categories
    assert "Parallel Mechanisms" in categories
