"""Unit tests for equipment module.

TEST-001: Added test coverage for equipment.py (previously 0% coverage).
"""

import pytest

from src.shared.python.equipment import CLUB_CONFIGS, get_club_config


class TestEquipmentModule:
    """Test cases for equipment module."""

    def test_club_configs_structure(self) -> None:
        """Test that CLUB_CONFIGS has expected structure."""
        assert isinstance(CLUB_CONFIGS, dict)
        assert len(CLUB_CONFIGS) == 3
        assert "driver" in CLUB_CONFIGS
        assert "iron_7" in CLUB_CONFIGS
        assert "wedge" in CLUB_CONFIGS

    def test_driver_config(self) -> None:
        """Test driver configuration has all expected fields."""
        driver = CLUB_CONFIGS["driver"]

        # Check all required fields exist
        required_fields = [
            "grip_length",
            "grip_radius",
            "grip_mass",
            "shaft_length",
            "shaft_radius",
            "shaft_mass",
            "head_mass",
            "head_size",
            "total_length",
            "club_loft",
            "flex_stiffness",
        ]

        for field in required_fields:
            assert field in driver, f"Missing field: {field}"

        # Check types
        assert isinstance(driver["grip_length"], float)
        assert isinstance(driver["grip_mass"], float)
        assert isinstance(driver["head_size"], list)
        assert len(driver["head_size"]) == 3
        assert isinstance(driver["flex_stiffness"], list)
        assert len(driver["flex_stiffness"]) == 3

    def test_iron_7_config(self) -> None:
        """Test 7-iron configuration."""
        iron = CLUB_CONFIGS["iron_7"]

        # Basic checks
        assert isinstance(iron["grip_length"], float)
        assert iron["grip_length"] > 0
        assert isinstance(iron["shaft_length"], float)
        assert iron["shaft_length"] > 0
        assert isinstance(iron["head_mass"], float)
        assert iron["head_mass"] > 0
        assert isinstance(iron["total_length"], float)
        assert iron["total_length"] > 0

        # Verify loft is higher than driver (7-iron has more loft)
        assert isinstance(iron["club_loft"], float)
        driver_loft = CLUB_CONFIGS["driver"]["club_loft"]
        assert isinstance(driver_loft, float)
        assert iron["club_loft"] > driver_loft

    def test_wedge_config(self) -> None:
        """Test wedge configuration."""
        wedge = CLUB_CONFIGS["wedge"]

        # Basic checks
        assert isinstance(wedge["grip_length"], float)
        assert wedge["grip_length"] > 0
        assert isinstance(wedge["shaft_length"], float)
        assert wedge["shaft_length"] > 0
        assert isinstance(wedge["head_mass"], float)
        assert wedge["head_mass"] > 0

        # Verify wedge has highest loft
        assert isinstance(wedge["club_loft"], float)
        iron_loft = CLUB_CONFIGS["iron_7"]["club_loft"]
        driver_loft = CLUB_CONFIGS["driver"]["club_loft"]
        assert isinstance(iron_loft, float)
        assert isinstance(driver_loft, float)
        assert wedge["club_loft"] > iron_loft
        assert wedge["club_loft"] > driver_loft

    def test_get_club_config_driver(self) -> None:
        """Test get_club_config returns driver config correctly."""
        config = get_club_config("driver")
        assert config == CLUB_CONFIGS["driver"]
        assert config["total_length"] == 1.16

    def test_get_club_config_iron_7(self) -> None:
        """Test get_club_config returns 7-iron config correctly."""
        config = get_club_config("iron_7")
        assert config == CLUB_CONFIGS["iron_7"]
        assert config["total_length"] == 0.95

    def test_get_club_config_wedge(self) -> None:
        """Test get_club_config returns wedge config correctly."""
        config = get_club_config("wedge")
        assert config == CLUB_CONFIGS["wedge"]
        assert config["total_length"] == 0.90

    def test_get_club_config_invalid_type(self) -> None:
        """Test get_club_config raises ValueError for invalid club type."""
        with pytest.raises(ValueError) as exc_info:
            get_club_config("putter")

        error_msg = str(exc_info.value)
        assert "Invalid club_type" in error_msg
        assert "putter" in error_msg
        assert "driver" in error_msg
        assert "iron_7" in error_msg
        assert "wedge" in error_msg

    def test_get_club_config_empty_string(self) -> None:
        """Test get_club_config raises ValueError for empty string."""
        with pytest.raises(ValueError) as exc_info:
            get_club_config("")

        assert "Invalid club_type" in str(exc_info.value)

    def test_realistic_values(self) -> None:
        """Test that equipment values are within realistic ranges."""
        for club_type, config in CLUB_CONFIGS.items():
            # Total length should be reasonable (0.5m to 1.5m)
            assert isinstance(config["total_length"], float)
            assert (
                0.5 <= config["total_length"] <= 1.5
            ), f"{club_type} length {config['total_length']} outside realistic range"

            # Head mass should be reasonable (100g to 500g)
            assert isinstance(config["head_mass"], float)
            assert (
                0.1 <= config["head_mass"] <= 0.5
            ), f"{club_type} head mass {config['head_mass']} outside realistic range"

            # Club loft should be in degrees converted to radians (0 to 90 degrees = 0 to 1.57 rad)
            assert isinstance(config["club_loft"], float)
            assert (
                0 <= config["club_loft"] <= 1.6
            ), f"{club_type} loft {config['club_loft']} outside realistic range"

    def test_club_ordering_by_length(self) -> None:
        """Test that clubs follow expected length ordering: driver > 7-iron > wedge."""
        driver_length = CLUB_CONFIGS["driver"]["total_length"]
        iron_length = CLUB_CONFIGS["iron_7"]["total_length"]
        wedge_length = CLUB_CONFIGS["wedge"]["total_length"]

        assert isinstance(driver_length, float)
        assert isinstance(iron_length, float)
        assert isinstance(wedge_length, float)

        assert driver_length > iron_length > wedge_length
