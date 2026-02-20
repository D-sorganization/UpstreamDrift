"""
Unit tests for club configurations module.

Tests club specifications and database.
"""

import pytest
from mujoco_humanoid_golf.club_configurations import (
    ClubDatabase,
    ClubSpecification,
    get_recommended_flex,
)


class TestClubSpecification:
    """Tests for ClubSpecification dataclass."""

    def test_club_specification_creation(self) -> None:
        """Test creating a club specification."""
        spec = ClubSpecification(
            name="Test Club",
            club_type="Driver",
            length_inches=45.0,
            head_mass_grams=200.0,
        )
        assert spec.name == "Test Club"
        assert spec.club_type == "Driver"
        assert spec.length_inches == 45.0
        assert spec.head_mass_grams == 200.0

    def test_club_specification_defaults(self) -> None:
        """Test club specification with default values."""
        spec = ClubSpecification(name="Test", club_type="Iron")
        assert spec.length_inches == 45.5  # Default
        assert spec.head_mass_grams == 200.0  # Default
        assert spec.loft_degrees == 10.0  # Default


class TestClubDatabase:
    """Tests for ClubDatabase class."""

    def test_club_database_has_clubs(self) -> None:
        """Test that club database contains clubs."""
        assert len(ClubDatabase.CLUBS) > 0

    def test_club_database_driver_exists(self) -> None:
        """Test driver club exists in database."""
        assert "driver" in ClubDatabase.CLUBS
        driver = ClubDatabase.CLUBS["driver"]
        assert isinstance(driver, ClubSpecification)
        assert driver.club_type == "Driver"

    def test_club_database_iron_exists(self) -> None:
        """Test iron club exists in database."""
        # Check for any iron club
        iron_clubs = [k for k in ClubDatabase.CLUBS if "iron" in k.lower()]
        assert len(iron_clubs) > 0

    def test_club_specification_values(self) -> None:
        """Test club specifications have reasonable values."""
        for spec in ClubDatabase.CLUBS.values():
            assert spec.length_inches > 0
            assert spec.head_mass_grams > 0
            assert spec.loft_degrees >= 0
            assert spec.loft_degrees <= 90

    def test_club_database_get_club(self) -> None:
        """Test getting a club by ID."""
        driver = ClubDatabase.get_club("driver")
        assert driver is not None
        assert driver.club_type == "Driver"

    def test_club_database_get_club_invalid(self) -> None:
        """Test getting invalid club returns None."""
        result = ClubDatabase.get_club("invalid_club")
        assert result is None

    def test_club_database_get_all_clubs(self) -> None:
        """Test getting all clubs."""
        all_clubs = ClubDatabase.get_all_clubs()
        assert isinstance(all_clubs, dict)
        assert len(all_clubs) > 0
        assert "driver" in all_clubs

    def test_club_database_get_clubs_by_type(self) -> None:
        """Test getting clubs by type."""
        drivers = ClubDatabase.get_clubs_by_type("Driver")
        assert len(drivers) > 0
        assert all(spec.club_type == "Driver" for spec in drivers)

    def test_club_database_get_club_types(self) -> None:
        """Test getting all club types."""
        types = ClubDatabase.get_club_types()
        assert isinstance(types, list)
        assert "Driver" in types

    def test_club_database_compute_total_mass(self) -> None:
        """Test computing total club mass."""
        driver = ClubDatabase.get_club("driver")
        assert driver is not None
        total_mass = ClubDatabase.compute_total_mass(driver)
        assert total_mass > 0
        assert total_mass == (
            driver.head_mass_grams + driver.shaft_mass_grams + driver.grip_mass_grams
        )

    def test_club_database_length_to_meters(self) -> None:
        """Test length conversion to meters."""
        length_m = ClubDatabase.length_to_meters(36.0)  # 36 inches
        assert abs(length_m - 0.9144) < 1e-4  # 36 * 0.0254 = 0.9144


class TestGetRecommendedFlex:
    """Tests for get_recommended_flex function."""

    def test_get_recommended_flex_ladies(self) -> None:
        """Test getting recommended flex for slow swing speed."""
        flex = get_recommended_flex(65.0)
        assert flex in ["Ladies", "Senior"]

    def test_get_recommended_flex_regular(self) -> None:
        """Test getting recommended flex for average swing speed."""
        flex = get_recommended_flex(85.0)
        assert flex == "Regular"

    def test_get_recommended_flex_stiff(self) -> None:
        """Test getting recommended flex for fast swing speed."""
        flex = get_recommended_flex(100.0)
        assert flex in ["Stiff", "X-Stiff"]

    def test_get_recommended_flex_very_slow(self) -> None:
        """Test getting recommended flex for very slow swing speed."""
        flex = get_recommended_flex(50.0)
        assert flex == "Ladies"

    def test_get_recommended_flex_very_fast(self) -> None:
        """Test getting recommended flex for very fast swing speed."""
        flex = get_recommended_flex(130.0)
        assert flex == "X-Stiff"


class TestCreateCustomClub:
    """Tests for create_custom_club class method."""

    def test_create_custom_club_basic(self) -> None:
        """Test creating a custom club with basic parameters."""
        spec = ClubDatabase.create_custom_club("Custom Driver", "Driver")
        assert isinstance(spec, ClubSpecification)
        assert spec.name == "Custom Driver"
        assert spec.club_type == "Driver"

    def test_create_custom_club_with_params(self) -> None:
        """Test creating a custom club with custom parameters."""
        spec = ClubDatabase.create_custom_club(
            "Custom Iron",
            "Iron",
            length_inches=38.0,
            head_mass_grams=250.0,
            loft_degrees=30.0,
        )
        assert spec.length_inches == 38.0
        assert spec.head_mass_grams == 250.0
        assert spec.loft_degrees == 30.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
