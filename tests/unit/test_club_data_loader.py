"""Unit tests for src/shared/python/club_data/loader.py.

Tests cover ClubSpecification, SwingMetrics, ProPlayerData, ClubDataLoader,
and the module-level convenience functions load_club_data and
load_pro_player_data.  All tests are headless-safe (no display server needed).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# ClubSpecification
# ---------------------------------------------------------------------------


class TestClubSpecification:
    """Tests for the ClubSpecification dataclass."""

    def test_minimal_instantiation(self) -> None:
        """ClubSpecification can be created with just name and club_type."""
        from src.shared.python.club_data.loader import ClubSpecification

        club = ClubSpecification(name="7-Iron", club_type="Iron")
        assert club.name == "7-Iron"
        assert club.club_type == "Iron"

    def test_derived_length_meters(self) -> None:
        """length_meters is computed from length_inches in __post_init__."""
        from src.shared.python.club_data.loader import ClubSpecification

        club = ClubSpecification(name="Driver", club_type="Driver", length_inches=45.5)
        assert abs(club.length_meters - 45.5 * 0.0254) < 1e-9

    def test_derived_head_mass_kg(self) -> None:
        """head_mass_kg is computed from head_mass_grams in __post_init__."""
        from src.shared.python.club_data.loader import ClubSpecification

        club = ClubSpecification(
            name="Driver", club_type="Driver", head_mass_grams=200.0
        )
        assert abs(club.head_mass_kg - 0.2) < 1e-9

    def test_total_mass_grams_property(self) -> None:
        """total_mass_grams sums head + shaft + grip masses."""
        from src.shared.python.club_data.loader import ClubSpecification

        club = ClubSpecification(
            name="Driver",
            club_type="Driver",
            head_mass_grams=200.0,
            shaft_mass_grams=65.0,
            grip_mass_grams=50.0,
        )
        assert club.total_mass_grams == pytest.approx(315.0)

    def test_total_mass_kg_property(self) -> None:
        """total_mass_kg = total_mass_grams / 1000."""
        from src.shared.python.club_data.loader import ClubSpecification

        club = ClubSpecification(
            name="Driver",
            club_type="Driver",
            head_mass_grams=200.0,
            shaft_mass_grams=65.0,
            grip_mass_grams=50.0,
        )
        assert abs(club.total_mass_kg - 0.315) < 1e-9

    def test_to_dict_contains_required_keys(self) -> None:
        """to_dict returns a dict with expected keys."""
        from src.shared.python.club_data.loader import ClubSpecification

        club = ClubSpecification(name="5-Iron", club_type="Iron")
        d = club.to_dict()
        for key in [
            "name",
            "club_type",
            "length_inches",
            "head_mass_grams",
            "loft_degrees",
        ]:
            assert key in d

    def test_from_dict_round_trip(self) -> None:
        """from_dict(to_dict()) preserves name and club_type."""
        from src.shared.python.club_data.loader import ClubSpecification

        original = ClubSpecification(
            name="Driver", club_type="Driver", loft_degrees=9.5
        )
        d = original.to_dict()
        restored = ClubSpecification.from_dict(d)
        assert restored.name == original.name
        assert restored.club_type == original.club_type

    def test_from_dict_with_defaults(self) -> None:
        """from_dict fills missing fields with defaults."""
        from src.shared.python.club_data.loader import ClubSpecification

        club = ClubSpecification.from_dict({"name": "Test Club", "club_type": "Iron"})
        assert club.name == "Test Club"
        assert club.loft_degrees is not None  # default filled

    def test_number_field_optional(self) -> None:
        """number field defaults to None and can be set."""
        from src.shared.python.club_data.loader import ClubSpecification

        club = ClubSpecification(name="7-Iron", club_type="Iron")
        assert club.number is None

        club_with_num = ClubSpecification(name="7-Iron", club_type="Iron", number="7")
        assert club_with_num.number == "7"


# ---------------------------------------------------------------------------
# SwingMetrics
# ---------------------------------------------------------------------------


class TestSwingMetrics:
    """Tests for the SwingMetrics dataclass."""

    def test_default_instantiation(self) -> None:
        """SwingMetrics with defaults creates valid object with zeros."""
        from src.shared.python.club_data.loader import SwingMetrics

        m = SwingMetrics()
        assert m.club_head_speed_mph == 0.0
        assert m.spin_rate_rpm == 0.0

    def test_mph_to_ms_conversion(self) -> None:
        """club_head_speed_ms is auto-computed from mph in __post_init__."""
        from src.shared.python.club_data.loader import SwingMetrics

        m = SwingMetrics(club_head_speed_mph=100.0)
        assert abs(m.club_head_speed_ms - 100.0 * 0.44704) < 1e-6

    def test_ball_speed_mph_to_ms(self) -> None:
        """ball_speed_ms is auto-computed from ball_speed_mph."""
        from src.shared.python.club_data.loader import SwingMetrics

        m = SwingMetrics(ball_speed_mph=150.0)
        assert abs(m.ball_speed_ms - 150.0 * 0.44704) < 1e-6

    def test_carry_distance_yards_to_meters(self) -> None:
        """carry_distance_meters is auto-computed from yards."""
        from src.shared.python.club_data.loader import SwingMetrics

        m = SwingMetrics(carry_distance_yards=300.0)
        assert abs(m.carry_distance_meters - 300.0 * 0.9144) < 1e-6

    def test_total_distance_yards_to_meters(self) -> None:
        """total_distance_meters is auto-computed from yards."""
        from src.shared.python.club_data.loader import SwingMetrics

        m = SwingMetrics(total_distance_yards=320.0)
        assert abs(m.total_distance_meters - 320.0 * 0.9144) < 1e-6

    def test_smash_factor_computed(self) -> None:
        """smash_factor = ball_speed / club_head_speed when both nonzero."""
        from src.shared.python.club_data.loader import SwingMetrics

        m = SwingMetrics(club_head_speed_mph=100.0, ball_speed_mph=150.0)
        assert abs(m.smash_factor - 1.5) < 1e-9

    def test_club_data_loader_to_dict_returns_dict(self) -> None:
        """to_dict returns a dict with all expected keys."""
        from src.shared.python.club_data.loader import SwingMetrics

        m = SwingMetrics()
        d = m.to_dict()
        assert isinstance(d, dict)
        assert "club_head_speed_mph" in d
        assert "smash_factor" in d

    def test_explicit_ms_not_overwritten(self) -> None:
        """Explicitly set ms value is not overwritten if mph > 0."""
        from src.shared.python.club_data.loader import SwingMetrics

        # If club_head_speed_ms already set, __post_init__ won't overwrite it
        m = SwingMetrics(club_head_speed_mph=100.0, club_head_speed_ms=50.0)
        # When ms is already set (nonzero), no overwrite occurs
        assert m.club_head_speed_ms == 50.0


# ---------------------------------------------------------------------------
# ProPlayerData
# ---------------------------------------------------------------------------


class TestProPlayerData:
    """Tests for ProPlayerData dataclass."""

    def test_minimal_instantiation(self) -> None:
        """ProPlayerData with just player_name is valid."""
        from src.shared.python.club_data.loader import ProPlayerData

        p = ProPlayerData(player_name="Tiger Woods")
        assert p.player_name == "Tiger Woods"
        assert p.skill_level == "Professional"
        assert p.handedness == "Right"

    def test_has_trajectory_data_empty(self) -> None:
        """has_trajectory_data returns False when no time series."""
        from src.shared.python.club_data.loader import ProPlayerData

        p = ProPlayerData(player_name="Player")
        assert not p.has_trajectory_data()

    def test_get_position_at_time_returns_none_without_data(self) -> None:
        """get_position_at_time returns None when no trajectory loaded."""
        from src.shared.python.club_data.loader import ProPlayerData

        p = ProPlayerData(player_name="Player")
        pos = p.get_position_at_time(0.5)
        assert pos is None


# ---------------------------------------------------------------------------
# ClubDataLoader
# ---------------------------------------------------------------------------


class TestClubDataLoader:
    """Tests for ClubDataLoader class."""

    def test_club_data_loader_instantiation(self) -> None:
        """ClubDataLoader instantiates without error."""
        from src.shared.python.club_data.loader import ClubDataLoader

        loader = ClubDataLoader()
        assert loader is not None

    def test_get_all_clubs_empty_on_init(self) -> None:
        """get_all_clubs returns empty list before loading."""
        from src.shared.python.club_data.loader import ClubDataLoader

        loader = ClubDataLoader()
        assert loader.get_all_clubs() == []

    def test_get_all_players_empty_on_init(self) -> None:
        """get_all_players returns empty list before loading."""
        from src.shared.python.club_data.loader import ClubDataLoader

        loader = ClubDataLoader()
        assert loader.get_all_players() == []

    def test_get_club_returns_none_when_not_found(self) -> None:
        """get_club returns None for unknown club name."""
        from src.shared.python.club_data.loader import ClubDataLoader

        loader = ClubDataLoader()
        assert loader.get_club("nonexistent club") is None

    def test_load_clubs_from_json_list_format(self, tmp_path: Path) -> None:
        """load_clubs_from_json parses a JSON list of club dicts."""
        from src.shared.python.club_data.loader import ClubDataLoader

        data = [
            {"name": "Driver", "club_type": "Driver", "loft_degrees": 10.5},
            {"name": "7-Iron", "club_type": "Iron", "loft_degrees": 34.0},
        ]
        json_file = tmp_path / "clubs.json"
        json_file.write_text(json.dumps(data))

        loader = ClubDataLoader()
        clubs = loader.load_clubs_from_json(json_file)
        assert len(clubs) == 2
        assert clubs[0].name == "Driver"
        assert clubs[1].name == "7-Iron"

    def test_load_clubs_from_json_dict_format(self, tmp_path: Path) -> None:
        """load_clubs_from_json parses a JSON dict keyed by club identifier."""
        from src.shared.python.club_data.loader import ClubDataLoader

        data = {
            "driver": {"name": "Driver", "club_type": "Driver"},
            "iron7": {"name": "7-Iron", "club_type": "Iron"},
        }
        json_file = tmp_path / "clubs_dict.json"
        json_file.write_text(json.dumps(data))

        loader = ClubDataLoader()
        clubs = loader.load_clubs_from_json(json_file)
        assert len(clubs) == 2

    def test_load_clubs_from_json_caches_by_name(self, tmp_path: Path) -> None:
        """After load_clubs_from_json, get_club returns loaded club."""
        from src.shared.python.club_data.loader import ClubDataLoader

        data = [{"name": "Putter", "club_type": "Putter"}]
        json_file = tmp_path / "clubs.json"
        json_file.write_text(json.dumps(data))

        loader = ClubDataLoader()
        loader.load_clubs_from_json(json_file)
        club = loader.get_club("Putter")
        assert club is not None
        assert club.club_type == "Putter"

    def test_load_clubs_from_json_file_not_found(self) -> None:
        """load_clubs_from_json raises FileNotFoundError for missing file."""
        from src.shared.python.club_data.loader import ClubDataLoader

        loader = ClubDataLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_clubs_from_json("/nonexistent/path/clubs.json")

    def test_load_clubs_from_excel_requires_pandas(self) -> None:
        """load_clubs_from_excel raises ImportError when pandas unavailable."""
        import unittest.mock as mock

        from src.shared.python.club_data.loader import ClubDataLoader

        loader = ClubDataLoader()
        with (
            mock.patch("src.shared.python.club_data.loader.PANDAS_AVAILABLE", False),
            pytest.raises(ImportError, match="pandas"),
        ):
            loader.load_clubs_from_excel("/some/file.xlsx")

    def test_get_player_returns_none_when_not_found(self) -> None:
        """get_player returns None for unknown player name."""
        from src.shared.python.club_data.loader import ClubDataLoader

        loader = ClubDataLoader()
        assert loader.get_player("Unknown Player") is None


# ---------------------------------------------------------------------------
# load_club_data convenience function
# ---------------------------------------------------------------------------


class TestLoadClubData:
    """Tests for the module-level load_club_data convenience function."""

    def test_load_from_json_list(self, tmp_path: Path) -> None:
        """load_club_data reads a JSON list of club specs."""
        from src.shared.python.club_data.loader import load_club_data

        data = [{"name": "Driver", "club_type": "Driver", "loft_degrees": 9.5}]
        json_file = tmp_path / "test_clubs.json"
        json_file.write_text(json.dumps(data))

        clubs = load_club_data(json_file)
        assert len(clubs) == 1
        assert clubs[0].name == "Driver"

    def test_club_data_loader_unsupported_format_raises(self, tmp_path: Path) -> None:
        """load_club_data raises ValueError for unsupported file extension."""
        from src.shared.python.club_data.loader import load_club_data

        csv_file = tmp_path / "clubs.csv"
        csv_file.write_text("name,club_type\nDriver,Driver\n")
        with pytest.raises(ValueError, match="Unsupported"):
            load_club_data(csv_file)

    def test_file_not_found_raises(self) -> None:
        """load_club_data raises FileNotFoundError for missing file."""
        from src.shared.python.club_data.loader import load_club_data

        with pytest.raises(FileNotFoundError):
            load_club_data("/nonexistent/clubs.json")


# ---------------------------------------------------------------------------
# ClubSpecification column_mappings class attribute
# ---------------------------------------------------------------------------


class TestClubDataLoaderMappings:
    """Smoke tests for column mapping attributes."""

    def test_club_column_mappings_nonempty(self) -> None:
        """CLUB_COLUMN_MAPPINGS is a non-empty dict."""
        from src.shared.python.club_data.loader import ClubDataLoader

        assert len(ClubDataLoader.CLUB_COLUMN_MAPPINGS) > 0

    def test_player_column_mappings_nonempty(self) -> None:
        """PLAYER_COLUMN_MAPPINGS is a non-empty dict."""
        from src.shared.python.club_data.loader import ClubDataLoader

        assert len(ClubDataLoader.PLAYER_COLUMN_MAPPINGS) > 0
