"""Tests for src.shared.python.club_data.loader (Issues #1949, #1744)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from src.shared.python.club_data.loader import (
    ClubDataLoader,
    ClubSpecification,
    load_club_data,
)


class TestClubSpecification:
    def test_loader_basic_construction(self) -> None:
        club = ClubSpecification(name="Driver", club_type="Driver")
        assert isinstance(club, ClubSpecification)

    def test_loader_name_stored(self) -> None:
        club = ClubSpecification(name="7-Iron", club_type="Iron")
        assert club.name == "7-Iron"

    def test_club_type_stored(self) -> None:
        club = ClubSpecification(name="PW", club_type="Wedge")
        assert club.club_type == "Wedge"

    def test_default_length_inches(self) -> None:
        club = ClubSpecification(name="Driver", club_type="Driver")
        assert club.length_inches == pytest.approx(45.5)

    def test_length_meters_derived_from_inches(self) -> None:
        club = ClubSpecification(name="Driver", club_type="Driver", length_inches=45.5)
        # post_init computes length_meters = length_inches * 0.0254
        assert club.length_meters == pytest.approx(45.5 * 0.0254)

    def test_head_mass_kg_derived_from_grams(self) -> None:
        club = ClubSpecification(
            name="Driver", club_type="Driver", head_mass_grams=200.0
        )
        assert club.head_mass_kg == pytest.approx(0.2)

    def test_total_mass_grams(self) -> None:
        club = ClubSpecification(
            name="Driver",
            club_type="Driver",
            head_mass_grams=200.0,
            shaft_mass_grams=65.0,
            grip_mass_grams=50.0,
        )
        assert club.total_mass_grams == pytest.approx(315.0)

    def test_total_mass_kg(self) -> None:
        club = ClubSpecification(
            name="Driver",
            club_type="Driver",
            head_mass_grams=200.0,
            shaft_mass_grams=65.0,
            grip_mass_grams=50.0,
        )
        assert club.total_mass_kg == pytest.approx(0.315)

    def test_loader_to_dict_returns_dict(self) -> None:
        club = ClubSpecification(name="Driver", club_type="Driver")
        result = club.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_has_name(self) -> None:
        club = ClubSpecification(name="3-Wood", club_type="Wood")
        d = club.to_dict()
        assert d["name"] == "3-Wood"

    def test_custom_loft(self) -> None:
        club = ClubSpecification(name="PW", club_type="Wedge", loft_degrees=45.0)
        assert club.loft_degrees == pytest.approx(45.0)

    def test_number_optional(self) -> None:
        club = ClubSpecification(name="Driver", club_type="Driver", number=None)
        assert club.number is None

    def test_number_stored(self) -> None:
        club = ClubSpecification(name="7-Iron", club_type="Iron", number="7")
        assert club.number == "7"


class TestClubDataLoader:
    def test_loader_construction(self) -> None:
        loader = ClubDataLoader()
        assert isinstance(loader, ClubDataLoader)

    def test_get_all_clubs_initially_empty(self) -> None:
        loader = ClubDataLoader()
        assert loader.get_all_clubs() == []

    def test_get_all_players_initially_empty(self) -> None:
        loader = ClubDataLoader()
        assert loader.get_all_players() == []

    def test_get_club_returns_none_when_empty(self) -> None:
        loader = ClubDataLoader()
        assert loader.get_club("Driver") is None

    def test_get_player_returns_none_when_empty(self) -> None:
        loader = ClubDataLoader()
        assert loader.get_player("Tiger Woods") is None

    def test_load_clubs_from_json(self, tmp_path: Path) -> None:
        clubs_data = [
            {
                "name": "Driver",
                "club_type": "Driver",
                "length_inches": 45.5,
            }
        ]
        json_file = tmp_path / "clubs.json"
        json_file.write_text(json.dumps(clubs_data))

        loader = ClubDataLoader()
        loader.load_clubs_from_json(str(json_file))
        clubs = loader.get_all_clubs()
        assert len(clubs) == 1
        assert clubs[0].name == "Driver"

    def test_get_club_after_load(self, tmp_path: Path) -> None:
        clubs_data = [{"name": "7-Iron", "club_type": "Iron"}]
        json_file = tmp_path / "clubs.json"
        json_file.write_text(json.dumps(clubs_data))

        loader = ClubDataLoader()
        loader.load_clubs_from_json(str(json_file))
        club = loader.get_club("7-Iron")
        assert club is not None
        assert club.name == "7-Iron"


class TestLoadClubData:
    def test_load_from_json_file(self, tmp_path: Path) -> None:
        clubs_data = [
            {"name": "Driver", "club_type": "Driver"},
            {"name": "Putter", "club_type": "Putter"},
        ]
        json_file = tmp_path / "clubs.json"
        json_file.write_text(json.dumps(clubs_data))

        result = load_club_data(str(json_file))
        assert isinstance(result, list)
        assert len(result) == 2

    def test_returns_club_specifications(self, tmp_path: Path) -> None:
        clubs_data = [{"name": "Driver", "club_type": "Driver"}]
        json_file = tmp_path / "clubs.json"
        json_file.write_text(json.dumps(clubs_data))

        result = load_club_data(str(json_file))
        assert all(isinstance(c, ClubSpecification) for c in result)
