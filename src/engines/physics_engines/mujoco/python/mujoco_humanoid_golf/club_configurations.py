"""Golf club configuration database.

Provides realistic club specifications for different club types.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast  # noqa: ICN003


@dataclass
class ClubSpecification:
    """Specification for a golf club."""

    name: str
    club_type: str  # Driver, Wood, Hybrid, Iron, Wedge, Putter
    number: str | None = None  # e.g., "3" for 3-wood, "7" for 7-iron
    length_inches: float = 45.5  # Club length
    head_mass_grams: float = 200.0  # Club head mass
    loft_degrees: float = 10.0  # Loft angle
    lie_angle_degrees: float = 56.0  # Lie angle
    shaft_flexibility: str = "Regular"  # Ladies, Senior, Regular, Stiff, X-Stiff
    shaft_mass_grams: float = 65.0  # Shaft mass
    grip_mass_grams: float = 50.0  # Grip mass
    swing_weight: str = "D2"  # Swing weight
    moment_of_inertia: float = 5000.0  # MOI (g·cm²)
    center_of_gravity_mm: float = 25.0  # CG distance from face
    description: str = ""


class ClubDatabase:
    """Database of golf club specifications."""

    # Standard club database
    CLUBS = {
        # Drivers
        "driver": ClubSpecification(
            name="Driver",
            club_type="Driver",
            number="1",
            length_inches=45.5,
            head_mass_grams=200,
            loft_degrees=10.5,
            lie_angle_degrees=56,
            shaft_flexibility="Regular",
            shaft_mass_grams=65,
            grip_mass_grams=50,
            swing_weight="D2",
            moment_of_inertia=5200,
            center_of_gravity_mm=25,
            description="Maximum distance driver",
        ),
        "driver_low_loft": ClubSpecification(
            name="Driver (Low Loft)",
            club_type="Driver",
            number="1",
            length_inches=45.5,
            head_mass_grams=200,
            loft_degrees=8.5,
            lie_angle_degrees=56,
            shaft_flexibility="Stiff",
            shaft_mass_grams=70,
            grip_mass_grams=50,
            swing_weight="D3",
            moment_of_inertia=5400,
            center_of_gravity_mm=23,
            description="Low loft for high swing speed",
        ),
        # Fairway Woods
        "3_wood": ClubSpecification(
            name="3-Wood",
            club_type="Wood",
            number="3",
            length_inches=43.0,
            head_mass_grams=210,
            loft_degrees=15.0,
            lie_angle_degrees=57,
            shaft_flexibility="Regular",
            shaft_mass_grams=70,
            grip_mass_grams=50,
            swing_weight="D2",
            moment_of_inertia=4500,
            center_of_gravity_mm=22,
            description="Versatile fairway wood",
        ),
        "5_wood": ClubSpecification(
            name="5-Wood",
            club_type="Wood",
            number="5",
            length_inches=42.0,
            head_mass_grams=215,
            loft_degrees=18.0,
            lie_angle_degrees=58,
            shaft_flexibility="Regular",
            shaft_mass_grams=72,
            grip_mass_grams=50,
            swing_weight="D2",
            moment_of_inertia=4300,
            center_of_gravity_mm=20,
            description="High launch fairway wood",
        ),
        # Hybrids
        "3_hybrid": ClubSpecification(
            name="3-Hybrid",
            club_type="Hybrid",
            number="3",
            length_inches=40.5,
            head_mass_grams=230,
            loft_degrees=19.0,
            lie_angle_degrees=59,
            shaft_flexibility="Regular",
            shaft_mass_grams=75,
            grip_mass_grams=50,
            swing_weight="D2",
            moment_of_inertia=3800,
            center_of_gravity_mm=18,
            description="Easy to hit hybrid",
        ),
        # Irons
        "3_iron": ClubSpecification(
            name="3-Iron",
            club_type="Iron",
            number="3",
            length_inches=39.0,
            head_mass_grams=240,
            loft_degrees=21.0,
            lie_angle_degrees=59.5,
            shaft_flexibility="Regular",
            shaft_mass_grams=85,
            grip_mass_grams=50,
            swing_weight="D2",
            moment_of_inertia=2800,
            center_of_gravity_mm=15,
            description="Long iron",
        ),
        "5_iron": ClubSpecification(
            name="5-Iron",
            club_type="Iron",
            number="5",
            length_inches=38.0,
            head_mass_grams=245,
            loft_degrees=27.0,
            lie_angle_degrees=61,
            shaft_flexibility="Regular",
            shaft_mass_grams=90,
            grip_mass_grams=50,
            swing_weight="D2",
            moment_of_inertia=2600,
            center_of_gravity_mm=14,
            description="Mid iron",
        ),
        "7_iron": ClubSpecification(
            name="7-Iron",
            club_type="Iron",
            number="7",
            length_inches=37.0,
            head_mass_grams=250,
            loft_degrees=34.0,
            lie_angle_degrees=62.5,
            shaft_flexibility="Regular",
            shaft_mass_grams=95,
            grip_mass_grams=50,
            swing_weight="D2",
            moment_of_inertia=2400,
            center_of_gravity_mm=13,
            description="Standard scoring iron",
        ),
        "9_iron": ClubSpecification(
            name="9-Iron",
            club_type="Iron",
            number="9",
            length_inches=36.0,
            head_mass_grams=255,
            loft_degrees=41.0,
            lie_angle_degrees=64,
            shaft_flexibility="Regular",
            shaft_mass_grams=100,
            grip_mass_grams=50,
            swing_weight="D2",
            moment_of_inertia=2200,
            center_of_gravity_mm=12,
            description="High trajectory iron",
        ),
        # Wedges
        "pitching_wedge": ClubSpecification(
            name="Pitching Wedge",
            club_type="Wedge",
            number="PW",
            length_inches=35.5,
            head_mass_grams=290,
            loft_degrees=46.0,
            lie_angle_degrees=64,
            shaft_flexibility="Regular",
            shaft_mass_grams=105,
            grip_mass_grams=50,
            swing_weight="D3",
            moment_of_inertia=2100,
            center_of_gravity_mm=11,
            description="Standard pitching wedge",
        ),
        "gap_wedge": ClubSpecification(
            name="Gap Wedge",
            club_type="Wedge",
            number="GW",
            length_inches=35.25,
            head_mass_grams=295,
            loft_degrees=52.0,
            lie_angle_degrees=64,
            shaft_flexibility="Regular",
            shaft_mass_grams=105,
            grip_mass_grams=50,
            swing_weight="D3",
            moment_of_inertia=2000,
            center_of_gravity_mm=10,
            description="Gap/Approach wedge",
        ),
        "sand_wedge": ClubSpecification(
            name="Sand Wedge",
            club_type="Wedge",
            number="SW",
            length_inches=35.0,
            head_mass_grams=300,
            loft_degrees=56.0,
            lie_angle_degrees=64,
            shaft_flexibility="Regular",
            shaft_mass_grams=105,
            grip_mass_grams=50,
            swing_weight="D4",
            moment_of_inertia=1900,
            center_of_gravity_mm=10,
            description="Bunker and short game wedge",
        ),
        "lob_wedge": ClubSpecification(
            name="Lob Wedge",
            club_type="Wedge",
            number="LW",
            length_inches=35.0,
            head_mass_grams=305,
            loft_degrees=60.0,
            lie_angle_degrees=64,
            shaft_flexibility="Regular",
            shaft_mass_grams=105,
            grip_mass_grams=50,
            swing_weight="D4",
            moment_of_inertia=1850,
            center_of_gravity_mm=9,
            description="High loft specialty wedge",
        ),
        # Putters
        "putter_blade": ClubSpecification(
            name="Blade Putter",
            club_type="Putter",
            length_inches=35.0,
            head_mass_grams=350,
            loft_degrees=3.0,
            lie_angle_degrees=70,
            shaft_flexibility="N/A",
            shaft_mass_grams=120,
            grip_mass_grams=75,
            swing_weight="E0",
            moment_of_inertia=2500,
            center_of_gravity_mm=5,
            description="Traditional blade putter",
        ),
        "putter_mallet": ClubSpecification(
            name="Mallet Putter",
            club_type="Putter",
            length_inches=34.0,
            head_mass_grams=360,
            loft_degrees=3.0,
            lie_angle_degrees=70,
            shaft_flexibility="N/A",
            shaft_mass_grams=120,
            grip_mass_grams=75,
            swing_weight="E2",
            moment_of_inertia=5500,
            center_of_gravity_mm=6,
            description="High MOI mallet putter",
        ),
    }

    @classmethod
    def get_club(cls, club_id: str) -> ClubSpecification | None:
        """Get club specification by ID.

        Args:
            club_id: Club identifier (e.g., 'driver', '7_iron')

        Returns:
            ClubSpecification or None if not found
        """
        return cls.CLUBS.get(club_id)

    @classmethod
    def get_all_clubs(cls) -> dict[str, ClubSpecification]:
        """Get all club specifications.

        Returns:
            Dictionary mapping club ID to specification
        """
        return cls.CLUBS.copy()

    @classmethod
    def get_clubs_by_type(cls, club_type: str) -> list[ClubSpecification]:
        """Get all clubs of a specific type.

        Args:
            club_type: Club type (Driver, Wood, Iron, Wedge, Putter)

        Returns:
            List of matching ClubSpecifications
        """
        return [spec for spec in cls.CLUBS.values() if spec.club_type == club_type]

    @classmethod
    def get_club_types(cls) -> list[str]:
        """Get list of all club types.

        Returns:
            List of unique club types
        """
        types = {spec.club_type for spec in cls.CLUBS.values()}
        return sorted(types)

    @classmethod
    def compute_total_mass(cls, spec: ClubSpecification) -> float:
        """Compute total club mass.

        Args:
            spec: Club specification

        Returns:
            Total mass in grams
        """
        return spec.head_mass_grams + spec.shaft_mass_grams + spec.grip_mass_grams

    @classmethod
    def compute_total_mass_kg(cls, spec: ClubSpecification) -> float:
        """Compute total club mass in kg.

        Args:
            spec: Club specification

        Returns:
            Total mass in kilograms
        """
        return cls.compute_total_mass(spec) / 1000.0

    @classmethod
    def length_to_meters(cls, length_inches: float) -> float:
        """Convert club length to meters.

        Args:
            length_inches: Length in inches

        Returns:
            Length in meters
        """
        return length_inches * 0.0254

    @classmethod
    def export_to_json(cls, output_path: str) -> None:
        """Export club database to JSON file.

        Args:
            output_path: Output JSON file path
        """
        data = {}
        for club_id, spec in cls.CLUBS.items():
            data[club_id] = {
                "name": spec.name,
                "club_type": spec.club_type,
                "number": spec.number,
                "length_inches": spec.length_inches,
                "head_mass_grams": spec.head_mass_grams,
                "loft_degrees": spec.loft_degrees,
                "lie_angle_degrees": spec.lie_angle_degrees,
                "shaft_flexibility": spec.shaft_flexibility,
                "shaft_mass_grams": spec.shaft_mass_grams,
                "grip_mass_grams": spec.grip_mass_grams,
                "swing_weight": spec.swing_weight,
                "moment_of_inertia": spec.moment_of_inertia,
                "center_of_gravity_mm": spec.center_of_gravity_mm,
                "description": spec.description,
            }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def create_custom_club(
        cls,
        name: str,
        club_type: str,
        **kwargs: Any,
    ) -> ClubSpecification:
        """Create a custom club specification.

        Args:
            name: Club name
            club_type: Club type
            **kwargs: Additional specification parameters

        Returns:
            ClubSpecification
        """
        # Start with default values
        defaults: dict[str, Any] = {
            "length_inches": 40.0,
            "head_mass_grams": 250.0,
            "loft_degrees": 30.0,
            "lie_angle_degrees": 62.0,
            "shaft_flexibility": "Regular",
            "shaft_mass_grams": 80.0,
            "grip_mass_grams": 50.0,
            "swing_weight": "D2",
            "moment_of_inertia": 3000.0,
            "center_of_gravity_mm": 15.0,
            "description": "",
        }

        # Update with provided values
        defaults.update(kwargs)

        return ClubSpecification(name=name, club_type=club_type, **defaults)


# Shaft flexibility characteristics
SHAFT_FLEX_DATA: dict[str, dict[str, Any]] = {
    "Ladies": {
        "swing_speed_mph": (60, 70),
        "stiffness_cpm": 220,
        "description": "Most flexible, for slower swing speeds",
    },
    "Senior": {
        "swing_speed_mph": (70, 80),
        "stiffness_cpm": 240,
        "description": "Flexible, for moderate swing speeds",
    },
    "Regular": {
        "swing_speed_mph": (80, 95),
        "stiffness_cpm": 260,
        "description": "Standard flex for average swing speeds",
    },
    "Stiff": {
        "swing_speed_mph": (95, 105),
        "stiffness_cpm": 280,
        "description": "Stiff flex for fast swing speeds",
    },
    "X-Stiff": {
        "swing_speed_mph": (105, 125),
        "stiffness_cpm": 300,
        "description": "Extra stiff for very fast swing speeds",
    },
}


def get_recommended_flex(swing_speed_mph: float) -> str:
    """Get recommended shaft flex for swing speed.

    Args:
        swing_speed_mph: Swing speed in mph

    Returns:
        Recommended shaft flexibility
    """
    for flex, data in SHAFT_FLEX_DATA.items():
        speed_range = cast("tuple[float, float]", data["swing_speed_mph"])
        min_speed, max_speed = speed_range
        if min_speed <= swing_speed_mph <= max_speed:
            return flex

    # Default fallbacks
    if swing_speed_mph < 60:
        return "Ladies"
    return "X-Stiff"
