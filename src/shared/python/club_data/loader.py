"""Club data loader with Excel support.

Loads golf club specifications and professional player swing data from
Excel files for use as target trajectories in physics simulations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.logging_pkg.logging_config import get_logger
import contextlib

logger = get_logger(__name__)

# Optional pandas import for Excel support
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    pd = None  # type: ignore[misc, assignment]
    PANDAS_AVAILABLE = False

# Optional openpyxl for Excel support
try:
    import openpyxl

    OPENPYXL_AVAILABLE = True
except ImportError:
    openpyxl = None  # type: ignore[misc, assignment]
    OPENPYXL_AVAILABLE = False


@dataclass
class ClubSpecification:
    """Complete specification for a golf club."""

    name: str
    club_type: str  # Driver, Wood, Hybrid, Iron, Wedge, Putter
    number: str | None = None  # e.g., "3" for 3-wood, "7" for 7-iron
    length_inches: float = 45.5  # Club length
    length_meters: float = 1.1557  # Club length in meters
    head_mass_grams: float = 200.0  # Club head mass
    head_mass_kg: float = 0.2  # Club head mass in kg
    loft_degrees: float = 10.0  # Loft angle
    lie_angle_degrees: float = 56.0  # Lie angle
    shaft_flexibility: str = "Regular"  # Ladies, Senior, Regular, Stiff, X-Stiff
    shaft_mass_grams: float = 65.0  # Shaft mass
    grip_mass_grams: float = 50.0  # Grip mass
    swing_weight: str = "D2"  # Swing weight
    moment_of_inertia: float = 5000.0  # MOI (g*cm^2)
    center_of_gravity_mm: float = 25.0  # CG distance from face
    description: str = ""

    def __post_init__(self) -> None:
        """Compute derived values."""
        self.length_meters = self.length_inches * 0.0254
        self.head_mass_kg = self.head_mass_grams / 1000.0

    @property
    def total_mass_grams(self) -> float:
        """Total club mass in grams."""
        return self.head_mass_grams + self.shaft_mass_grams + self.grip_mass_grams

    @property
    def total_mass_kg(self) -> float:
        """Total club mass in kg."""
        return self.total_mass_grams / 1000.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "club_type": self.club_type,
            "number": self.number,
            "length_inches": self.length_inches,
            "length_meters": self.length_meters,
            "head_mass_grams": self.head_mass_grams,
            "head_mass_kg": self.head_mass_kg,
            "loft_degrees": self.loft_degrees,
            "lie_angle_degrees": self.lie_angle_degrees,
            "shaft_flexibility": self.shaft_flexibility,
            "shaft_mass_grams": self.shaft_mass_grams,
            "grip_mass_grams": self.grip_mass_grams,
            "swing_weight": self.swing_weight,
            "moment_of_inertia": self.moment_of_inertia,
            "center_of_gravity_mm": self.center_of_gravity_mm,
            "total_mass_grams": self.total_mass_grams,
            "total_mass_kg": self.total_mass_kg,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClubSpecification:
        """Create from dictionary."""
        return cls(
            name=data.get("name", "Unknown"),
            club_type=data.get("club_type", "Iron"),
            number=data.get("number"),
            length_inches=float(data.get("length_inches", 37.0)),
            head_mass_grams=float(data.get("head_mass_grams", 250.0)),
            loft_degrees=float(data.get("loft_degrees", 30.0)),
            lie_angle_degrees=float(data.get("lie_angle_degrees", 62.0)),
            shaft_flexibility=data.get("shaft_flexibility", "Regular"),
            shaft_mass_grams=float(data.get("shaft_mass_grams", 80.0)),
            grip_mass_grams=float(data.get("grip_mass_grams", 50.0)),
            swing_weight=data.get("swing_weight", "D2"),
            moment_of_inertia=float(data.get("moment_of_inertia", 3000.0)),
            center_of_gravity_mm=float(data.get("center_of_gravity_mm", 15.0)),
            description=data.get("description", ""),
        )


@dataclass
class SwingMetrics:
    """Swing performance metrics from professional player data."""

    club_head_speed_mph: float = 0.0
    club_head_speed_ms: float = 0.0  # meters per second
    ball_speed_mph: float = 0.0
    ball_speed_ms: float = 0.0
    launch_angle_degrees: float = 0.0
    spin_rate_rpm: float = 0.0
    carry_distance_yards: float = 0.0
    carry_distance_meters: float = 0.0
    total_distance_yards: float = 0.0
    total_distance_meters: float = 0.0
    smash_factor: float = 0.0
    attack_angle_degrees: float = 0.0
    club_path_degrees: float = 0.0
    face_angle_degrees: float = 0.0
    dynamic_loft_degrees: float = 0.0
    face_to_path_degrees: float = 0.0
    swing_tempo_ratio: float = 3.0  # Ideal is 3:1 backswing to downswing

    def __post_init__(self) -> None:
        """Compute derived values."""
        # Convert mph to m/s (1 mph = 0.44704 m/s)
        if self.club_head_speed_mph > 0 and self.club_head_speed_ms == 0:
            self.club_head_speed_ms = self.club_head_speed_mph * 0.44704
        if self.ball_speed_mph > 0 and self.ball_speed_ms == 0:
            self.ball_speed_ms = self.ball_speed_mph * 0.44704

        # Convert yards to meters (1 yard = 0.9144 m)
        if self.carry_distance_yards > 0 and self.carry_distance_meters == 0:
            self.carry_distance_meters = self.carry_distance_yards * 0.9144
        if self.total_distance_yards > 0 and self.total_distance_meters == 0:
            self.total_distance_meters = self.total_distance_yards * 0.9144

        # Compute smash factor if not provided
        if (
            self.smash_factor == 0
            and self.club_head_speed_mph > 0
            and self.ball_speed_mph > 0
        ):
            self.smash_factor = self.ball_speed_mph / self.club_head_speed_mph

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "club_head_speed_mph": self.club_head_speed_mph,
            "club_head_speed_ms": self.club_head_speed_ms,
            "ball_speed_mph": self.ball_speed_mph,
            "ball_speed_ms": self.ball_speed_ms,
            "launch_angle_degrees": self.launch_angle_degrees,
            "spin_rate_rpm": self.spin_rate_rpm,
            "carry_distance_yards": self.carry_distance_yards,
            "carry_distance_meters": self.carry_distance_meters,
            "total_distance_yards": self.total_distance_yards,
            "total_distance_meters": self.total_distance_meters,
            "smash_factor": self.smash_factor,
            "attack_angle_degrees": self.attack_angle_degrees,
            "club_path_degrees": self.club_path_degrees,
            "face_angle_degrees": self.face_angle_degrees,
            "dynamic_loft_degrees": self.dynamic_loft_degrees,
            "face_to_path_degrees": self.face_to_path_degrees,
            "swing_tempo_ratio": self.swing_tempo_ratio,
        }


@dataclass
class ProPlayerData:
    """Professional player swing data for target matching."""

    player_name: str
    skill_level: str = "Professional"  # Professional, Amateur, Beginner
    handedness: str = "Right"  # Left or Right
    club: ClubSpecification | None = None
    metrics: SwingMetrics = field(default_factory=SwingMetrics)

    # Time-series trajectory data (optional)
    time_series: np.ndarray | None = None  # Time points [s]
    club_head_positions: np.ndarray | None = None  # (N, 3) positions [m]
    club_head_velocities: np.ndarray | None = None  # (N, 3) velocities [m/s]
    joint_angles: np.ndarray | None = None  # (N, M) joint angles [rad]
    joint_names: list[str] = field(default_factory=list)

    # Swing phase markers
    address_time: float = 0.0
    top_of_backswing_time: float = 0.0
    impact_time: float = 0.0
    finish_time: float = 0.0

    def has_trajectory_data(self) -> bool:
        """Check if trajectory data is available."""
        return (
            self.time_series is not None
            and len(self.time_series) > 0
            and self.club_head_positions is not None
        )

    def get_position_at_time(self, t: float) -> np.ndarray | None:
        """Interpolate position at a specific time."""
        if not self.has_trajectory_data():
            return None

        if self.time_series is None or self.club_head_positions is None:
            return None

        if t < self.time_series[0] or t > self.time_series[-1]:
            return None

        # Linear interpolation
        idx = np.searchsorted(self.time_series, t)
        if idx == 0:
            return self.club_head_positions[0]
        if idx >= len(self.time_series):
            return self.club_head_positions[-1]

        t0, t1 = self.time_series[idx - 1], self.time_series[idx]
        alpha = (t - t0) / (t1 - t0)

        p0 = self.club_head_positions[idx - 1]
        p1 = self.club_head_positions[idx]
        return p0 + alpha * (p1 - p0)

    def get_velocity_at_time(self, t: float) -> np.ndarray | None:
        """Interpolate velocity at a specific time."""
        if self.club_head_velocities is None or self.time_series is None:
            return None

        if t < self.time_series[0] or t > self.time_series[-1]:
            return None

        idx = np.searchsorted(self.time_series, t)
        if idx == 0:
            return self.club_head_velocities[0]
        if idx >= len(self.time_series):
            return self.club_head_velocities[-1]

        t0, t1 = self.time_series[idx - 1], self.time_series[idx]
        alpha = (t - t0) / (t1 - t0)

        v0 = self.club_head_velocities[idx - 1]
        v1 = self.club_head_velocities[idx]
        return v0 + alpha * (v1 - v0)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (without numpy arrays for JSON serialization)."""
        return {
            "player_name": self.player_name,
            "skill_level": self.skill_level,
            "handedness": self.handedness,
            "club": self.club.to_dict() if self.club else None,
            "metrics": self.metrics.to_dict(),
            "joint_names": self.joint_names,
            "address_time": self.address_time,
            "top_of_backswing_time": self.top_of_backswing_time,
            "impact_time": self.impact_time,
            "finish_time": self.finish_time,
        }


class ClubDataLoader:
    """Loader for club data from Excel files and other sources."""

    # Column name mappings for Excel files
    CLUB_COLUMN_MAPPINGS = {
        "name": ["Name", "Club Name", "club_name", "name"],
        "club_type": ["Type", "Club Type", "club_type", "type"],
        "number": ["Number", "Club Number", "number", "No."],
        "length_inches": ["Length (in)", "Length", "length_inches", "length"],
        "head_mass_grams": [
            "Head Mass (g)",
            "Head Mass",
            "head_mass_grams",
            "head_mass",
        ],
        "loft_degrees": ["Loft (deg)", "Loft", "loft_degrees", "loft"],
        "lie_angle_degrees": [
            "Lie (deg)",
            "Lie Angle",
            "lie_angle_degrees",
            "lie_angle",
        ],
        "shaft_flexibility": ["Shaft Flex", "Flex", "shaft_flexibility", "flex"],
        "shaft_mass_grams": ["Shaft Mass (g)", "Shaft Mass", "shaft_mass_grams"],
        "grip_mass_grams": ["Grip Mass (g)", "Grip Mass", "grip_mass_grams"],
        "swing_weight": ["Swing Weight", "SW", "swing_weight"],
        "moment_of_inertia": ["MOI", "MOI (g*cm2)", "moment_of_inertia"],
        "center_of_gravity_mm": ["CG (mm)", "CG", "center_of_gravity_mm"],
    }

    PLAYER_COLUMN_MAPPINGS = {
        "player_name": ["Player", "Player Name", "player_name", "name"],
        "club_type": ["Club", "Club Type", "club_type"],
        "club_head_speed_mph": [
            "Club Speed (mph)",
            "Club Head Speed",
            "club_head_speed_mph",
        ],
        "ball_speed_mph": ["Ball Speed (mph)", "Ball Speed", "ball_speed_mph"],
        "launch_angle_degrees": [
            "Launch Angle",
            "Launch (deg)",
            "launch_angle_degrees",
        ],
        "spin_rate_rpm": ["Spin Rate", "Spin (rpm)", "spin_rate_rpm"],
        "carry_distance_yards": ["Carry (yds)", "Carry", "carry_distance_yards"],
        "total_distance_yards": [
            "Total (yds)",
            "Total Distance",
            "total_distance_yards",
        ],
        "attack_angle_degrees": ["Attack Angle", "AoA (deg)", "attack_angle_degrees"],
        "club_path_degrees": ["Club Path", "Path (deg)", "club_path_degrees"],
        "face_angle_degrees": ["Face Angle", "Face (deg)", "face_angle_degrees"],
    }

    def __init__(self) -> None:
        """Initialize the loader."""
        self._club_cache: dict[str, ClubSpecification] = {}
        self._player_cache: dict[str, ProPlayerData] = {}

    def load_clubs_from_excel(
        self, file_path: str | Path, sheet_name: str | int = 0
    ) -> list[ClubSpecification]:
        """Load club specifications from an Excel file.

        Args:
            file_path: Path to the Excel file
            sheet_name: Sheet name or index to read

        Returns:
            List of ClubSpecification objects

        Raises:
            ImportError: If pandas or openpyxl is not available
            FileNotFoundError: If file does not exist
            ValueError: If file format is invalid
        """
        if not PANDAS_AVAILABLE:
            raise ImportError(
                "pandas is required for Excel loading. Install with: pip install pandas"
            )
        if not OPENPYXL_AVAILABLE:
            raise ImportError(
                "openpyxl is required for Excel loading. Install with: pip install openpyxl"
            )

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Club data file not found: {file_path}")

        logger.info("Loading club data from: %s", file_path)

        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        except (RuntimeError, TypeError, ValueError) as e:
            raise ValueError(f"Failed to read Excel file: {e}") from e

        clubs = []
        for _, row in df.iterrows():
            club_data = self._extract_club_data(row)
            if club_data:
                club = ClubSpecification.from_dict(club_data)
                clubs.append(club)
                self._club_cache[club.name.lower()] = club

        logger.info("Loaded %d clubs from Excel", len(clubs))
        return clubs

    def load_player_data_from_excel(
        self, file_path: str | Path, sheet_name: str | int = 0
    ) -> list[ProPlayerData]:
        """Load professional player data from an Excel file.

        Args:
            file_path: Path to the Excel file
            sheet_name: Sheet name or index to read

        Returns:
            List of ProPlayerData objects
        """
        if not PANDAS_AVAILABLE or not OPENPYXL_AVAILABLE:
            raise ImportError("pandas and openpyxl are required for Excel loading")

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Player data file not found: {file_path}")

        logger.info("Loading player data from: %s", file_path)

        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        except (RuntimeError, TypeError, ValueError) as e:
            raise ValueError(f"Failed to read Excel file: {e}") from e

        players = []
        for _, row in df.iterrows():
            player_data = self._extract_player_data(row)
            if player_data:
                players.append(player_data)
                self._player_cache[player_data.player_name.lower()] = player_data

        logger.info("Loaded %d player records from Excel", len(players))
        return players

    def load_trajectory_from_excel(
        self,
        file_path: str | Path,
        player_name: str = "Unknown",
        sheet_name: str | int = 0,
    ) -> ProPlayerData:
        """Load trajectory time-series data from an Excel file.

        Expected columns: Time, X, Y, Z, Vx, Vy, Vz (optional)

        Args:
            file_path: Path to the Excel file
            player_name: Name to assign to the player
            sheet_name: Sheet name or index

        Returns:
            ProPlayerData with trajectory information
        """
        if not PANDAS_AVAILABLE or not OPENPYXL_AVAILABLE:
            raise ImportError("pandas and openpyxl are required for Excel loading")

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {file_path}")

        df = pd.read_excel(file_path, sheet_name=sheet_name)

        # Find time column
        time_col = None
        for col in df.columns:
            if col.lower() in ["time", "t", "time (s)", "time_s"]:
                time_col = col
                break

        if time_col is None:
            raise ValueError("No time column found in trajectory data")

        time_series = df[time_col].values.astype(float)

        # Find position columns
        positions = None
        x_col = self._find_column(df, ["x", "pos_x", "position_x"])
        y_col = self._find_column(df, ["y", "pos_y", "position_y"])
        z_col = self._find_column(df, ["z", "pos_z", "position_z"])

        if x_col and y_col and z_col:
            positions = np.column_stack(
                [
                    df[x_col].to_numpy(),
                    df[y_col].to_numpy(),
                    df[z_col].to_numpy(),
                ]
            )

        # Find velocity columns (optional)
        velocities = None
        vx_col = self._find_column(df, ["vx", "vel_x", "velocity_x"])
        vy_col = self._find_column(df, ["vy", "vel_y", "velocity_y"])
        vz_col = self._find_column(df, ["vz", "vel_z", "velocity_z"])

        if vx_col and vy_col and vz_col:
            velocities = np.column_stack(
                [
                    df[vx_col].to_numpy(),
                    df[vy_col].to_numpy(),
                    df[vz_col].to_numpy(),
                ]
            )

        player = ProPlayerData(
            player_name=player_name,
            time_series=time_series,
            club_head_positions=positions,
            club_head_velocities=velocities,
        )

        logger.info(
            "Loaded trajectory with %d frames for player: %s",
            len(time_series),
            player_name,
        )
        return player

    def load_clubs_from_json(self, file_path: str | Path) -> list[ClubSpecification]:
        """Load club specifications from a JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            List of ClubSpecification objects
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Club data file not found: {file_path}")

        with open(file_path) as f:
            data = json.load(f)

        clubs = []
        if isinstance(data, dict):
            for _club_id, club_data in data.items():
                if isinstance(club_data, dict):
                    club = ClubSpecification.from_dict(club_data)
                    clubs.append(club)
                    self._club_cache[club.name.lower()] = club
        elif isinstance(data, list):
            for club_data in data:
                if isinstance(club_data, dict):
                    club = ClubSpecification.from_dict(club_data)
                    clubs.append(club)
                    self._club_cache[club.name.lower()] = club

        return clubs

    def get_club(self, name: str) -> ClubSpecification | None:
        """Get a cached club by name."""
        return self._club_cache.get(name.lower())

    def get_player(self, name: str) -> ProPlayerData | None:
        """Get cached player data by name."""
        return self._player_cache.get(name.lower())

    def get_all_clubs(self) -> list[ClubSpecification]:
        """Get all cached clubs."""
        return list(self._club_cache.values())

    def get_all_players(self) -> list[ProPlayerData]:
        """Get all cached players."""
        return list(self._player_cache.values())

    def _find_column(self, df: Any, possible_names: list[str]) -> str | None:
        """Find a column by checking multiple possible names."""
        for name in possible_names:
            for col in df.columns:
                if col.lower() == name.lower():
                    return col
        return None

    def _extract_club_data(self, row: Any) -> dict[str, Any] | None:
        """Extract club data from a DataFrame row."""
        data: dict[str, Any] = {}

        for field_name, possible_cols in self.CLUB_COLUMN_MAPPINGS.items():
            for col in possible_cols:
                if col in row.index:
                    value = row[col]
                    if pd.notna(value):
                        data[field_name] = value
                    break

        # Must have at least a name
        if "name" not in data:
            return None

        return data

    def _extract_player_data(self, row: Any) -> ProPlayerData | None:
        """Extract player data from a DataFrame row."""
        metrics_data: dict[str, float] = {}
        player_name = None

        for field_name, possible_cols in self.PLAYER_COLUMN_MAPPINGS.items():
            for col in possible_cols:
                if col in row.index:
                    value = row[col]
                    if pd.notna(value):
                        if field_name == "player_name":
                            player_name = str(value)
                        else:
                            with contextlib.suppress(ValueError, TypeError):
                                metrics_data[field_name] = float(value)
                    break

        if not player_name:
            return None

        # Create metrics from extracted data
        metrics = SwingMetrics(
            club_head_speed_mph=metrics_data.get("club_head_speed_mph", 0.0),
            ball_speed_mph=metrics_data.get("ball_speed_mph", 0.0),
            launch_angle_degrees=metrics_data.get("launch_angle_degrees", 0.0),
            spin_rate_rpm=metrics_data.get("spin_rate_rpm", 0.0),
            carry_distance_yards=metrics_data.get("carry_distance_yards", 0.0),
            total_distance_yards=metrics_data.get("total_distance_yards", 0.0),
            attack_angle_degrees=metrics_data.get("attack_angle_degrees", 0.0),
            club_path_degrees=metrics_data.get("club_path_degrees", 0.0),
            face_angle_degrees=metrics_data.get("face_angle_degrees", 0.0),
        )

        return ProPlayerData(
            player_name=player_name,
            metrics=metrics,
        )


def load_club_data(file_path: str | Path) -> list[ClubSpecification]:
    """Convenience function to load club data from a file.

    Automatically detects file format (Excel or JSON).

    Args:
        file_path: Path to the data file

    Returns:
        List of ClubSpecification objects
    """
    loader = ClubDataLoader()
    path = Path(file_path)

    if path.suffix.lower() in [".xlsx", ".xls"]:
        return loader.load_clubs_from_excel(path)
    elif path.suffix.lower() == ".json":
        return loader.load_clubs_from_json(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def load_pro_player_data(file_path: str | Path) -> list[ProPlayerData]:
    """Convenience function to load professional player data.

    Args:
        file_path: Path to the Excel file

    Returns:
        List of ProPlayerData objects
    """
    loader = ClubDataLoader()
    return loader.load_player_data_from_excel(file_path)
