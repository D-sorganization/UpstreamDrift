"""Ground truth validation data for golf ball flight models.

This module contains reference data from published sources that can be used
to validate and calibrate flight model trajectory predictions.

Sources:
1. TrackMan PGA Tour Averages (2023-2024) - trackman.com
2. Golf Monthly - PGA Tour Statistics
3. USGA/R&A Equipment Research

IMPORTANT: These are aggregate statistics, not individual shot data.
Individual shot trajectories have significant variance around these means.
"""

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar  # noqa: F401 - Reserved for future use


class DataSource(Enum):
    """Source of validation data."""

    TRACKMAN_PGA_TOUR = "trackman_pga_tour"  # Published PGA Tour averages
    GOLF_MONTHLY = "golf_monthly"  # Golf Monthly article compilations
    USGA_RESEARCH = "usga_research"  # USGA equipment research
    KAGGLE_GARMIN = "kaggle_garmin"  # Kaggle Garmin R50 dataset


@dataclass(frozen=True)
class ValidationDataPoint:
    """A single ground truth data point for model validation.

    All values are averages from published sources. Units are SI
    for internal consistency, with display converters provided.
    """

    club: str  # Club name (e.g., "Driver", "7-Iron")

    # Input launch conditions
    ball_speed_mps: float  # Ball speed [m/s]
    launch_angle_deg: float  # Launch angle [degrees]
    spin_rate_rpm: float  # Backspin [rpm]

    # Expected output (ground truth)
    carry_distance_m: float  # Carry distance [meters]
    max_height_m: float | None  # Apex height [meters] (if available)
    flight_time_s: float | None  # Flight time [seconds] (if available)
    landing_angle_deg: float | None  # Descent angle [degrees] (if available)

    # Metadata
    source: DataSource
    year: int  # Year of data collection
    notes: str = ""  # Additional context

    # Tolerance for validation (± percentage)
    carry_tolerance_pct: float = 5.0  # ±5% considered passing

    @property
    def ball_speed_mph(self) -> float:
        """Ball speed in mph."""
        return self.ball_speed_mps * 2.23694

    @property
    def carry_distance_yards(self) -> float:
        """Carry distance in yards."""
        return self.carry_distance_m * 1.09361

    def is_valid_carry(self, predicted_m: float) -> bool:
        """Check if predicted carry is within tolerance of ground truth.

        Args:
            predicted_m: Predicted carry distance in meters

        Returns:
            True if within tolerance
        """
        lower = self.carry_distance_m * (1 - self.carry_tolerance_pct / 100)
        upper = self.carry_distance_m * (1 + self.carry_tolerance_pct / 100)
        return lower <= predicted_m <= upper


# =============================================================================
# PGA Tour TrackMan Averages (2023-2024)
# Source: trackman.com, Golf Monthly
# =============================================================================

# Unit conversions
MPH_TO_MPS = 0.44704
YARDS_TO_METERS = 0.9144

PGA_TOUR_2024: list[ValidationDataPoint] = [
    # Driver
    ValidationDataPoint(
        club="Driver",
        ball_speed_mps=174.36 * MPH_TO_MPS,  # 174.36 mph average
        launch_angle_deg=10.4,
        spin_rate_rpm=2545,
        carry_distance_m=282 * YARDS_TO_METERS,  # 282 yards
        max_height_m=32.0,  # ~105 ft typical
        flight_time_s=6.5,
        landing_angle_deg=38.0,
        source=DataSource.TRACKMAN_PGA_TOUR,
        year=2024,
        notes="PGA Tour average 2024, trackman.com",
    ),
    # 3-Wood
    ValidationDataPoint(
        club="3-Wood",
        ball_speed_mps=158 * MPH_TO_MPS,  # ~158 mph
        launch_angle_deg=9.5,
        spin_rate_rpm=3655,
        carry_distance_m=249 * YARDS_TO_METERS,  # 249 yards
        max_height_m=28.0,
        flight_time_s=None,
        landing_angle_deg=None,
        source=DataSource.TRACKMAN_PGA_TOUR,
        year=2024,
        notes="PGA Tour average, Golf Monthly",
    ),
    # 5-Iron
    ValidationDataPoint(
        club="5-Iron",
        ball_speed_mps=135 * MPH_TO_MPS,  # 135 mph
        launch_angle_deg=12.1,
        spin_rate_rpm=5361,
        carry_distance_m=199 * YARDS_TO_METERS,  # 199 yards
        max_height_m=30.0,
        flight_time_s=None,
        landing_angle_deg=47.0,
        source=DataSource.TRACKMAN_PGA_TOUR,
        year=2024,
        notes="PGA Tour average",
    ),
    # 7-Iron
    ValidationDataPoint(
        club="7-Iron",
        ball_speed_mps=123 * MPH_TO_MPS,  # 123 mph
        launch_angle_deg=16.3,
        spin_rate_rpm=7097,
        carry_distance_m=176 * YARDS_TO_METERS,  # 176 yards
        max_height_m=31.0,
        flight_time_s=5.8,
        landing_angle_deg=50.0,
        source=DataSource.TRACKMAN_PGA_TOUR,
        year=2024,
        notes="PGA Tour average",
    ),
    # Pitching Wedge
    ValidationDataPoint(
        club="PW",
        ball_speed_mps=102 * MPH_TO_MPS,  # ~102 mph estimated
        launch_angle_deg=24.2,
        spin_rate_rpm=9304,
        carry_distance_m=142 * YARDS_TO_METERS,  # 142 yards
        max_height_m=29.0,
        flight_time_s=5.5,
        landing_angle_deg=52.0,
        source=DataSource.TRACKMAN_PGA_TOUR,
        year=2024,
        notes="PGA Tour average",
    ),
]


# =============================================================================
# Amateur golfer averages (for comparison)
# =============================================================================

AMATEUR_AVERAGES: list[ValidationDataPoint] = [
    # Average male amateur driver
    ValidationDataPoint(
        club="Driver (Amateur)",
        ball_speed_mps=132 * MPH_TO_MPS,  # ~132 mph
        launch_angle_deg=12.6,
        spin_rate_rpm=3275,
        carry_distance_m=208 * YARDS_TO_METERS,  # ~208 yards
        max_height_m=24.0,
        flight_time_s=5.0,
        landing_angle_deg=None,
        source=DataSource.TRACKMAN_PGA_TOUR,
        year=2023,
        notes="10-handicap male amateur average",
        carry_tolerance_pct=8.0,  # Higher tolerance for amateur variance
    ),
    # Amateur 7-Iron
    ValidationDataPoint(
        club="7-Iron (Amateur)",
        ball_speed_mps=105 * MPH_TO_MPS,  # ~105 mph
        launch_angle_deg=19.0,
        spin_rate_rpm=6500,
        carry_distance_m=140 * YARDS_TO_METERS,  # ~140 yards
        max_height_m=25.0,
        flight_time_s=None,
        landing_angle_deg=None,
        source=DataSource.TRACKMAN_PGA_TOUR,
        year=2023,
        notes="10-handicap male amateur average",
        carry_tolerance_pct=8.0,
    ),
]


# =============================================================================
# Combined reference data
# =============================================================================

ALL_VALIDATION_DATA: list[ValidationDataPoint] = PGA_TOUR_2024 + AMATEUR_AVERAGES


def get_validation_data_for_club(club_name: str) -> list[ValidationDataPoint]:
    """Get all validation data points for a given club.

    Args:
        club_name: Club name (case-insensitive partial match)

    Returns:
        List of matching validation data points
    """
    club_lower = club_name.lower()
    return [d for d in ALL_VALIDATION_DATA if club_lower in d.club.lower()]


def print_validation_summary() -> None:
    """Print a summary of available validation data."""
    import logging

    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("Ground Truth Validation Data Summary")
    logger.info("=" * 80)
    logger.info(
        f"{'Club':<20} {'Speed (mph)':<12} {'Launch (°)':<10} "
        f"{'Spin (rpm)':<10} {'Carry (yd)':<12} {'Source'}"
    )
    logger.info("-" * 80)

    for data in ALL_VALIDATION_DATA:
        logger.info(
            f"{data.club:<20} {data.ball_speed_mph:<12.0f} "
            f"{data.launch_angle_deg:<10.1f} {data.spin_rate_rpm:<10.0f} "
            f"{data.carry_distance_yards:<12.0f} {data.source.value}"
        )

    logger.info("=" * 80)
    logger.info(f"Total data points: {len(ALL_VALIDATION_DATA)}")
    logger.info("Note: These are aggregate averages, not individual shot data.")
    logger.info("Individual shots have significant variance (typically ±5-10% carry).")
