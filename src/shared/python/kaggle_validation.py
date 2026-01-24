"""Kaggle Golf Trajectory Dataset Loader and Validator.

Loads the Garmin R50 launch monitor data from Kaggle for validation
of ball flight models against real-world measurements.

Dataset source: Kaggle "Golf Swing and Trajectory Data" (MIT License)
Columns:
- Ball Speed (mph), Launch Angle (deg), Spin Rate (rpm)
- Carry Distance (yards), Apex Height (ft), Total Distance (yards)
- Atmospheric: Air Density (g/L), Temperature (F), Air Pressure (kPA)
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.shared.python.logging_config import get_logger
from src.shared.python.physics_constants import (
    FEET_TO_METERS,
    MPH_TO_MPS,
    YARDS_TO_METERS,
)

logger = get_logger(__name__)


@dataclass
class ShotRecord:
    """Single shot from the Kaggle dataset."""

    # Launch conditions
    ball_speed_mph: float
    launch_angle_deg: float
    launch_direction_deg: float
    backspin_rpm: float
    sidespin_rpm: float
    spin_rate_rpm: float
    spin_axis_deg: float

    # Measured outcomes
    carry_distance_yards: float
    total_distance_yards: float
    apex_height_ft: float

    # Atmospheric
    air_density_g_l: float
    temperature_f: float
    air_pressure_kpa: float

    @property
    def ball_speed_mps(self) -> float:
        """Ball speed in m/s."""
        return self.ball_speed_mph * MPH_TO_MPS

    @property
    def carry_distance_m(self) -> float:
        """Carry distance in meters."""
        return self.carry_distance_yards * YARDS_TO_METERS

    @property
    def apex_height_m(self) -> float:
        """Apex height in meters."""
        return self.apex_height_ft * FEET_TO_METERS

    @property
    def air_density_kg_m3(self) -> float:
        """Air density in kg/m³ (from g/L, which is same value)."""
        return self.air_density_g_l


def load_kaggle_dataset(
    csv_path: Path | str | None = None,
) -> pd.DataFrame:
    """Load the Kaggle golf trajectory dataset.

    Args:
        csv_path: Path to CSV file. Defaults to data/golf_trajectory.csv

    Returns:
        DataFrame with shot data
    """
    if csv_path is None:
        # Default path relative to this file
        csv_path = Path(__file__).parent.parent.parent / "data" / "golf_trajectory.csv"

    df = pd.read_csv(csv_path)

    # Rename columns for easier access
    column_map = {
        "Club Speed (mph)": "club_speed_mph",
        "Attack Angle (deg)": "attack_angle_deg",
        "Club Path (deg)": "club_path_deg",
        "Club Face (deg)": "club_face_deg",
        "Face to Path (deg)": "face_to_path_deg",
        "Ball Speed (mph)": "ball_speed_mph",
        "Smash Factor": "smash_factor",
        "Launch Angle (deg)": "launch_angle_deg",
        "Launch Direction (deg)": "launch_direction_deg",
        "Backspin (rpm)": "backspin_rpm",
        "Sidespin (rpm)": "sidespin_rpm",
        "Spin Rate (rpm)": "spin_rate_rpm",
        "Spin Rate Type": "spin_rate_type",
        "Spin Axis (deg)": "spin_axis_deg",
        "Apex Height (ft)": "apex_height_ft",
        "Total Distance (yards)": "total_distance_yards",
        "Total Deviation Angle (deg)": "total_deviation_angle_deg",
        "Total Deviation Distance (yards)": "total_deviation_distance_yards",
        "Air Density (g/L)": "air_density_g_l",
        "Temperature (F)": "temperature_f",
        "Air Pressure (kPA)": "air_pressure_kpa",
        "Carry Distance (yards)": "carry_distance_yards",
        "Carry Deviation Angle (deg)": "carry_deviation_angle_deg",
        "Carry Deviation Distance (yards)": "carry_deviation_distance_yards",
    }
    df = df.rename(columns=column_map)

    logger.info(f"Loaded {len(df)} shots from Kaggle dataset")

    return df


def get_clean_shots(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to shots with complete launch and outcome data.

    Args:
        df: Raw DataFrame from load_kaggle_dataset

    Returns:
        Filtered DataFrame with only complete records
    """
    required_cols = [
        "ball_speed_mph",
        "launch_angle_deg",
        "spin_rate_rpm",
        "carry_distance_yards",
        "apex_height_ft",
    ]

    clean = df.dropna(subset=required_cols)

    # Filter out unreasonable values
    clean = clean[clean["ball_speed_mph"] > 50]  # At least 50 mph
    clean = clean[clean["ball_speed_mph"] < 200]  # Less than 200 mph
    clean = clean[clean["launch_angle_deg"] > -5]  # Not too negative
    clean = clean[clean["launch_angle_deg"] < 50]  # Not too high
    clean = clean[clean["carry_distance_yards"] > 20]  # At least 20 yards
    clean = clean[clean["carry_distance_yards"] < 400]  # Less than 400 yards

    logger.info(f"Filtered to {len(clean)} clean shots")

    return clean


def get_dataset_statistics(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Calculate summary statistics for the dataset.

    Args:
        df: DataFrame with shot data

    Returns:
        Dictionary of statistics by column
    """
    stats = {}
    numeric_cols = [
        "ball_speed_mph",
        "launch_angle_deg",
        "spin_rate_rpm",
        "carry_distance_yards",
        "apex_height_ft",
    ]

    for col in numeric_cols:
        if col in df.columns:
            stats[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "median": float(df[col].median()),
            }

    return stats


def validate_model_against_dataset(
    df: pd.DataFrame,
    model_func: Callable[[float, float, float], float],
    sample_size: int = 100,
) -> dict[str, float]:
    """Validate a flight model against the Kaggle dataset.

    Args:
        df: DataFrame with shot data
        model_func: Function that takes (ball_speed_mph, launch_angle_deg, spin_rpm)
                   and returns predicted carry in yards
        sample_size: Number of shots to sample for validation

    Returns:
        Validation metrics (MAE, RMSE, R², bias)
    """

    clean = get_clean_shots(df)

    if len(clean) > sample_size:
        sample = clean.sample(sample_size, random_state=42)
    else:
        sample = clean

    predictions_list: list[float] = []
    actuals_list: list[float] = []

    for _, row in sample.iterrows():
        try:
            pred = model_func(
                row["ball_speed_mph"],
                row["launch_angle_deg"],
                row["spin_rate_rpm"],
            )
            predictions_list.append(pred)
            actuals_list.append(float(row["carry_distance_yards"]))
        except Exception as e:
            logger.warning(f"Prediction failed: {e}")

    predictions = np.array(predictions_list)
    actuals = np.array(actuals_list)

    # Calculate metrics
    errors = predictions - actuals
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    bias = float(np.mean(errors))

    # R² (coefficient of determination)
    ss_res = float(np.sum(errors**2))
    ss_tot = float(np.sum((actuals - np.mean(actuals)) ** 2))
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    # MAPE (mean absolute percentage error)
    mape = float(np.mean(np.abs(errors / actuals) * 100))

    return {
        "mae_yards": mae,
        "rmse_yards": rmse,
        "bias_yards": bias,
        "r2": r2,
        "mape_percent": mape,
        "n_samples": len(actuals),
    }


def compare_all_models_to_dataset(
    df: pd.DataFrame | None = None,
    sample_size: int = 100,
) -> dict[str, dict[str, float]]:
    """Compare all flight models against the Kaggle dataset.

    Args:
        df: DataFrame with shot data (loads default if None)
        sample_size: Number of shots to sample

    Returns:
        Dictionary mapping model name to validation metrics
    """
    import sys
    from pathlib import Path

    # Import flight models
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared" / "python"))

    from flight_models import (
        BallFlightModel,
        FlightModelRegistry,
        UnifiedLaunchConditions,
    )

    if df is None:
        df = load_kaggle_dataset()

    results = {}

    for model in FlightModelRegistry.get_all_models():
        # Capture model in closure via default argument
        def make_model_func(
            m: "BallFlightModel",
        ) -> Callable[[float, float, float], float]:
            def model_func(speed: float, angle: float, spin: float) -> float:
                launch = UnifiedLaunchConditions.from_imperial(
                    ball_speed_mph=speed,
                    launch_angle_deg=angle,
                    spin_rate_rpm=spin,
                )
                result = m.simulate(launch)
                return float(result.carry_distance * 1.09361)  # Convert to yards

            return model_func

        model_fn = make_model_func(model)

        try:
            metrics = validate_model_against_dataset(df, model_fn, sample_size)
            results[model.name] = metrics
            logger.info(
                f"{model.name}: MAE={metrics['mae_yards']:.1f} yd, "
                f"RMSE={metrics['rmse_yards']:.1f} yd, R²={metrics['r2']:.3f}"
            )
        except Exception as e:
            logger.exception(f"Validation failed for {model.name}: {e}")

    return results


def print_validation_report(results: dict[str, dict[str, float]]) -> None:
    """Print formatted validation report.

    Args:
        results: Dictionary from compare_all_models_to_dataset
    """
    logger.info("\n" + "=" * 80)
    logger.info("Model Validation Against Kaggle Dataset")
    logger.info("=" * 80)
    logger.info(
        f"{'Model':<20} {'MAE (yd)':<10} {'RMSE (yd)':<11} "
        f"{'Bias (yd)':<11} {'R²':<8} {'MAPE %':<8}"
    )
    logger.info("-" * 80)

    # Sort by RMSE
    sorted_results = sorted(results.items(), key=lambda x: x[1]["rmse_yards"])

    for name, metrics in sorted_results:
        logger.info(
            f"{name:<20} {metrics['mae_yards']:<10.1f} "
            f"{metrics['rmse_yards']:<11.1f} {metrics['bias_yards']:<+11.1f} "
            f"{metrics['r2']:<8.3f} {metrics['mape_percent']:<8.1f}"
        )

    logger.info("=" * 80)
    logger.info(f"Validated on {sorted_results[0][1].get('n_samples', 'N/A')} shots")


if __name__ == "__main__":
    from src.shared.python.logging_config import setup_logging

    setup_logging()

    # Load and analyze dataset
    df = load_kaggle_dataset()
    clean = get_clean_shots(df)

    stats = get_dataset_statistics(clean)
    logger.info("\nDataset Statistics:")
    for col, col_stats in stats.items():
        logger.info(
            f"  {col}: mean={col_stats['mean']:.1f}, std={col_stats['std']:.1f}"
        )

    # Validate all models
    logger.info("\nValidating models against Kaggle dataset...")
    results = compare_all_models_to_dataset(clean, sample_size=100)
    print_validation_report(results)
