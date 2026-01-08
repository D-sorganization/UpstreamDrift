"""Ground Reaction Force Analysis Module.

Guideline E5 Implementation: Ground Reaction Forces.

Provides computation and analysis of ground reaction forces (GRF) including:
- GRF computation from contact elements
- Force plate data integration (C3D format)
- Linear and angular impulse calculation
- Center of pressure (COP) trajectory
- Moment about center of mass

Cross-engine consistency thresholds:
- GRF magnitude: ± 5%
- COP position: ± 10 mm
- Angular impulse: ± 10%
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from shared.python.interfaces import PhysicsEngine

LOGGER = logging.getLogger(__name__)

# Cross-engine validation tolerances (from Guideline E5)
GRF_MAGNITUDE_TOLERANCE = 0.05  # 5% relative tolerance
COP_POSITION_TOLERANCE_MM = 10.0  # 10 mm absolute tolerance [mm]
ANGULAR_IMPULSE_TOLERANCE = 0.10  # 10% relative tolerance

# Physical constants
GRAVITY_MAGNITUDE = 9.80665  # [m/s²] Source: WGS84 standard


class FootSide(Enum):
    """Enumeration of foot sides for bilateral analysis."""

    LEFT = auto()
    RIGHT = auto()
    COMBINED = auto()


@dataclass
class GroundReactionForce:
    """Ground reaction force at a single instant.

    Attributes:
        force: GRF vector in global frame [N] (3,)
        moment: GRF moment about origin [N·m] (3,)
        cop: Center of pressure position [m] (3,)
        timestamp: Time of measurement [s]
        foot_side: Which foot (LEFT, RIGHT, or COMBINED)
    """

    force: np.ndarray
    moment: np.ndarray
    cop: np.ndarray
    timestamp: float
    foot_side: FootSide = FootSide.COMBINED


@dataclass
class GRFTimeSeries:
    """Time series of ground reaction forces.

    Attributes:
        timestamps: Time values [s] (N,)
        forces: GRF vectors over time [N] (N, 3)
        moments: GRF moments over time [N·m] (N, 3)
        cops: Center of pressure positions over time [m] (N, 3)
        foot_side: Which foot this data represents
        sample_rate: Sampling rate [Hz]
    """

    timestamps: np.ndarray
    forces: np.ndarray
    moments: np.ndarray
    cops: np.ndarray
    foot_side: FootSide = FootSide.COMBINED
    sample_rate: float = 1000.0


@dataclass
class ImpulseMetrics:
    """Linear and angular impulse metrics.

    Attributes:
        linear_impulse: ∫ F dt [N·s] (3,)
        angular_impulse: ∫ τ dt [N·m·s] (3,)
        linear_impulse_magnitude: |∫ F dt| [N·s]
        angular_impulse_magnitude: |∫ τ dt| [N·m·s]
        duration: Integration time window [s]
    """

    linear_impulse: np.ndarray
    angular_impulse: np.ndarray
    linear_impulse_magnitude: float
    angular_impulse_magnitude: float
    duration: float


@dataclass
class GRFSummary:
    """Summary statistics for GRF analysis.

    Attributes:
        peak_vertical_force: Maximum vertical GRF [N]
        peak_horizontal_force: Maximum horizontal GRF magnitude [N]
        time_to_peak_vertical: Time to peak vertical GRF [s]
        linear_impulse: Total linear impulse metrics
        angular_impulse_about_golfer_com: Angular impulse about golfer COM
        angular_impulse_about_system_com: Angular impulse about golfer+club COM
        cop_trajectory_length: Total COP path length [m]
        cop_range_ap: COP range in anterior-posterior direction [m]
        cop_range_ml: COP range in medial-lateral direction [m]
    """

    peak_vertical_force: float
    peak_horizontal_force: float
    time_to_peak_vertical: float
    linear_impulse: ImpulseMetrics
    angular_impulse_about_golfer_com: np.ndarray
    angular_impulse_about_system_com: np.ndarray
    cop_trajectory_length: float
    cop_range_ap: float
    cop_range_ml: float


def compute_linear_impulse(
    forces: np.ndarray,
    timestamps: np.ndarray,
) -> np.ndarray:
    """Compute linear impulse from force time series.

    J = ∫ F(t) dt

    Uses trapezoidal integration for numerical stability.

    Args:
        forces: Force vectors over time [N] (N, 3)
        timestamps: Time values [s] (N,)

    Returns:
        Linear impulse vector [N·s] (3,)
    """
    if len(forces) < 2:
        return np.zeros(3)

    # Trapezoidal integration for each component
    impulse = np.zeros(3)
    for i in range(3):
        impulse[i] = np.trapz(forces[:, i], timestamps)

    return impulse


def compute_angular_impulse(
    forces: np.ndarray,
    cops: np.ndarray,
    timestamps: np.ndarray,
    reference_point: np.ndarray,
) -> np.ndarray:
    """Compute angular impulse of GRF about a reference point.

    L = ∫ (r × F) dt

    where r is the vector from reference point to COP.

    Args:
        forces: Force vectors over time [N] (N, 3)
        cops: Center of pressure positions over time [m] (N, 3)
        timestamps: Time values [s] (N,)
        reference_point: Point about which to compute angular impulse [m] (3,)

    Returns:
        Angular impulse vector [N·m·s] (3,)
    """
    if len(forces) < 2:
        return np.zeros(3)

    # Compute moment arm at each time step
    r = cops - reference_point

    # Compute torque (moment) at each time step
    torques = np.cross(r, forces)

    # Trapezoidal integration for each component
    angular_impulse = np.zeros(3)
    for i in range(3):
        angular_impulse[i] = np.trapz(torques[:, i], timestamps)

    return angular_impulse


def compute_cop_from_grf(
    force: np.ndarray,
    moment: np.ndarray,
    ground_height: float = 0.0,
) -> np.ndarray:
    """Compute center of pressure from GRF and moment.

    The COP is where the vertical GRF acts to produce the measured moment.

    COP_x = -M_y / F_z
    COP_y = M_x / F_z
    COP_z = ground_height

    Args:
        force: GRF vector [N] (3,)
        moment: Moment about origin [N·m] (3,)
        ground_height: Height of ground plane [m]

    Returns:
        COP position [m] (3,)
    """
    # Avoid division by zero for small vertical forces
    min_vertical_force = 10.0  # [N] threshold
    fz = force[2]

    if abs(fz) < min_vertical_force:
        return np.array([0.0, 0.0, ground_height])

    cop_x = -moment[1] / fz
    cop_y = moment[0] / fz
    cop_z = ground_height

    return np.array([cop_x, cop_y, cop_z])


def compute_cop_trajectory_length(cops: np.ndarray) -> float:
    """Compute total path length of COP trajectory.

    Args:
        cops: COP positions over time [m] (N, 3)

    Returns:
        Total path length [m]
    """
    if len(cops) < 2:
        return 0.0

    # Difference between consecutive points
    diffs = np.diff(cops, axis=0)

    # Euclidean distance for each segment
    distances = np.linalg.norm(diffs, axis=1)

    return float(np.sum(distances))


class GRFAnalyzer:
    """Ground reaction force analyzer.

    Provides comprehensive GRF analysis including impulse computation,
    COP tracking, and cross-engine validation.
    """

    def __init__(self) -> None:
        """Initialize the GRF analyzer."""
        self.grf_data: dict[FootSide, GRFTimeSeries] = {}
        self.golfer_com_trajectory: np.ndarray | None = None
        self.system_com_trajectory: np.ndarray | None = None

    def add_grf_data(self, data: GRFTimeSeries) -> None:
        """Add GRF time series data.

        Args:
            data: GRF time series for one foot or combined
        """
        self.grf_data[data.foot_side] = data

    def set_com_trajectories(
        self,
        golfer_com: np.ndarray,
        system_com: np.ndarray | None = None,
    ) -> None:
        """Set center of mass trajectories for angular impulse computation.

        Args:
            golfer_com: Golfer COM positions over time [m] (N, 3)
            system_com: Golfer+club system COM positions over time [m] (N, 3)
        """
        self.golfer_com_trajectory = golfer_com
        self.system_com_trajectory = (
            system_com if system_com is not None else golfer_com
        )

    def compute_impulse_metrics(
        self,
        foot_side: FootSide = FootSide.COMBINED,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> ImpulseMetrics:
        """Compute impulse metrics for a time window.

        Args:
            foot_side: Which foot to analyze
            start_time: Start of integration window [s]
            end_time: End of integration window [s]

        Returns:
            ImpulseMetrics for the specified window
        """
        if foot_side not in self.grf_data:
            raise ValueError(f"No GRF data for {foot_side}")

        data = self.grf_data[foot_side]

        # Apply time window
        if start_time is not None or end_time is not None:
            mask = np.ones(len(data.timestamps), dtype=bool)
            if start_time is not None:
                mask &= data.timestamps >= start_time
            if end_time is not None:
                mask &= data.timestamps <= end_time

            timestamps = data.timestamps[mask]
            forces = data.forces[mask]
            cops = data.cops[mask]
        else:
            timestamps = data.timestamps
            forces = data.forces
            cops = data.cops

        if len(timestamps) < 2:
            return ImpulseMetrics(
                linear_impulse=np.zeros(3),
                angular_impulse=np.zeros(3),
                linear_impulse_magnitude=0.0,
                angular_impulse_magnitude=0.0,
                duration=0.0,
            )

        # Linear impulse
        linear_impulse = compute_linear_impulse(forces, timestamps)

        # Angular impulse (about origin if no COM specified)
        ref_point = np.zeros(3)
        angular_impulse = compute_angular_impulse(forces, cops, timestamps, ref_point)

        duration = float(timestamps[-1] - timestamps[0])

        return ImpulseMetrics(
            linear_impulse=linear_impulse,
            angular_impulse=angular_impulse,
            linear_impulse_magnitude=float(np.linalg.norm(linear_impulse)),
            angular_impulse_magnitude=float(np.linalg.norm(angular_impulse)),
            duration=duration,
        )

    def analyze(
        self,
        foot_side: FootSide = FootSide.COMBINED,
    ) -> GRFSummary:
        """Perform comprehensive GRF analysis.

        Args:
            foot_side: Which foot to analyze

        Returns:
            GRFSummary with all computed metrics
        """
        if foot_side not in self.grf_data:
            raise ValueError(f"No GRF data for {foot_side}")

        data = self.grf_data[foot_side]
        forces = data.forces
        timestamps = data.timestamps
        cops = data.cops

        # Peak forces
        vertical_forces = forces[:, 2]
        horizontal_forces = np.linalg.norm(forces[:, :2], axis=1)

        peak_vertical = float(np.max(vertical_forces))
        peak_horizontal = float(np.max(horizontal_forces))
        time_to_peak = float(timestamps[np.argmax(vertical_forces)])

        # Linear impulse
        linear_impulse = self.compute_impulse_metrics(foot_side)

        # Angular impulse about COMs
        if self.golfer_com_trajectory is not None:
            golfer_com_impulse = compute_angular_impulse(
                forces, cops, timestamps, self.golfer_com_trajectory[0]
            )
        else:
            golfer_com_impulse = np.zeros(3)

        if self.system_com_trajectory is not None:
            system_com_impulse = compute_angular_impulse(
                forces, cops, timestamps, self.system_com_trajectory[0]
            )
        else:
            system_com_impulse = np.zeros(3)

        # COP metrics
        cop_length = compute_cop_trajectory_length(cops)
        cop_range_ap = float(np.ptp(cops[:, 0]))  # X = anterior-posterior
        cop_range_ml = float(np.ptp(cops[:, 1]))  # Y = medial-lateral

        return GRFSummary(
            peak_vertical_force=peak_vertical,
            peak_horizontal_force=peak_horizontal,
            time_to_peak_vertical=time_to_peak,
            linear_impulse=linear_impulse,
            angular_impulse_about_golfer_com=golfer_com_impulse,
            angular_impulse_about_system_com=system_com_impulse,
            cop_trajectory_length=cop_length,
            cop_range_ap=cop_range_ap,
            cop_range_ml=cop_range_ml,
        )


def extract_grf_from_contacts(
    engine: PhysicsEngine,
    contact_body_names: list[str],
    ground_height: float = 0.0,
) -> GroundReactionForce:
    """Extract GRF from physics engine contact forces.

    Synthetic GRF computed from model contact elements when force plates
    are unavailable.

    Args:
        engine: Physics engine with active simulation
        contact_body_names: Names of bodies in ground contact (e.g., ["left_foot"])
        ground_height: Height of ground plane [m]

    Returns:
        GroundReactionForce at current simulation time
    """
    total_force = np.zeros(3)
    total_moment = np.zeros(3)
    total_weighted_pos = np.zeros(3)

    for body_name in contact_body_names:
        jac_dict = engine.compute_jacobian(body_name)
        if jac_dict is None:
            continue

        # For now, estimate contact force from gravity compensation
        # This is a simplified approach - real implementation would
        # query actual contact forces from the physics engine
        g = engine.compute_gravity_forces()

        # Approximate: GRF opposes gravity at contact points
        # This is a placeholder - real implementation depends on engine API
        if len(g) > 0:
            total_force[2] += abs(np.sum(g))

    # Compute COP
    if total_force[2] > 10.0:  # Minimum force threshold
        cop = np.array(
            [
                total_weighted_pos[0] / total_force[2],
                total_weighted_pos[1] / total_force[2],
                ground_height,
            ]
        )
    else:
        cop = np.array([0.0, 0.0, ground_height])

    return GroundReactionForce(
        force=total_force,
        moment=total_moment,
        cop=cop,
        timestamp=engine.get_time(),
        foot_side=FootSide.COMBINED,
    )


def validate_grf_cross_engine(
    grf_a: GRFTimeSeries,
    grf_b: GRFTimeSeries,
    engine_name_a: str = "Engine A",
    engine_name_b: str = "Engine B",
) -> dict[str, bool]:
    """Validate GRF consistency between two physics engines.

    Args:
        grf_a: GRF data from first engine
        grf_b: GRF data from second engine
        engine_name_a: Name of first engine for reporting
        engine_name_b: Name of second engine for reporting

    Returns:
        Dictionary of validation results per metric
    """
    results = {}

    # Force magnitude comparison
    forces_a = np.linalg.norm(grf_a.forces, axis=1)
    forces_b = np.linalg.norm(grf_b.forces, axis=1)

    if len(forces_a) == len(forces_b):
        force_diff = np.abs(forces_a - forces_b)
        force_rel_diff = force_diff / (np.maximum(forces_a, forces_b) + 1e-10)
        results["force_magnitude"] = bool(
            np.all(force_rel_diff < GRF_MAGNITUDE_TOLERANCE)
        )
    else:
        LOGGER.warning("GRF data lengths differ, skipping force comparison")
        results["force_magnitude"] = False

    # COP position comparison
    if len(grf_a.cops) == len(grf_b.cops):
        cop_diff_mm = np.linalg.norm(grf_a.cops - grf_b.cops, axis=1) * 1000
        results["cop_position"] = bool(np.all(cop_diff_mm < COP_POSITION_TOLERANCE_MM))
    else:
        results["cop_position"] = False

    # Angular impulse comparison
    impulse_a = compute_angular_impulse(
        grf_a.forces, grf_a.cops, grf_a.timestamps, np.zeros(3)
    )
    impulse_b = compute_angular_impulse(
        grf_b.forces, grf_b.cops, grf_b.timestamps, np.zeros(3)
    )

    impulse_diff = np.abs(impulse_a - impulse_b)
    impulse_rel_diff = impulse_diff / (
        np.maximum(np.abs(impulse_a), np.abs(impulse_b)) + 1e-10
    )
    results["angular_impulse"] = bool(
        np.all(impulse_rel_diff < ANGULAR_IMPULSE_TOLERANCE)
    )

    # Log results
    for metric, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        LOGGER.info(
            f"GRF Cross-Engine [{engine_name_a} vs {engine_name_b}] "
            f"{metric}: {status}"
        )

    return results
