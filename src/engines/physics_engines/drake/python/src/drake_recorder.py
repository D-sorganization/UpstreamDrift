"""Drake simulation recorder and induced acceleration analyzer.

Extracted from drake_gui_app.py to reduce monolith size.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.shared.python.logging_config import (
    configure_gui_logging,
    get_logger,
)

# Drake imports (optional)
try:
    from pydrake.all import (
        Context,
        MultibodyPlant,
    )
except ImportError:
    Context = None  # type: ignore[misc, assignment]
    MultibodyPlant = None  # type: ignore[misc, assignment]

LOGGER = get_logger(__name__)


def setup_logging() -> None:
    """Setup logging configuration."""
    configure_gui_logging()


class DrakeInducedAccelerationAnalyzer:
    """Induced acceleration analyzer for Drake."""

    def __init__(self, plant: MultibodyPlant | None) -> None:
        self.plant = plant

    def compute_components(self, context: Context) -> dict[str, np.ndarray]:
        """Compute induced acceleration components.

        Args:
            context: The plant context (with q, v set)

        Returns:
            Dict with 'gravity', 'velocity', 'total' (passive)
        """
        if self.plant is None:
            return {
                "gravity": np.array([]),
                "velocity": np.array([]),
                "total": np.array([]),
            }

        # 1. Calc Mass Matrix
        M = self.plant.CalcMassMatrix(context)

        # 2. Calc Gravity Torque
        tau_g = self.plant.CalcGravityGeneralizedForces(context)

        # 3. Calc Bias Term (Cv - tau_g)
        bias = self.plant.CalcBiasTerm(context)

        # Invert M
        try:
            Minv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            Minv = np.linalg.pinv(M)

        a_g = Minv @ tau_g

        # Force due to velocity = -Cv = -(bias + tau_g)
        a_v = Minv @ (-(bias + tau_g))

        return {"gravity": a_g, "velocity": a_v, "total": a_g + a_v}

    def compute_counterfactuals(self, context: Context) -> dict[str, np.ndarray]:
        """Compute ZTCF and ZVCF."""
        if self.plant is None:
            return {}

        # ZTCF (Zero Torque Accel): a = -M^-1 (C + G).
        # We assume zero torque applied.
        # This is essentially the passive dynamics accel.
        # We already computed this as 'total' in compute_components if we sum a_g + a_v.
        # Or specifically: M a + Cv - tau_g = 0 => M a = tau_g - Cv = -bias.
        # So a_ztcf = -M^-1 * bias.

        M = self.plant.CalcMassMatrix(context)
        bias = self.plant.CalcBiasTerm(context)

        try:
            Minv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            Minv = np.linalg.pinv(M)

        ztcf_accel = Minv @ (-bias)

        # ZVCF (Zero Velocity Torque):
        # M a + G = tau. If v=0, C=0.
        # If we hold position (a=0, v=0), tau = G.
        # tau_g = CalcGravityGeneralizedForces.
        # Wait, equation is M vdot + Cv - tau_g = tau.
        # If v=0 => Cv=0. If a=0 => M vdot = 0.
        # So -tau_g = tau => tau = -tau_g?
        # Drake defines tau_g as forces on RHS.
        # So tau_holding = -tau_g.

        tau_g = self.plant.CalcGravityGeneralizedForces(context)
        zvcf_torque = -tau_g

        return {"ztcf_accel": ztcf_accel, "zvcf_torque": zvcf_torque}

    def compute_specific_control(self, context: Context, tau: np.ndarray) -> np.ndarray:
        """Compute induced acceleration for a specific control vector.

        Note:
            This method calculates the acceleration induced solely by the provided
            torque vector `tau`.
            It solves M * a = tau. If `tau` represents a unit torque
            (e.g., [0, 1, 0]), the result is the sensitivity of acceleration
            to that specific actuator.
        """
        if self.plant is None:
            return np.array([])

        # M * a = tau
        M = self.plant.CalcMassMatrix(context)
        try:
            Minv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            Minv = np.linalg.pinv(M)

        return np.array(Minv @ tau)  # type: ignore[no-any-return]


class DrakeRecorder:
    """Records simulation data for analysis.

    Implements RecorderInterface for LivePlotWidget.
    """

    def __init__(self, engine: Any = None) -> None:
        """Initialize recorder.

        Args:
            engine: Optional reference to the physics engine/app wrapper.
        """
        self.reset()
        self.engine = engine  # Reference for joint names
        self.analysis_config: dict[str, Any] = {}

    def reset(self) -> None:
        self.times: list[float] = []
        self.q_history: list[np.ndarray] = []
        self.v_history: list[np.ndarray] = []
        self.club_head_pos_history: list[np.ndarray] = []
        self.com_position_history: list[np.ndarray] = []
        self.angular_momentum_history: list[np.ndarray] = []
        self.ground_forces_history: list[np.ndarray] = []
        self.cop_position_history: list[np.ndarray] = []
        # Store computed metrics
        self.induced_accelerations: dict[str, list[np.ndarray]] = {}
        self.counterfactuals: dict[str, list[np.ndarray]] = {}
        self.is_recording = False

    def start(self) -> None:
        self.reset()
        self.is_recording = True

    def stop(self) -> None:
        self.is_recording = False

    def record(
        self,
        t: float,
        q: np.ndarray,
        v: np.ndarray,
        club_pos: np.ndarray | None = None,
        com_pos: np.ndarray | None = None,
        angular_momentum: np.ndarray | None = None,
    ) -> None:
        if not self.is_recording:
            return
        self.times.append(t)
        self.q_history.append(q.copy())
        self.v_history.append(v.copy())
        if club_pos is not None:
            self.club_head_pos_history.append(club_pos.copy())
        else:
            self.club_head_pos_history.append(np.zeros(3))

        if com_pos is not None:
            self.com_position_history.append(com_pos.copy())
        else:
            self.com_position_history.append(np.zeros(3))

        if angular_momentum is not None:
            self.angular_momentum_history.append(angular_momentum.copy())
        else:
            self.angular_momentum_history.append(np.zeros(3))

        # Placeholders for now
        self.ground_forces_history.append(np.zeros(3))
        self.cop_position_history.append(np.zeros(3))

    def set_analysis_config(self, config: dict[str, Any]) -> None:
        """Update analysis configuration."""
        self.analysis_config = config

    def get_time_series(self, field_name: str) -> tuple[np.ndarray, np.ndarray | list]:
        """Implement RecorderInterface."""
        times = np.array(self.times)
        if field_name == "club_head_position":
            return times, np.array(self.club_head_pos_history)
        if field_name == "joint_positions":
            return times, np.array(self.q_history)
        if field_name == "joint_velocities":
            return times, np.array(self.v_history)

        # Counterfactuals via standard get_time_series if stored
        if field_name == "ztcf_accel":
            return self.get_counterfactual_series("ztcf_accel")
        if field_name == "zvcf_accel":
            # Drake computes zvcf_torque in example logic,
            # but let's assume we store accels if available
            return self.get_counterfactual_series("zvcf_torque")

        # Fallback
        return times, []

    def get_induced_acceleration_series(
        self, source_name: str | int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get induced accelerations."""
        if (
            isinstance(source_name, int)
            or source_name not in self.induced_accelerations
        ):
            # If int, maybe we have it stored by int key?
            # Or map int to name if possible?
            # For now, return empty if not found.
            key = str(source_name)
            if key in self.induced_accelerations:
                # If stored by int key or str(int) key
                vals = self.induced_accelerations[key]  # type: ignore
                times = np.array(self.times)
                min_len = min(len(vals), len(times))
                return times[:min_len], np.array(vals[:min_len])

            is_int_key = isinstance(source_name, int)
            if is_int_key and source_name in self.induced_accelerations:
                # Check for int key (less common in json but possible in dict)
                vals = self.induced_accelerations[source_name]  # type: ignore
                times = np.array(self.times)
                min_len = min(len(vals), len(times))
                return times[:min_len], np.array(vals[:min_len])

            return np.array([]), np.array([])

        times = np.array(self.times)
        # Ensure alignment
        vals = self.induced_accelerations[source_name]
        if len(vals) != len(times):
            # Truncate to match
            min_len = min(len(vals), len(times))
            return times[:min_len], np.array(vals[:min_len])

        return times, np.array(vals)

    def get_counterfactual_series(self, cf_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Get counterfactual data."""
        if cf_name not in self.counterfactuals:
            return np.array([]), np.array([])

        times = np.array(self.times)
        vals = self.counterfactuals[cf_name]

        if len(vals) != len(times):
            min_len = min(len(vals), len(times))
            return times[:min_len], np.array(vals[:min_len])

        return times, np.array(vals)

    def export_to_dict(self) -> dict[str, Any]:
        """Export all recorded data to a dictionary."""
        data: dict[str, Any] = {"times": np.array(self.times)}

        def add_series(target: dict, name: str, arr_list: list) -> None:
            if not arr_list:
                return
            arr = np.array(arr_list)
            if len(arr) != len(self.times):
                # Simple alignment
                min_len = min(len(arr), len(self.times))
                arr = arr[:min_len]

            target[name] = arr

        add_series(data, "joint_positions", self.q_history)
        add_series(data, "joint_velocities", self.v_history)
        add_series(data, "club_head_position", self.club_head_pos_history)
        add_series(data, "com_position", self.com_position_history)
        add_series(data, "angular_momentum", self.angular_momentum_history)
        add_series(data, "ground_forces", self.ground_forces_history)
        add_series(data, "cop_position", self.cop_position_history)

        # Export Induced Accel
        if self.induced_accelerations:
            data["induced_accelerations"] = {}
            for k, v in self.induced_accelerations.items():
                add_series(data["induced_accelerations"], str(k), v)

        # Export Counterfactuals
        if self.counterfactuals:
            data["counterfactuals"] = {}
            for k, v in self.counterfactuals.items():
                add_series(data["counterfactuals"], str(k), v)

        return data
