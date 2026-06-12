"""Headless analysis orchestrator shared by the PyQt6 and web frontends.

Extracts the analysis/plot-data logic that historically lived inside the
PyQt6 ``UnifiedDashboardWindow`` (issue #7446) so that the FastAPI layer
can serve identical results without importing Qt or matplotlib.

The :class:`AnalysisOrchestrator` returns structured, JSON-serializable
:class:`~src.shared.python.analysis.plot_data.PlotData` /
:class:`~src.shared.python.analysis.plot_data.CounterfactualResult`
objects — never matplotlib figures.  Rendering remains the job of the
frontend (``GolfSwingPlotter`` for PyQt6, Plotly/SVG for the web app).

This module must stay importable with no Qt or matplotlib import at
module import time (enforced by tests).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from src.shared.python.analysis.plot_data import (
    CounterfactualResult,
    PlotData,
    PlotSeries,
)
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.plot_labels import aligned_joint_label, joint_name

logger = get_logger(__name__)

__all__ = [
    "COUNTERFACTUAL_KIND_REQUIREMENTS",
    "AnalysisOrchestrator",
    "RecorderLike",
    "get_plot_data",
    "supported_counterfactual_kinds",
]

#: m/s -> mph conversion used by the dashboard club-head-speed plot.
MPS_TO_MPH = 2.23694

#: Counterfactual kinds backed by ``recorder.get_counterfactual_series``.
COUNTERFACTUAL_KINDS = frozenset({"ztcf", "zvcf"})

#: Induced-acceleration kinds backed by
#: ``recorder.get_induced_acceleration_series``.
INDUCED_ACCELERATION_KINDS = frozenset({"gravity", "drift", "control", "total"})

#: Engine methods each counterfactual kind requires (issue #7450).
#:
#: Single source of truth for capability gating: the API only offers a
#: kind when the active engine implements every required method, and the
#: desktop post-hoc path (``GenericPhysicsRecorder.compute_analysis_post_hoc``)
#: calls exactly these methods.  Keep this conservative — a kind listed as
#: supported must actually be computable end-to-end.
COUNTERFACTUAL_KIND_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "ztcf": ("compute_ztcf",),
    "zvcf": ("compute_zvcf",),
    "gravity": ("compute_drift_acceleration",),
    "drift": ("compute_drift_acceleration",),
    "control": ("compute_control_acceleration",),
    "total": ("compute_drift_acceleration", "compute_control_acceleration"),
}

#: Methods the shared post-hoc recompute loop needs regardless of kind
#: (it replays every recorded frame through the engine and computes every
#: kind in one pass — see ``_AnalysisMixin.compute_analysis_post_hoc``).
_POST_HOC_ENGINE_METHODS: tuple[str, ...] = (
    "set_state",
    "set_control",
    "forward",
    "compute_ztcf",
    "compute_zvcf",
    "compute_drift_acceleration",
    "compute_control_acceleration",
)


def supported_counterfactual_kinds(engine: Any) -> list[str]:
    """Return the counterfactual kinds ``engine`` can compute, sorted.

    Conservative probe used for capability gating (issue #7450): because
    the shared post-hoc recompute path computes every kind in a single
    pass over the recorded frames, a kind is only reported as supported
    when the engine implements *all* methods that path calls.  Engines
    derived from ``BasePhysicsEngine`` implement the full surface; ad-hoc
    or partial engines are gated out rather than failing mid-task.

    Args:
        engine: Physics engine instance (may be None).

    Returns:
        Sorted list of supported kind identifiers (subset of
        ``COUNTERFACTUAL_KINDS | INDUCED_ACCELERATION_KINDS``); empty when
        the engine is missing or lacks any required method.
    """
    if engine is None:
        return []
    if not all(callable(getattr(engine, m, None)) for m in _POST_HOC_ENGINE_METHODS):
        return []
    return sorted(
        kind
        for kind, methods in COUNTERFACTUAL_KIND_REQUIREMENTS.items()
        if all(callable(getattr(engine, m, None)) for m in methods)
    )


@runtime_checkable
class RecorderLike(Protocol):
    """Minimal recorder interface the orchestrator needs (Qt-free)."""

    def get_time_series(
        self, field_name: str
    ) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover - protocol
        ...


class AnalysisOrchestrator:
    """Computes structured plot data and counterfactuals from a recorder.

    The orchestrator is the single source of truth for the dashboard's
    static-plot catalogue: the PyQt6 dashboard builds its combo box from
    :data:`DASHBOARD_PLOT_LABELS`, and the API can enumerate
    :data:`PLOT_TYPES` to expose the same plots over HTTP.
    """

    #: Registry: plot-type id -> extractor method name.
    PLOT_TYPES: dict[str, str] = {
        "joint_angles": "_plot_joint_angles",
        "joint_velocities": "_plot_joint_velocities",
        "joint_torques": "_plot_joint_torques",
        "energies": "_plot_energies",
        "club_head_speed": "_plot_club_head_speed",
        "angular_momentum": "_plot_angular_momentum",
        "power_flow": "_plot_power_flow",
        "joint_power_curves": "_plot_joint_power_curves",
        "impulse_accumulation": "_plot_impulse_accumulation",
        "phase_diagram": "_plot_phase_diagram",
        "cop_trajectory": "_plot_cop_trajectory",
        "stability_diagram": "_plot_stability_diagram",
        "club_head_trajectory_3d": "_plot_club_head_trajectory_3d",
        "swing_profile_radar": "_plot_swing_profile_radar",
    }

    #: Dashboard combo-box labels, in display order.  Single source of
    #: truth shared with ``src.shared.python.dashboard.window``.
    DASHBOARD_PLOT_LABELS: tuple[str, ...] = (
        "Joint Angles",
        "Joint Velocities",
        "Joint Torques",
        "Energies",
        "Club Head Speed",
        "Angular Momentum",
        "Power Flow",
        "Joint Power Curves",
        "Impulse Accumulation",
        "Phase Diagram (Joint 0)",
        "Poincaré Map (3D)",
        "Chaos Analysis (Lyapunov)",
        "Recurrence Plot",
        "Stability Diagram (CoM vs CoP)",
        "CoP Trajectory",
        "GRF Butterfly Diagram",
        "Club Head Trajectory (3D)",
        "Kinematic Sequence (Bars)",
        "Swing Profile (Radar)",
        "Summary Dashboard",
    )

    #: GUI label -> headless plot-type id (labels without a headless
    #: extractor yet are intentionally absent; see issue #7446).
    DASHBOARD_LABEL_TO_PLOT_TYPE: dict[str, str] = {
        "Joint Angles": "joint_angles",
        "Joint Velocities": "joint_velocities",
        "Joint Torques": "joint_torques",
        "Energies": "energies",
        "Club Head Speed": "club_head_speed",
        "Angular Momentum": "angular_momentum",
        "Power Flow": "power_flow",
        "Joint Power Curves": "joint_power_curves",
        "Impulse Accumulation": "impulse_accumulation",
        "Phase Diagram (Joint 0)": "phase_diagram",
        "Stability Diagram (CoM vs CoP)": "stability_diagram",
        "CoP Trajectory": "cop_trajectory",
        "Club Head Trajectory (3D)": "club_head_trajectory_3d",
        "Swing Profile (Radar)": "swing_profile_radar",
    }

    def __init__(
        self,
        recorder: RecorderLike,
        joint_names: list[str] | None = None,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            recorder: Object providing ``get_time_series(field_name)``
                (e.g. ``GenericPhysicsRecorder`` or any stub).
            joint_names: Optional human-readable joint names.

        Raises:
            ValueError: If recorder is None.
            TypeError: If recorder lacks ``get_time_series``.
        """
        if recorder is None:
            raise ValueError("recorder must be provided")
        if not hasattr(recorder, "get_time_series"):
            raise TypeError(
                "recorder must provide a get_time_series(field_name) method"
            )
        self.recorder = recorder
        self.joint_names = list(joint_names or [])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def available_plot_types(cls) -> list[str]:
        """Return the sorted list of registered plot-type ids."""
        return sorted(cls.PLOT_TYPES)

    def get_plot_data(self, plot_type: str) -> PlotData:
        """Compute structured plot data for a registered plot type.

        Args:
            plot_type: One of :meth:`available_plot_types`.

        Returns:
            A JSON-serializable :class:`PlotData`.  When nothing has been
            recorded the result has empty series and a ``"message"``
            metadata entry (mirrors the GUI's "No data recorded" text).

        Raises:
            TypeError: If ``plot_type`` is not a string.
            ValueError: If ``plot_type`` is not registered.
        """
        if not isinstance(plot_type, str):
            raise TypeError(
                f"plot_type must be a string, got {type(plot_type).__name__}"
            )
        method_name = self.PLOT_TYPES.get(plot_type)
        if method_name is None:
            raise ValueError(
                f"Unknown plot type '{plot_type}'. "
                f"Valid types: {self.available_plot_types()}"
            )
        result: PlotData = getattr(self, method_name)()
        return result

    def compute_counterfactual(
        self, kind: str, *, run_post_hoc: bool = True
    ) -> CounterfactualResult:
        """Return counterfactual / induced-acceleration data headlessly.

        Args:
            kind: ``"ztcf"`` / ``"zvcf"`` (counterfactual accelerations) or
                ``"gravity"`` / ``"drift"`` / ``"control"`` / ``"total"``
                (induced accelerations).
            run_post_hoc: When True and no data is stored yet, trigger the
                recorder's ``compute_analysis_post_hoc()`` (engine-backed,
                Qt-free) before fetching.

        Raises:
            TypeError: If ``kind`` is not a string.
            ValueError: If ``kind`` is not a recognized kind.
        """
        if not isinstance(kind, str):
            raise TypeError(f"kind must be a string, got {type(kind).__name__}")
        valid = COUNTERFACTUAL_KINDS | INDUCED_ACCELERATION_KINDS
        if kind not in valid:
            raise ValueError(
                f"Unknown counterfactual kind '{kind}'. Valid kinds: {sorted(valid)}"
            )

        times, values = self._fetch_counterfactual(kind)
        if times.size == 0 and run_post_hoc:
            post_hoc = getattr(self.recorder, "compute_analysis_post_hoc", None)
            if callable(post_hoc):
                logger.info("No stored '%s' data; running post-hoc analysis.", kind)
                post_hoc()
                times, values = self._fetch_counterfactual(kind)

        values_2d = np.atleast_2d(np.asarray(values, dtype=float))
        if times.size == 0:
            values_list: list[list[float]] = []
        else:
            values_list = values_2d.reshape(times.size, -1).tolist()
        return CounterfactualResult(
            kind=kind,
            times=np.asarray(times, dtype=float).tolist(),
            values=values_list,
            units="rad/s^2",
            metadata={"n_frames": int(times.size)},
        )

    # ------------------------------------------------------------------
    # Headless computations previously embedded in the PyQt6 window
    # ------------------------------------------------------------------

    def compute_recurrence_matrix(self) -> np.ndarray:
        """Compute the recurrence matrix from recorded joint data.

        Raises:
            ValueError: If no data has been recorded.
        """
        analyzer = self._build_statistical_analyzer(require_torques=False)
        return np.asarray(analyzer.compute_recurrence_matrix())

    def compute_swing_profile_metrics(self) -> dict[str, float]:
        """Compute the Swing Profile radar metrics (0-100 scores).

        Raises:
            ValueError: If no data was recorded or the profile cannot be
                computed.
        """
        analyzer = self._build_statistical_analyzer(require_torques=True)
        dna = analyzer.compute_swing_profile()
        if not dna:
            raise ValueError("Could not compute Swing Profile")
        return {
            "Speed": float(dna.speed_score),
            "Sequence": float(dna.sequence_score),
            "Stability": float(dna.stability_score),
            "Efficiency": float(dna.efficiency_score),
            "Power": float(dna.power_score),
        }

    def derive_kinematic_sequence_indices(self) -> dict[str, int]:
        """Derive proximal-to-distal segment indices from recorded DoFs.

        Returns:
            Mapping of segment name to joint index (empty when no data).
        """
        _, vels = self._series("joint_velocities")
        n_joints = vels.shape[1] if vels.ndim == 2 and len(vels) > 0 else 0
        if n_joints >= 3:
            indices: dict[str, int] = {
                "proximal": 0,
                "mid_proximal": 1,
                "mid_distal": min(2, n_joints - 1),
            }
            if n_joints > 3:
                indices["distal"] = n_joints - 1
            return indices
        return {f"Joint {i}": i for i in range(n_joints)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_counterfactual(self, kind: str) -> tuple[np.ndarray, np.ndarray]:
        if kind in COUNTERFACTUAL_KINDS:
            getter = getattr(self.recorder, "get_counterfactual_series", None)
        else:
            getter = getattr(self.recorder, "get_induced_acceleration_series", None)
        if not callable(getter):
            return np.array([]), np.array([])
        times, values = getter(kind)
        return np.asarray(times), np.asarray(values)

    def _build_statistical_analyzer(self, *, require_torques: bool) -> Any:
        """Build a ``StatisticalAnalyzer`` from recorded data (lazy import)."""
        from src.shared.python.validation_pkg.statistical_analysis import (
            StatisticalAnalyzer,
        )

        times, positions = self._series("joint_positions")
        _, velocities = self._series("joint_velocities")
        if len(times) == 0:
            raise ValueError("No data available")

        if require_torques:
            _, torques = self._series("joint_torques")
            _, club_speed = self._series("club_head_speed")
            return StatisticalAnalyzer(
                times=times,
                joint_positions=positions,
                joint_velocities=velocities,
                joint_torques=torques,
                club_head_speed=club_speed if club_speed.size > 0 else None,
            )
        return StatisticalAnalyzer(
            times=times,
            joint_positions=positions,
            joint_velocities=velocities,
            joint_torques=np.zeros_like(positions),
        )

    def _series(self, field_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Fetch a recorder time series as numpy arrays."""
        try:
            times, values = self.recorder.get_time_series(field_name)
        except (KeyError, AttributeError):
            return np.array([]), np.array([])
        return np.asarray(times), np.asarray(values)

    def _joint_label(self, idx: int, data_dim: int) -> str:
        """Label aligned with the data dimension (mirrors DataManager)."""
        return aligned_joint_label(self.joint_names, idx, data_dim)

    def _joint_name(self, idx: int) -> str:
        """Plain joint name (mirrors DataManager.get_joint_name)."""
        return joint_name(self.joint_names, idx)

    @staticmethod
    def _empty(
        plot_type: str,
        title: str,
        x_label: str,
        y_label: str,
        message: str = "No data recorded",
    ) -> PlotData:
        return PlotData(
            plot_type=plot_type,
            title=title,
            x_label=x_label,
            y_label=y_label,
            series=[],
            metadata={"message": message},
        )

    def _per_joint_plot(
        self,
        plot_type: str,
        field_name: str,
        title: str,
        y_label: str,
        units: str,
        transform: Any = None,
    ) -> PlotData:
        """Build a per-joint multi-series time plot."""
        times, values = self._series(field_name)
        if len(times) == 0 or values.size == 0 or values.ndim != 2:
            return self._empty(plot_type, title, "Time (s)", y_label)
        series = []
        for idx in range(values.shape[1]):
            y = values[:, idx]
            if transform is not None:
                y = transform(y)
            series.append(
                PlotSeries(
                    name=self._joint_label(idx, values.shape[1]),
                    x=times.tolist(),
                    y=np.asarray(y, dtype=float).tolist(),
                    units=units,
                )
            )
        return PlotData(
            plot_type=plot_type,
            title=title,
            x_label="Time (s)",
            y_label=y_label,
            series=series,
            metadata={"n_frames": int(len(times)), "n_joints": values.shape[1]},
        )

    # ------------------------------------------------------------------
    # Plot-type extractors (data parity with the matplotlib renderers)
    # ------------------------------------------------------------------

    def _plot_joint_angles(self) -> PlotData:
        return self._per_joint_plot(
            "joint_angles",
            "joint_positions",
            "Joint Angles vs Time",
            "Joint Angle (degrees)",
            "deg",
            transform=np.rad2deg,
        )

    def _plot_joint_velocities(self) -> PlotData:
        return self._per_joint_plot(
            "joint_velocities",
            "joint_velocities",
            "Joint Velocities vs Time",
            "Angular Velocity (deg/s)",
            "deg/s",
            transform=np.rad2deg,
        )

    def _plot_joint_torques(self) -> PlotData:
        return self._per_joint_plot(
            "joint_torques",
            "joint_torques",
            "Applied Joint Torques vs Time",
            "Torque (Nm)",
            "Nm",
        )

    def _plot_power_flow(self) -> PlotData:
        data = self._per_joint_plot(
            "power_flow",
            "actuator_powers",
            "Power Flow (Generation/Absorption)",
            "Power (W)",
            "W",
        )
        # The GUI labels power-flow series with plain joint names.
        for idx, s in enumerate(data.series):
            s.name = self._joint_name(idx)
        return data

    def _plot_energies(self) -> PlotData:
        times_ke, ke = self._series("kinetic_energy")
        times_pe, pe = self._series("potential_energy")
        times_te, te = self._series("total_energy")
        if len(times_ke) == 0:
            return self._empty("energies", "Energy Analysis", "Time (s)", "Energy (J)")
        series = [
            PlotSeries("Kinetic Energy", times_ke.tolist(), ke.tolist(), units="J"),
            PlotSeries("Potential Energy", times_pe.tolist(), pe.tolist(), units="J"),
            PlotSeries("Total Energy", times_te.tolist(), te.tolist(), units="J"),
        ]
        return PlotData(
            plot_type="energies",
            title="Energy Analysis",
            x_label="Time (s)",
            y_label="Energy (J)",
            series=series,
            metadata={"n_frames": int(len(times_ke))},
        )

    def _plot_club_head_speed(self) -> PlotData:
        times, speeds = self._series("club_head_speed")
        if len(times) == 0 or speeds.size == 0:
            return self._empty(
                "club_head_speed",
                "Club Head Speed vs Time",
                "Time (s)",
                "Club Head Speed (mph)",
                message="No club head data",
            )
        speeds_mph = np.asarray(speeds, dtype=float) * MPS_TO_MPH
        max_idx = int(np.argmax(speeds_mph))
        return PlotData(
            plot_type="club_head_speed",
            title="Club Head Speed vs Time",
            x_label="Time (s)",
            y_label="Club Head Speed (mph)",
            series=[
                PlotSeries(
                    "Club Head Speed",
                    times.tolist(),
                    speeds_mph.tolist(),
                    units="mph",
                )
            ],
            metadata={
                "peak_speed_mph": float(speeds_mph[max_idx]),
                "peak_time_s": float(times[max_idx]),
            },
        )

    def _plot_angular_momentum(self) -> PlotData:
        times, am = self._series("angular_momentum")
        if len(times) == 0 or am.size == 0 or am.ndim != 2 or am.shape[1] < 3:
            return self._empty(
                "angular_momentum",
                "System Angular Momentum",
                "Time (s)",
                "Angular Momentum (kg m²/s)",
                message="No Angular Momentum Data",
            )
        am_f = am.astype(float, copy=False)
        magnitude = np.sqrt(np.einsum("...i,...i->...", am_f, am_f))
        units = "kg m^2/s"
        x = times.tolist()
        series = [
            PlotSeries("Lx", x, am_f[:, 0].tolist(), units=units),
            PlotSeries("Ly", x, am_f[:, 1].tolist(), units=units),
            PlotSeries("Lz", x, am_f[:, 2].tolist(), units=units),
            PlotSeries("Magnitude", x, magnitude.tolist(), units=units),
        ]
        return PlotData(
            plot_type="angular_momentum",
            title="System Angular Momentum",
            x_label="Time (s)",
            y_label="Angular Momentum (kg m²/s)",
            series=series,
            metadata={"n_frames": int(len(times))},
        )

    def _plot_joint_power_curves(self) -> PlotData:
        times, torques = self._series("joint_torques")
        _, velocities = self._series("joint_velocities")
        if (
            len(times) == 0
            or torques.size == 0
            or velocities.size == 0
            or torques.ndim != 2
            or velocities.ndim != 2
        ):
            return self._empty(
                "joint_power_curves",
                "Joint Power: Generation (+) vs Absorption (-)",
                "Time (s)",
                "Power (W)",
                message="No data available",
            )
        n = min(torques.shape[1], velocities.shape[1])
        series = []
        for idx in range(n):
            power = torques[:, idx] * velocities[:, idx]
            series.append(
                PlotSeries(
                    name=self._joint_label(idx, torques.shape[1]),
                    x=times.tolist(),
                    y=np.asarray(power, dtype=float).tolist(),
                    units="W",
                )
            )
        return PlotData(
            plot_type="joint_power_curves",
            title="Joint Power: Generation (+) vs Absorption (-)",
            x_label="Time (s)",
            y_label="Power (W)",
            series=series,
            metadata={"n_frames": int(len(times)), "n_joints": n},
        )

    def _plot_impulse_accumulation(self) -> PlotData:
        times, torques = self._series("joint_torques")
        if len(times) < 2 or torques.size == 0 or torques.ndim != 2:
            return self._empty(
                "impulse_accumulation",
                "Angular Impulse Accumulation",
                "Time (s)",
                "Cumulative Impulse (Nms)",
                message="No data available",
            )
        dt = float(np.mean(np.diff(times)))
        if dt <= 0:
            return self._empty(
                "impulse_accumulation",
                "Angular Impulse Accumulation",
                "Time (s)",
                "Cumulative Impulse (Nms)",
                message="Non-increasing time base",
            )
        series = []
        for idx in range(torques.shape[1]):
            tau = np.asarray(torques[:, idx], dtype=float)
            # Cumulative trapezoidal integration with uniform dx=dt,
            # initial=0 (matches scipy.integrate.cumulative_trapezoid).
            increments = (tau[1:] + tau[:-1]) * 0.5 * dt
            impulse = np.concatenate(([0.0], np.cumsum(increments)))
            series.append(
                PlotSeries(
                    name=self._joint_label(idx, torques.shape[1]),
                    x=times.tolist(),
                    y=impulse.tolist(),
                    units="Nms",
                )
            )
        return PlotData(
            plot_type="impulse_accumulation",
            title="Angular Impulse Accumulation",
            x_label="Time (s)",
            y_label="Cumulative Impulse (Nms)",
            series=series,
            metadata={"n_frames": int(len(times)), "dt": dt},
        )

    def _plot_phase_diagram(self, joint_idx: int = 0) -> PlotData:
        times, positions = self._series("joint_positions")
        _, velocities = self._series("joint_velocities")
        name = self._joint_name(joint_idx)
        title = f"Phase Diagram: {name}"
        if (
            len(times) == 0
            or positions.ndim != 2
            or velocities.ndim != 2
            or joint_idx >= positions.shape[1]
            or joint_idx >= velocities.shape[1]
        ):
            return self._empty(
                "phase_diagram",
                title,
                f"{name} Angle (deg)",
                "Angular Velocity (deg/s)",
                message="No data available or index out of bounds",
            )
        angles = np.rad2deg(positions[:, joint_idx])
        ang_vels = np.rad2deg(velocities[:, joint_idx])
        return PlotData(
            plot_type="phase_diagram",
            title=title,
            x_label=f"{name} Angle (deg)",
            y_label="Angular Velocity (deg/s)",
            series=[
                PlotSeries(
                    name,
                    angles.tolist(),
                    ang_vels.tolist(),
                    units="deg/s",
                    metadata={"color_by": "time", "times": times.tolist()},
                )
            ],
            metadata={"joint_idx": joint_idx},
        )

    def _plot_cop_trajectory(self) -> PlotData:
        times, cop = self._series("cop_position")
        if len(times) == 0 or cop.size == 0 or cop.ndim != 2 or cop.shape[1] < 2:
            return self._empty(
                "cop_trajectory",
                "Center of Pressure Trajectory",
                "X Position (m)",
                "Y Position (m)",
                message="No CoP Data",
            )
        return PlotData(
            plot_type="cop_trajectory",
            title="Center of Pressure Trajectory",
            x_label="X Position (m)",
            y_label="Y Position (m)",
            series=[
                PlotSeries(
                    "CoP",
                    cop[:, 0].tolist(),
                    cop[:, 1].tolist(),
                    units="m",
                    metadata={"color_by": "time", "times": times.tolist()},
                )
            ],
            metadata={"n_frames": int(len(times))},
        )

    def _plot_stability_diagram(self) -> PlotData:
        times, cop = self._series("cop_position")
        _, com = self._series("com_position")
        if (
            len(times) == 0
            or cop.size == 0
            or com.size == 0
            or cop.ndim != 2
            or com.ndim != 2
        ):
            return self._empty(
                "stability_diagram",
                "Stability Diagram (CoM vs CoP)",
                "X Position (m)",
                "Y Position (m)",
                message="No Stability Data",
            )
        return PlotData(
            plot_type="stability_diagram",
            title="Stability Diagram (CoM vs CoP)",
            x_label="X Position (m)",
            y_label="Y Position (m)",
            series=[
                PlotSeries("CoP", cop[:, 0].tolist(), cop[:, 1].tolist(), units="m"),
                PlotSeries(
                    "CoM (Proj)", com[:, 0].tolist(), com[:, 1].tolist(), units="m"
                ),
            ],
            metadata={"n_frames": int(len(times))},
        )

    def _plot_club_head_trajectory_3d(self) -> PlotData:
        times, pos = self._series("club_head_position")
        if len(times) == 0 or pos.size == 0 or pos.ndim != 2 or pos.shape[1] < 3:
            return self._empty(
                "club_head_trajectory_3d",
                "Club Head 3D Trajectory",
                "X (m)",
                "Y (m)",
                message="No club head data",
            )
        return PlotData(
            plot_type="club_head_trajectory_3d",
            title="Club Head 3D Trajectory",
            x_label="X (m)",
            y_label="Y (m)",
            series=[
                PlotSeries(
                    "Club Head",
                    pos[:, 0].tolist(),
                    pos[:, 1].tolist(),
                    z=pos[:, 2].tolist(),
                    units="m",
                    metadata={"color_by": "time", "times": times.tolist()},
                )
            ],
            metadata={"n_frames": int(len(times)), "z_label": "Z (m)"},
        )

    def _plot_swing_profile_radar(self) -> PlotData:
        try:
            metrics = self.compute_swing_profile_metrics()
        except ValueError as e:
            return self._empty(
                "swing_profile_radar",
                "Swing Profile",
                "",
                "Score",
                message=str(e),
            )
        categories = list(metrics)
        return PlotData(
            plot_type="swing_profile_radar",
            title="Swing Profile",
            x_label="",
            y_label="Score",
            series=[
                PlotSeries(
                    "Swing Profile",
                    x=list(range(len(categories))),
                    y=[metrics[c] for c in categories],
                    units="score",
                    metadata={"categories": categories},
                )
            ],
            metadata={"chart": "radar"},
        )


def get_plot_data(
    plot_type: str,
    recorder: RecorderLike,
    joint_names: list[str] | None = None,
) -> PlotData:
    """Module-level convenience wrapper (issue #7446 spec signature).

    Equivalent to ``AnalysisOrchestrator(recorder, joint_names)
    .get_plot_data(plot_type)``.
    """
    return AnalysisOrchestrator(recorder, joint_names).get_plot_data(plot_type)
