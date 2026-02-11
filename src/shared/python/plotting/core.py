"""Main plotting orchestration module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.shared.python.plotting.config import COLORS
from src.shared.python.plotting.renderers.club import ClubRenderer
from src.shared.python.plotting.renderers.comparison import ComparisonRenderer
from src.shared.python.plotting.renderers.coordination import CoordinationRenderer
from src.shared.python.plotting.renderers.dashboard import DashboardRenderer
from src.shared.python.plotting.renderers.energy import EnergyRenderer
from src.shared.python.plotting.renderers.kinematics import KinematicsRenderer
from src.shared.python.plotting.renderers.kinetics import KineticsRenderer
from src.shared.python.plotting.renderers.signal import SignalRenderer
from src.shared.python.plotting.renderers.stability import StabilityRenderer
from src.shared.python.plotting.transforms import DataManager, RecorderInterface

if TYPE_CHECKING:
    from src.shared.python.validation_pkg.statistical_analysis import PCAResult


class GolfSwingPlotter:
    """Creates advanced plots for golf swing analysis.

    This class generates various plots from recorded swing data,
    including kinematics, kinetics, energetics, and phase diagrams.
    It acts as a facade, delegating rendering to specialized modules.
    """

    def __init__(
        self,
        recorder: RecorderInterface,
        joint_names: list[str] | None = None,
        enable_cache: bool = True,
    ) -> None:
        """Initialize plotter with recorded data.

        Args:
            recorder: Object providing get_time_series(field_name) method
            joint_names: Optional list of joint names.
            enable_cache: If True, cache data fetches to improve performance
        """
        self.recorder = recorder
        self.joint_names = joint_names or []
        self.enable_cache = enable_cache

        # Data Manager
        self.data_manager = DataManager(recorder, joint_names, enable_cache)

        # Colors (exposed for backward compatibility)
        self.colors = COLORS

        # Initialize Renderers
        self.kinematics = KinematicsRenderer(self.data_manager)
        self.kinetics = KineticsRenderer(self.data_manager)
        self.energy = EnergyRenderer(self.data_manager)
        self.club = ClubRenderer(self.data_manager)
        self.signal = SignalRenderer(self.data_manager)
        self.coordination = CoordinationRenderer(self.data_manager)
        self.stability = StabilityRenderer(self.data_manager)
        self.dashboard = DashboardRenderer(self.data_manager)
        self.comparison = ComparisonRenderer(self.data_manager)

    # -------------------------------------------------------------------------
    # Delegated Methods
    # -------------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Clear the data cache."""
        self.data_manager.clear_cache()

    def get_joint_name(self, joint_idx: int) -> str:
        """Get human-readable joint name."""
        return self.data_manager.get_joint_name(joint_idx)

    def _get_aligned_label(self, idx: int, data_dim: int) -> str:
        """Get label aligned with data dimension (handling nq != nv)."""
        return self.data_manager.get_aligned_label(idx, data_dim)

    # --- Kinematics ---

    def plot_joint_angles(
        self, fig: Figure, joint_indices: list[int] | None = None
    ) -> None:
        """Plot joint angles over time."""
        self.kinematics.plot_joint_angles(fig, joint_indices)

    def plot_joint_velocities(
        self, fig: Figure, joint_indices: list[int] | None = None
    ) -> None:
        """Plot joint velocities over time."""
        self.kinematics.plot_joint_velocities(fig, joint_indices)

    def plot_angle_angle_diagram(
        self,
        fig: Figure,
        joint_idx_1: int,
        joint_idx_2: int,
        title: str | None = None,
        ax: Axes | None = None,
    ) -> None:
        """Plot Angle-Angle diagram (Cyclogram) for two joints."""
        self.kinematics.plot_angle_angle_diagram(
            fig, joint_idx_1, joint_idx_2, title, ax
        )

    def plot_phase_diagram(self, fig: Figure, joint_idx: int = 0) -> None:
        """Plot phase diagram (angle vs angular velocity) for a joint."""
        self.kinematics.plot_phase_diagram(fig, joint_idx)

    def plot_3d_phase_space(self, fig: Figure, joint_idx: int = 0) -> None:
        """Plot 3D phase space (Position vs Velocity vs Acceleration)."""
        self.kinematics.plot_3d_phase_space(fig, joint_idx)

    def plot_poincare_map_3d(
        self,
        fig: Figure,
        dimensions: list[tuple[str, int]],
        section_condition: tuple[str, int, float] = ("velocity", 0, 0.0),
        direction: str = "both",
        title: str | None = None,
    ) -> None:
        """Plot 3D Poincaré Map (Poincaré Section)."""
        self.kinematics.plot_poincare_map_3d(
            fig, dimensions, section_condition, direction, title
        )

    def plot_phase_space_reconstruction(
        self,
        fig: Figure,
        joint_idx: int = 0,
        delay: int = 10,
        embedding_dim: int = 3,
        signal_type: str = "position",
    ) -> None:
        """Plot Phase Space Reconstruction using Time-Delay Embedding."""
        self.kinematics.plot_phase_space_reconstruction(
            fig, joint_idx, delay, embedding_dim, signal_type
        )

    def plot_phase_space_density(
        self, fig: Figure, joint_idx: int = 0, bins: int = 50
    ) -> None:
        """Plot 2D Phase Space Density (Histogram)."""
        self.kinematics.plot_phase_space_density(fig, joint_idx, bins)

    # --- Kinetics ---

    def plot_joint_torques(
        self, fig: Figure, joint_indices: list[int] | None = None
    ) -> None:
        """Plot applied joint torques over time."""
        self.kinetics.plot_joint_torques(fig, joint_indices)

    def plot_actuator_powers(self, fig: Figure) -> None:
        """Plot actuator mechanical powers over time."""
        self.kinetics.plot_actuator_powers(fig)

    def plot_torque_comparison(self, fig: Figure) -> None:
        """Plot comparison of all joint torques."""
        self.kinetics.plot_torque_comparison(fig)

    def plot_work_loop(
        self, fig: Figure, joint_idx: int = 0, title: str | None = None
    ) -> None:
        """Plot Work Loop (Torque vs Angle) for a joint."""
        self.kinetics.plot_work_loop(fig, joint_idx, title)

    def plot_power_flow(self, fig: Figure) -> None:
        """Plot power flow (stacked bar) over time."""
        self.kinetics.plot_power_flow(fig)

    def plot_joint_power_curves(
        self, fig: Figure, joint_indices: list[int] | None = None
    ) -> None:
        """Plot joint power curves with generation/absorption regions."""
        self.kinetics.plot_joint_power_curves(fig, joint_indices)

    def plot_impulse_accumulation(
        self, fig: Figure, joint_indices: list[int] | None = None
    ) -> None:
        """Plot cumulative impulse (integrated torque) over time."""
        self.kinetics.plot_impulse_accumulation(fig, joint_indices)

    def plot_joint_stiffness(
        self, fig: Figure, joint_idx: int = 0, ax: Axes | None = None
    ) -> None:
        """Plot joint stiffness (moment-angle relationship)."""
        self.kinetics.plot_joint_stiffness(fig, joint_idx, ax)

    def plot_dynamic_stiffness(
        self, fig: Figure, joint_idx: int = 0, window_size: int = 20
    ) -> None:
        """Plot dynamic (time-varying) stiffness with R² quality metric."""
        self.kinetics.plot_dynamic_stiffness(fig, joint_idx, window_size)

    def plot_activation_heatmap(self, fig: Figure, data_type: str = "torque") -> None:
        """Plot activation heatmap (Joints vs Time)."""
        self.kinetics.plot_activation_heatmap(fig, data_type)

    def plot_angular_momentum(self, fig: Figure) -> None:
        """Plot Angular Momentum over time (Magnitude and Components)."""
        self.kinetics.plot_angular_momentum(fig)

    def plot_angular_momentum_3d(self, fig: Figure) -> None:
        """Plot 3D trajectory of the Angular Momentum vector."""
        self.kinetics.plot_angular_momentum_3d(fig)

    def plot_induced_acceleration(
        self,
        fig: Figure,
        source_name: str | int,
        joint_idx: int | None = None,
        breakdown_mode: bool = False,
    ) -> None:
        """Plot induced accelerations."""
        self.kinetics.plot_induced_acceleration(
            fig, source_name, joint_idx, breakdown_mode
        )

    # --- Energy ---

    def plot_energy_analysis(self, fig: Figure) -> None:
        """Plot kinetic, potential, and total energy over time."""
        self.energy.plot_energy_analysis(fig)

    # --- Club ---

    def plot_club_head_speed(self, fig: Figure) -> None:
        """Plot club head speed over time."""
        self.club.plot_club_head_speed(fig)

    def plot_club_head_trajectory(self, fig: Figure) -> None:
        """Plot 3D club head trajectory."""
        self.club.plot_club_head_trajectory(fig)

    def plot_swing_plane(self, fig: Figure) -> None:
        """Plot fitted swing plane and trajectory deviation."""
        self.club.plot_swing_plane(fig)

    def plot_club_induced_acceleration(
        self, fig: Figure, breakdown_mode: bool = True
    ) -> None:
        """Plot club head task-space induced accelerations."""
        self.club.plot_club_induced_acceleration(fig, breakdown_mode)

    # --- Signal ---

    def plot_jerk_trajectory(
        self, fig: Figure, joint_indices: list[int] | None = None
    ) -> None:
        """Plot jerk (rate of change of acceleration) over time."""
        self.signal.plot_jerk_trajectory(fig, joint_indices)

    def plot_frequency_analysis(
        self, fig: Figure, joint_idx: int = 0, signal_type: str = "velocity"
    ) -> None:
        """Plot frequency content (PSD) of a joint signal."""
        self.signal.plot_frequency_analysis(fig, joint_idx, signal_type)

    def plot_spectrogram(
        self, fig: Figure, joint_idx: int = 0, signal_type: str = "velocity"
    ) -> None:
        """Plot spectrogram of a joint signal."""
        self.signal.plot_spectrogram(fig, joint_idx, signal_type)

    def plot_multiscale_entropy(
        self, fig: Figure, joint_indices: list[int] | None = None, max_scale: int = 20
    ) -> None:
        """Plot Multiscale Entropy (MSE) curves."""
        self.signal.plot_multiscale_entropy(fig, joint_indices, max_scale)

    def plot_lyapunov_exponent(
        self, fig: Figure, joint_idx: int = 0, tau: int = 5, dim: int = 3
    ) -> None:
        """Plot divergence of nearest neighbors over time to estimate Lyapunov Exponent."""
        self.signal.plot_lyapunov_exponent(fig, joint_idx, tau, dim)

    def plot_wavelet_scalogram(
        self,
        fig: Figure,
        joint_idx: int,
        signal_type: str = "velocity",
        freq_range: tuple[float, float] = (1.0, 50.0),
        title_prefix: str = "",
    ) -> None:
        """Plot Continuous Wavelet Transform (CWT) scalogram."""
        self.signal.plot_wavelet_scalogram(
            fig, joint_idx, signal_type, freq_range, title_prefix
        )

    def plot_cross_wavelet(
        self,
        fig: Figure,
        joint_idx_1: int,
        joint_idx_2: int,
        signal_type: str = "velocity",
        freq_range: tuple[float, float] = (1.0, 50.0),
    ) -> None:
        """Plot Cross Wavelet Transform (XWT) between two signals."""
        self.signal.plot_cross_wavelet(
            fig, joint_idx_1, joint_idx_2, signal_type, freq_range
        )

    # --- Coordination ---

    def plot_coupling_angle(
        self,
        fig: Figure,
        coupling_angles: np.ndarray,
        title: str | None = None,
        ax: Axes | None = None,
    ) -> None:
        """Plot Coupling Angle time series (Vector Coding)."""
        self.coordination.plot_coupling_angle(fig, coupling_angles, title, ax)

    def plot_coordination_patterns(
        self, fig: Figure, coupling_angles: np.ndarray, title: str | None = None
    ) -> None:
        """Plot coordination patterns as a color-coded strip over time."""
        self.coordination.plot_coordination_patterns(fig, coupling_angles, title)

    def plot_continuous_relative_phase(
        self, fig: Figure, crp_data: np.ndarray, title: str | None = None
    ) -> None:
        """Plot Continuous Relative Phase (CRP) time series."""
        self.coordination.plot_continuous_relative_phase(fig, crp_data, title)

    def plot_dtw_alignment(
        self,
        fig: Figure,
        times1: np.ndarray,
        data1: np.ndarray,
        times2: np.ndarray,
        data2: np.ndarray,
        path: list[tuple[int, int]],
        title: str = "Sequence Alignment",
    ) -> None:
        """Plot alignment between two sequences (DTW)."""
        self.coordination.plot_dtw_alignment(
            fig, times1, data1, times2, data2, path, title
        )

    def plot_cross_recurrence_plot(
        self,
        fig: Figure,
        recurrence_matrix: np.ndarray,
        title: str = "Cross Recurrence Plot",
    ) -> None:
        """Plot Cross Recurrence Plot."""
        self.coordination.plot_cross_recurrence_plot(fig, recurrence_matrix, title)

    def plot_recurrence_plot(
        self,
        fig: Figure,
        recurrence_matrix: np.ndarray,
        title: str = "Recurrence Plot",
    ) -> None:
        """Plot Recurrence Plot."""
        self.coordination.plot_recurrence_plot(fig, recurrence_matrix, title)

    def plot_correlation_sum(
        self,
        fig: Figure,
        radii: np.ndarray,
        counts: np.ndarray,
        slope_region: slice | None = None,
        slope_val: float | None = None,
    ) -> None:
        """Plot Correlation Sum C(r) vs r on log-log scale."""
        self.coordination.plot_correlation_sum(
            fig, radii, counts, slope_region, slope_val
        )

    def plot_lag_matrix(
        self, fig: Figure, data_type: str = "velocity", max_lag: float = 0.5
    ) -> None:
        """Plot time lag matrix between joints."""
        self.coordination.plot_lag_matrix(fig, data_type, max_lag)

    def plot_kinematic_sequence(
        self,
        fig: Figure,
        segment_indices: dict[str, int],
        analyzer_result: Any | None = None,
    ) -> None:
        """Plot kinematic sequence (normalized velocities)."""
        self.coordination.plot_kinematic_sequence(fig, segment_indices, analyzer_result)

    def plot_kinematic_sequence_bars(
        self,
        fig: Figure,
        segment_indices: dict[str, int],
        impact_time: float | None = None,
    ) -> None:
        """Plot kinematic sequence as a Gantt-style bar chart of peak times."""
        self.coordination.plot_kinematic_sequence_bars(
            fig, segment_indices, impact_time
        )

    def plot_x_factor_cycle(self, fig: Figure, shoulder_idx: int, hip_idx: int) -> None:
        """Plot X-Factor Cycle (Stretch-Shortening Cycle)."""
        self.coordination.plot_x_factor_cycle(fig, shoulder_idx, hip_idx)

    def plot_muscle_synergies(self, fig: Figure, synergy_result: Any) -> None:
        """Plot extracted muscle synergies (Weights and Activations)."""
        self.coordination.plot_muscle_synergies(fig, synergy_result)

    def plot_correlation_matrix(self, fig: Figure, data_type: str = "velocity") -> None:
        """Plot correlation matrix between joints."""
        self.coordination.plot_correlation_matrix(fig, data_type)

    def plot_dynamic_correlation(
        self,
        fig: Figure,
        joint_idx_1: int,
        joint_idx_2: int,
        window_size: int = 20,
    ) -> None:
        """Plot Rolling Correlation between two joint velocities."""
        self.coordination.plot_dynamic_correlation(
            fig, joint_idx_1, joint_idx_2, window_size
        )

    def plot_synergy_trajectory(
        self,
        fig: Figure,
        synergy_result: Any,
        dim1: int = 0,
        dim2: int = 1,
    ) -> None:
        """Plot trajectory in synergy space."""
        self.coordination.plot_synergy_trajectory(fig, synergy_result, dim1, dim2)

    def plot_principal_component_analysis(
        self,
        fig: Figure,
        pca_result: PCAResult,
        modes_to_plot: int = 3,
    ) -> None:
        """Plot PCA/Principal Movements analysis results."""
        self.coordination.plot_principal_component_analysis(
            fig, pca_result, modes_to_plot
        )

    # --- Stability ---

    def plot_stability_metrics(self, fig: Figure) -> None:
        """Plot stability metrics (CoM-CoP distance and Inclination Angle)."""
        self.stability.plot_stability_metrics(fig)

    def plot_cop_trajectory(self, fig: Figure) -> None:
        """Plot Center of Pressure trajectory (top-down view)."""
        self.stability.plot_cop_trajectory(fig)

    def plot_cop_vector_field(self, fig: Figure, skip_steps: int = 5) -> None:
        """Plot CoP velocity vector field."""
        self.stability.plot_cop_vector_field(fig, skip_steps)

    def plot_grf_butterfly_diagram(
        self, fig: Figure, skip_steps: int = 5, scale: float = 0.001
    ) -> None:
        """Plot Ground Reaction Force 'Butterfly Diagram'."""
        self.stability.plot_grf_butterfly_diagram(fig, skip_steps, scale)

    def plot_3d_vector_field(
        self,
        fig: Figure,
        vector_name: str,
        position_name: str,
        skip_steps: int = 5,
        scale: float = 0.1,
    ) -> None:
        """Plot 3D vector field along a trajectory."""
        self.stability.plot_3d_vector_field(
            fig, vector_name, position_name, skip_steps, scale
        )

    def plot_local_stability(
        self,
        fig: Figure,
        joint_idx: int = 0,
        embedding_dim: int = 3,
        tau: int = 5,
    ) -> None:
        """Plot local divergence rate (Local Stability) over time."""
        self.stability.plot_local_stability(fig, joint_idx, embedding_dim, tau)

    def plot_stability_diagram(self, fig: Figure) -> None:
        """Plot Stability Diagram (CoM vs CoP on Ground Plane)."""
        self.stability.plot_stability_diagram(fig)

    # --- Dashboard ---

    def plot_summary_dashboard(self, fig: Figure) -> None:
        """Create a comprehensive dashboard with multiple subplots."""
        self.dashboard.plot_summary_dashboard(fig)

    def plot_radar_chart(
        self,
        fig: Figure,
        metrics: dict[str, float],
        title: str = "Swing Profile",
        ax: Axes | None = None,
    ) -> None:
        """Plot a radar chart of swing metrics."""
        self.dashboard.plot_radar_chart(fig, metrics, title, ax)

    # --- Comparison ---

    def plot_counterfactual_comparison(
        self, fig: Figure, cf_name: str, metric_idx: int = 0
    ) -> None:
        """Plot counterfactual data against actual data."""
        self.comparison.plot_counterfactual_comparison(fig, cf_name, metric_idx)
