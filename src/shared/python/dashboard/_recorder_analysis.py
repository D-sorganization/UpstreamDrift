from __future__ import annotations

from typing import Any

import numpy as np

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


class _AnalysisMixin:
    data: dict[str, Any]
    current_idx: int
    _buffers_initialized: bool
    engine: Any

    def compute_analysis_post_hoc(self) -> None:
        logger.info("Computing post-hoc analysis...")

        if not self._buffers_initialized or self.current_idx == 0:
            logger.warning("No data recorded for post-hoc analysis")
            return

        n_frames = self.current_idx
        times = self.data["times"][:n_frames]
        qs = self.data["joint_positions"][:n_frames]
        vs = self.data["joint_velocities"][:n_frames]
        taus = self.data["joint_torques"][:n_frames]

        ztcf_accels = []
        zvcf_accels = []
        drift_accels = []
        control_accels = []

        for i in range(n_frames):
            q = qs[i]
            v = vs[i]
            tau = taus[i]

            self.engine.set_state(q, v)
            self.engine.set_control(tau)

            self.engine.forward()

            ztcf = self.engine.compute_ztcf(q, v)
            ztcf_accels.append(ztcf)

            zvcf = self.engine.compute_zvcf(q)
            zvcf_accels.append(zvcf)

            drift = self.engine.compute_drift_acceleration()
            drift_accels.append(drift)

            ctrl_acc = self.engine.compute_control_acceleration(tau)
            control_accels.append(ctrl_acc)

        times_arr = np.array(times)
        self.data["counterfactuals"]["ztcf_accel"] = (times_arr, np.array(ztcf_accels))
        self.data["counterfactuals"]["zvcf_accel"] = (times_arr, np.array(zvcf_accels))
        self.data["counterfactuals"]["ztcf"] = (times_arr, np.array(ztcf_accels))
        self.data["counterfactuals"]["zvcf"] = (times_arr, np.array(zvcf_accels))

        self.data["induced_accelerations"]["gravity"] = (
            times_arr,
            np.array(drift_accels),
        )
        self.data["induced_accelerations"]["drift"] = (
            times_arr,
            np.array(drift_accels),
        )
        self.data["induced_accelerations"]["control"] = (
            times_arr,
            np.array(control_accels),
        )
        self.data["induced_accelerations"]["total"] = (
            times_arr,
            np.array(drift_accels) + np.array(control_accels),
        )

        logger.info("Post-hoc analysis complete.")

    def compute_grf_and_wrench_analysis(
        self, impact_time: float | None = None, fsp_window_ms: float = 100.0
    ) -> dict[str, Any]:
        if fsp_window_ms is None:
            raise ValueError("fsp_window_ms must be provided")
        if not self._buffers_initialized or self.current_idx == 0:
            logger.warning("No data recorded for GRF/wrench analysis")
            return {}

        n = self.current_idx
        times = self.data["times"][:n]
        forces = self.data["ground_forces"][:n]
        moments = self.data["ground_moments"][:n]
        cops = self.data["cop_position"][:n]

        grf_summary = self._run_grf_analysis(times, forces, moments, cops)

        impact_time = self._resolve_impact_time(impact_time, times, forces, n)

        fsp = self._fit_swing_plane(times, impact_time, fsp_window_ms, n)

        wrench_arrays = self._compute_wrench_decomposition(forces, moments, fsp, n)

        result = self._build_grf_wrench_result(grf_summary, fsp, wrench_arrays)

        self.data["grf_analysis"] = result["grf_analysis"]
        self.data["fsp"] = result["fsp"]
        self.data["wrench_swing_plane"] = result["wrench_swing_plane"]

        logger.info(
            "GRF and wrench analysis complete. FSP RMSE=%.4f m",
            fsp.fitting_rmse if fsp else float("nan"),
        )
        return result

    def _run_grf_analysis(
        self,
        times: np.ndarray,
        forces: np.ndarray,
        moments: np.ndarray,
        cops: np.ndarray,
    ) -> Any:
        if times is None:
            raise ValueError("times must be provided")
        from src.shared.python.physics.ground_reaction_forces import (
            FootSide,
            GRFAnalyzer,
            GRFTimeSeries,
        )

        grf_ts = GRFTimeSeries(
            timestamps=times,
            forces=forces,
            moments=moments,
            cops=cops,
            foot_side=FootSide.COMBINED,
        )

        analyzer = GRFAnalyzer()
        analyzer.add_grf_data(grf_ts)

        try:
            return analyzer.analyze(FootSide.COMBINED)
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning("GRF analysis failed: %s", e)
            return None

    @staticmethod
    def _resolve_impact_time(
        impact_time: float | None,
        times: np.ndarray,
        forces: np.ndarray,
        n: int,
    ) -> float:
        if times is None:
            raise ValueError("times must be provided")
        if impact_time is not None:
            return impact_time
        vertical_forces = forces[:, 2]
        if np.max(np.abs(vertical_forces)) > 0:
            return float(times[np.argmax(np.abs(vertical_forces))])
        return float(times[n // 2])

    def _fit_swing_plane(
        self,
        times: np.ndarray,
        impact_time: float,
        fsp_window_ms: float,
        n: int,
    ) -> Any:
        if times is None:
            raise ValueError("times must be provided")
        from src.shared.python.spatial_algebra.reference_frames import (
            fit_functional_swing_plane,
        )

        clubhead_traj = self.data["club_head_position"][:n]
        if clubhead_traj is not None and np.any(clubhead_traj != 0):
            try:
                return fit_functional_swing_plane(
                    clubhead_traj, times, impact_time, window_ms=fsp_window_ms
                )
            except (RuntimeError, ValueError, OSError) as e:
                logger.warning("FSP fitting failed: %s", e)
        return None

    @staticmethod
    def _compute_wrench_decomposition(
        forces: np.ndarray,
        moments: np.ndarray,
        fsp: Any,
        n: int,
    ) -> dict[str, np.ndarray]:
        if forces is None:
            raise ValueError("forces must be provided")
        from src.shared.python.spatial_algebra.reference_frames import (
            ReferenceFrame,
            ReferenceFrameTransformer,
            WrenchInFrame,
        )

        transformer = ReferenceFrameTransformer()
        if fsp is not None:
            transformer.set_swing_plane(fsp)

        wrench_decompositions: list[dict[str, float]] = []
        for i in range(n):
            wrench = WrenchInFrame(
                force=forces[i],
                torque=moments[i],
                frame=ReferenceFrame.GLOBAL,
                body_name="ground",
            )
            if fsp is not None:
                decomp = transformer.get_swing_plane_decomposition(wrench)
            else:
                decomp = {
                    "force_in_plane": 0.0,
                    "force_out_of_plane": 0.0,
                    "force_along_grip": 0.0,
                    "torque_in_plane": 0.0,
                    "torque_out_of_plane": 0.0,
                    "torque_about_grip": 0.0,
                }
            wrench_decompositions.append(decomp)

        decomp_keys = [
            "force_in_plane",
            "force_out_of_plane",
            "force_along_grip",
            "torque_in_plane",
            "torque_out_of_plane",
            "torque_about_grip",
        ]
        return {k: np.array([d[k] for d in wrench_decompositions]) for k in decomp_keys}

    @staticmethod
    def _build_grf_wrench_result(
        grf_summary: Any,
        fsp: Any,
        wrench_arrays: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        if wrench_arrays is None:
            raise ValueError("wrench_arrays must be provided")
        result: dict[str, Any] = {
            "grf_analysis": {},
            "fsp": {},
            "wrench_swing_plane": wrench_arrays,
        }

        if grf_summary is not None:
            result["grf_analysis"] = {
                "peak_vertical_force": grf_summary.peak_vertical_force,
                "peak_horizontal_force": grf_summary.peak_horizontal_force,
                "time_to_peak_vertical": grf_summary.time_to_peak_vertical,
                "cop_trajectory_length": grf_summary.cop_trajectory_length,
                "cop_range_ap": grf_summary.cop_range_ap,
                "cop_range_ml": grf_summary.cop_range_ml,
                "linear_impulse_magnitude": grf_summary.linear_impulse.linear_impulse_magnitude,
                "angular_impulse_magnitude": grf_summary.linear_impulse.angular_impulse_magnitude,
                "duration": grf_summary.linear_impulse.duration,
            }

        if fsp is not None:
            result["fsp"] = {
                "origin": fsp.origin,
                "normal": fsp.normal,
                "in_plane_x": fsp.in_plane_x,
                "in_plane_y": fsp.in_plane_y,
                "grip_axis": fsp.grip_axis,
                "fitting_rmse": fsp.fitting_rmse,
                "fitting_window_ms": fsp.fitting_window_ms,
            }

        return result
