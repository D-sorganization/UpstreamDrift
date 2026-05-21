from __future__ import annotations

from typing import Any, cast

import numpy as np

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


class _RecordingMixin:
    data: dict[str, Any]
    current_idx: int
    is_recording: bool
    _buffers_initialized: bool
    analysis_config: dict[str, Any]
    engine: Any

    def record_step(self, control_input: np.ndarray | None = None) -> None:
        if not self.is_recording:
            return

        self._ensure_capacity()  # type: ignore[attr-defined]

        if not self.is_recording:
            return

        full_state = self.engine.get_full_state()
        q = full_state["q"]
        v = full_state["v"]
        t = full_state["t"]
        M = full_state.get("M")

        if not self._buffers_initialized:
            self._initialize_array_buffers(q, v)  # type: ignore[attr-defined]

        tau = control_input if control_input is not None else np.zeros(len(v))
        ke = self._compute_kinetic_energy(v, M)

        idx = self.current_idx
        self._record_realtime_analysis(idx, q, v, tau, M)
        self._store_basic_data(idx, t, q, v, ke, tau)
        self._record_ground_forces(idx)

        self.current_idx += 1

    def _compute_kinetic_energy(self, v: np.ndarray, M: np.ndarray | None) -> float:
        if v is None:
            raise ValueError("v must be provided")
        if M is not None and M.size > 0:
            try:
                return 0.5 * v.T @ M @ v
            except (RuntimeError, ValueError, OSError) as e:
                logger.warning("Failed to compute kinetic energy: %s", e)
        return 0.0

    def _record_realtime_analysis(  # noqa: C901
        self,
        idx: int,
        q: np.ndarray,
        v: np.ndarray,
        tau: np.ndarray,
        M: np.ndarray | None,
    ) -> None:
        if idx is None:
            raise ValueError("idx must be provided")
        if self.analysis_config["ztcf"] and self.data["ztcf_accel"] is not None:
            try:
                self.data["ztcf_accel"][idx] = self.engine.compute_ztcf(q, v)
            except (ValueError, RuntimeError, AttributeError) as e:
                logger.warning("Failed to compute ZTCF at frame %d: %s", idx, e)

        if self.analysis_config["zvcf"] and self.data["zvcf_accel"] is not None:
            try:
                self.data["zvcf_accel"][idx] = self.engine.compute_zvcf(q)
            except (ValueError, RuntimeError, AttributeError) as e:
                logger.warning("Failed to compute ZVCF at frame %d: %s", idx, e)

        if self.analysis_config["track_drift"] and self.data["drift_accel"] is not None:
            try:
                self.data["drift_accel"][idx] = self.engine.compute_drift_acceleration()
            except (ValueError, RuntimeError, AttributeError) as e:
                logger.warning(
                    "Failed to compute drift acceleration at frame %d: %s", idx, e
                )

        if (
            self.analysis_config["track_total_control"]
            and self.data["control_accel"] is not None
        ):
            try:
                self.data["control_accel"][idx] = (
                    self.engine.compute_control_acceleration(tau)
                )
            except (ValueError, RuntimeError, AttributeError) as e:
                logger.warning(
                    "Failed to compute control acceleration at frame %d: %s", idx, e
                )

        self._record_induced_accelerations(idx, tau, M)

    def _record_induced_accelerations(  # noqa: C901
        self, idx: int, tau: np.ndarray, M: np.ndarray | None
    ) -> None:
        if idx is None:
            raise ValueError("idx must be provided")
        sources = cast(list[int], self.analysis_config["induced_accel_sources"])
        if sources and M is not None and M.size > 0:
            try:
                M_inv = np.linalg.inv(M)
                for src_idx in sources:
                    if src_idx in self.data["induced_accelerations"]:
                        self.data["induced_accelerations"][src_idx][idx] = (
                            M_inv[:, src_idx] * tau[src_idx]
                        )
            except (ValueError, TypeError, RuntimeError) as e:
                logger.warning(
                    "Failed to compute induced accelerations at frame %d: %s", idx, e
                )
        elif sources:
            for src_idx in sources:
                if src_idx in self.data["induced_accelerations"]:
                    try:
                        tau_single = np.zeros_like(tau)
                        tau_single[src_idx] = tau[src_idx]
                        self.data["induced_accelerations"][src_idx][idx] = (
                            self.engine.compute_control_acceleration(tau_single)
                        )
                    except (ValueError, TypeError, RuntimeError) as e:
                        logger.warning(
                            "Failed to compute induced acceleration for source %d: %s",
                            src_idx,
                            e,
                        )

    def _store_basic_data(
        self,
        idx: int,
        t: float,
        q: np.ndarray,
        v: np.ndarray,
        ke: float,
        tau: np.ndarray,
    ) -> None:
        if idx is None:
            raise ValueError("idx must be provided")
        self.data["times"][idx] = t
        self.data["joint_positions"][idx] = q
        self.data["joint_velocities"][idx] = v
        self.data["kinetic_energy"][idx] = ke
        self.data["joint_torques"][idx] = tau

    def _record_ground_forces(self, idx: int) -> None:
        try:
            grf = self.engine.compute_contact_forces()
            if grf is not None and len(grf) >= 3:
                self.data["ground_forces"][idx] = grf[:3]
                if len(grf) >= 6:
                    self.data["ground_moments"][idx] = grf[3:6]
        except (ValueError, RuntimeError, AttributeError) as e:
            logger.warning("Failed to compute ground forces at frame %d: %s", idx, e)

    def update_control(self, u: np.ndarray) -> None:
        if self.is_recording and self.current_idx > 0 and self._buffers_initialized:
            self.data["joint_torques"][self.current_idx - 1] = u
