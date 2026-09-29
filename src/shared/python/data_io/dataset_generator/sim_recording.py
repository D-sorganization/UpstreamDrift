"""Simulation-step recording mixin for DatasetGenerator.

NM-02 (#10617): native acceleration, applied controls, and optional
dynamics channels are recorded with failure trackers so finalize can
drop unavailable buffers instead of emitting zero measurements.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.shared.python.logging_pkg.logging_config import get_logger

from .config import GeneratorConfig
from .labels import ModelDoFLayout
from .sim_buffers import allocate_sim_buffers

logger = get_logger(__name__)

_OPTIONAL_CATCH = (
    ValueError,
    RuntimeError,
    AttributeError,
    TypeError,
    NotImplementedError,
)


class _SimRecordingMixin:
    """Allocate buffers and record per-step kinematics/dynamics.

    Requires ``self.engine`` from the composing DatasetGenerator.
    """

    engine: Any

    @staticmethod
    def _allocate_sim_buffers(
        config: GeneratorConfig,
        n_steps: int,
        layout: ModelDoFLayout,
    ) -> tuple[dict[str, np.ndarray | None], dict[str, bool]]:
        """Pre-allocate recording arrays and per-channel failure trackers."""
        return allocate_sim_buffers(config, n_steps, layout)

    def _execute_sim_loop(
        self,
        config: GeneratorConfig,
        control_sequence: np.ndarray,
        n_steps: int,
        buffers: dict[str, np.ndarray | None],
        trackers: dict[str, bool],
    ) -> None:
        """Execute the simulation loop, recording state and dynamics each step."""
        for step in range(n_steps):
            requested = np.asarray(control_sequence[step], dtype=float)
            buffers["requested_controls"][step] = requested  # type: ignore[index]
            self.engine.set_control(requested)

            applied = self._read_applied_control(requested, trackers)
            buffers["applied_controls"][step] = applied  # type: ignore[index]

            # Refresh instantaneous dynamics so native labels match current u.
            self._refresh_instantaneous_dynamics()

            q, v = self.engine.get_state()
            t = self.engine.get_time()
            buffers["times"][step] = t  # type: ignore[index]
            buffers["positions"][step] = q  # type: ignore[index]
            buffers["velocities"][step] = v  # type: ignore[index]

            self._record_native_acceleration(step, buffers, trackers)
            self._record_dynamics_step(config, step, applied, v, buffers, trackers)

            self.engine.step(config.timestep)

            try:
                _, v_new = self.engine.get_state()
                buffers["interval_accelerations"][step] = (  # type: ignore[index]
                    (v_new - v) / config.timestep
                )
            except _OPTIONAL_CATCH:
                trackers["interval_accelerations"] = True

    def _refresh_instantaneous_dynamics(self) -> None:
        """Call ``forward()`` when available so native a matches current u."""
        forward = getattr(self.engine, "forward", None)
        if not callable(forward):
            return
        try:
            forward()
        except _OPTIONAL_CATCH:
            logger.debug(
                "engine.forward() unavailable; native acceleration may be stale",
                exc_info=True,
            )

    def _read_applied_control(
        self, requested: np.ndarray, trackers: dict[str, bool]
    ) -> np.ndarray:
        getter = getattr(self.engine, "get_applied_control", None)
        if callable(getter):
            try:
                return np.asarray(getter(), dtype=float).reshape(-1)
            except _OPTIONAL_CATCH:
                trackers["applied_controls"] = True
                return requested.copy()
        return requested.copy()

    def _record_native_acceleration(
        self,
        step: int,
        buffers: dict[str, np.ndarray | None],
        trackers: dict[str, bool],
    ) -> None:
        """Record native a after ``_refresh_instantaneous_dynamics``."""
        getter = getattr(self.engine, "get_joint_accelerations", None)
        if not callable(getter):
            trackers["native_accelerations"] = True
            return
        try:
            accel = np.asarray(getter(), dtype=float).reshape(-1).copy()
            target = buffers["native_accelerations"]
            if target is None:
                trackers["native_accelerations"] = True
                return
            n_v = target.shape[1]
            if accel.size < n_v:
                trackers["native_accelerations"] = True
                return
            target[step] = accel[:n_v]
        except _OPTIONAL_CATCH:
            trackers["native_accelerations"] = True

    def _record_dynamics_step(
        self,
        config: GeneratorConfig,
        step: int,
        tau: np.ndarray,
        v: np.ndarray,
        buffers: dict[str, np.ndarray | None],
        trackers: dict[str, bool],
    ) -> None:
        """Record optional dynamics quantities; failures mark trackers."""
        if config is None:
            raise ValueError("config must be provided")

        if config.record_mass_matrix and buffers["mass_matrices"] is not None:
            try:
                buffers["mass_matrices"][step] = self.engine.compute_mass_matrix()
            except _OPTIONAL_CATCH:
                trackers["mass_matrices"] = True

        if config.record_bias_forces and buffers["bias_forces"] is not None:
            try:
                buffers["bias_forces"][step] = self.engine.compute_bias_forces()
            except _OPTIONAL_CATCH:
                trackers["bias_forces"] = True

        if config.record_gravity and buffers["gravity"] is not None:
            try:
                buffers["gravity"][step] = self.engine.compute_gravity_forces()
            except _OPTIONAL_CATCH:
                trackers["gravity"] = True

        if config.record_contact_forces and buffers["contact"] is not None:
            try:
                cf = np.asarray(self.engine.compute_contact_forces(), dtype=float)
                buffers["contact"][step, : min(3, cf.size)] = cf.reshape(-1)[:3]
            except _OPTIONAL_CATCH:
                trackers["contact"] = True

        if config.record_drift_control:
            try:
                if buffers["drift"] is not None:
                    buffers["drift"][step] = self.engine.compute_drift_acceleration()
            except _OPTIONAL_CATCH:
                trackers["drift"] = True
            try:
                if buffers["control_accel"] is not None:
                    buffers["control_accel"][step] = (
                        self.engine.compute_control_acceleration(tau)
                    )
            except _OPTIONAL_CATCH:
                trackers["control_accel"] = True

        try:
            M = self.engine.compute_mass_matrix()
            buffers["kinetic_energy"][step] = 0.5 * float(v.T @ M @ v)  # type: ignore[index]
        except _OPTIONAL_CATCH:
            trackers["kinetic_energy"] = True
        try:
            pe = getattr(self.engine, "compute_potential_energy", None)
            if callable(pe):
                buffers["potential_energy"][step] = float(pe())  # type: ignore[index]
            else:
                trackers["potential_energy"] = True
        except _OPTIONAL_CATCH:
            trackers["potential_energy"] = True


__all__ = ["_OPTIONAL_CATCH", "_SimRecordingMixin"]
