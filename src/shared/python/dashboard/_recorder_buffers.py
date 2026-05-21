from __future__ import annotations

from typing import Any, cast

import numpy as np

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


class _BuffersMixin:
    data: dict[str, Any]
    current_idx: int
    current_capacity: int
    initial_capacity: int
    max_samples: int
    growth_factor: float
    is_recording: bool
    _buffers_initialized: bool
    analysis_config: dict[str, Any]

    def _reset_buffers(self) -> None:
        self.current_idx = 0
        self.current_capacity = self.initial_capacity
        self._buffers_initialized = False
        self.data = {
            "times": np.zeros(self.current_capacity),
            "kinetic_energy": np.zeros(self.current_capacity),
            "potential_energy": np.zeros(self.current_capacity),
            "total_energy": np.zeros(self.current_capacity),
            "club_head_speed": np.zeros(self.current_capacity),
            "joint_positions": None,
            "joint_velocities": None,
            "joint_accelerations": None,
            "joint_torques": None,
            "actuator_powers": None,
            "angular_momentum": None,
            "club_head_position": None,
            "cop_position": None,
            "com_position": None,
            "ground_forces": None,
            "ground_moments": None,
            "ztcf_accel": None,
            "zvcf_accel": None,
            "drift_accel": None,
            "control_accel": None,
            "induced_accelerations": {},
            "counterfactuals": {},
        }

    def _ensure_capacity(self) -> None:
        if self.current_idx >= self.current_capacity:
            new_capacity = min(
                int(self.current_capacity * self.growth_factor), self.max_samples
            )

            if new_capacity <= self.current_capacity:
                logger.warning(
                    f"Recorder buffer full at {self.max_samples} samples. "
                    "Stopping recording."
                )
                self.is_recording = False
                return

            logger.debug(
                f"Growing recorder buffers from {self.current_capacity} "
                f"to {new_capacity} samples"
            )

            for key, arr in self.data.items():
                if isinstance(arr, np.ndarray):
                    new_shape = list(arr.shape)
                    new_shape[0] = new_capacity
                    new_arr = np.zeros(new_shape, dtype=arr.dtype)
                    new_arr[: self.current_capacity] = arr
                    self.data[key] = new_arr

            self.current_capacity = new_capacity

    def _ensure_buffers_allocated(self) -> None:
        if self.data["joint_velocities"] is None:
            return

        nv = self.data["joint_velocities"].shape[1]

        if self.analysis_config["ztcf"] and self.data["ztcf_accel"] is None:
            self.data["ztcf_accel"] = np.zeros((self.max_samples, nv))
            logger.debug("Allocated ZTCF buffer dynamically.")

        if self.analysis_config["zvcf"] and self.data["zvcf_accel"] is None:
            self.data["zvcf_accel"] = np.zeros((self.max_samples, nv))
            logger.debug("Allocated ZVCF buffer dynamically.")

        if self.analysis_config["track_drift"] and self.data["drift_accel"] is None:
            self.data["drift_accel"] = np.zeros((self.max_samples, nv))
            logger.debug("Allocated Drift buffer dynamically.")

        if (
            self.analysis_config["track_total_control"]
            and self.data["control_accel"] is None
        ):
            self.data["control_accel"] = np.zeros((self.max_samples, nv))
            logger.debug("Allocated Control buffer dynamically.")

        sources = cast(list[int], self.analysis_config["induced_accel_sources"])
        for idx in sources:
            if idx not in self.data["induced_accelerations"]:
                self.data["induced_accelerations"][idx] = np.zeros(
                    (self.max_samples, nv)
                )
                logger.debug(f"Allocated Induced Accel buffer for source {idx}.")

    def _initialize_array_buffers(self, q: np.ndarray, v: np.ndarray) -> None:
        if q is None:
            raise ValueError("q must be provided")
        nq = len(q)
        nv = len(v)

        self.data["joint_positions"] = np.zeros((self.current_capacity, nq))
        self.data["joint_velocities"] = np.zeros((self.current_capacity, nv))
        self.data["joint_accelerations"] = np.zeros((self.current_capacity, nv))
        self.data["joint_torques"] = np.zeros((self.current_capacity, nv))
        self.data["actuator_powers"] = np.zeros((self.current_capacity, nv))
        self.data["angular_momentum"] = np.zeros((self.current_capacity, 3))
        self.data["club_head_position"] = np.zeros((self.current_capacity, 3))
        self.data["cop_position"] = np.zeros((self.current_capacity, 3))
        self.data["com_position"] = np.zeros((self.max_samples, 3))
        self.data["ground_forces"] = np.zeros((self.max_samples, 3))
        self.data["ground_moments"] = np.zeros((self.max_samples, 3))

        if self.analysis_config["ztcf"]:
            self.data["ztcf_accel"] = np.zeros((self.max_samples, nv))
        if self.analysis_config["zvcf"]:
            self.data["zvcf_accel"] = np.zeros((self.max_samples, nv))
        if self.analysis_config["track_drift"]:
            self.data["drift_accel"] = np.zeros((self.max_samples, nv))
        if self.analysis_config["track_total_control"]:
            self.data["control_accel"] = np.zeros((self.max_samples, nv))

        sources = cast(list[int], self.analysis_config["induced_accel_sources"])
        for idx in sources:
            self.data["induced_accelerations"][idx] = np.zeros((self.max_samples, nv))

        self._buffers_initialized = True
        logger.debug(
            f"Initialized recorder buffers: nq={nq}, nv={nv}, max_samples={self.max_samples}"
        )
