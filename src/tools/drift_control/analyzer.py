"""Data-backed drift-control force-ratio analysis.

The public tool works on generalized-force trajectories so MuJoCo, Pinocchio,
or offline exported expert trajectories can feed the same stable contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray


FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class ForceTrajectory:
    """Generalized-force trajectory for drift-control analysis."""

    drift_generalized_force: FloatArray
    control_generalized_force: FloatArray
    time: FloatArray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sample_count(self) -> int:
        return int(self.drift_generalized_force.shape[0])

    @property
    def dimensions(self) -> int:
        return int(self.drift_generalized_force.shape[1])


class DriftControlAnalyzer:
    """Compute drift-to-control generalized-force ratios for trajectories."""

    _DRIFT_KEYS = (
        "drift_generalized_force",
        "drift_force",
        "qfrc_bias",
        "f_x",
    )
    _CONTROL_KEYS = (
        "control_generalized_force",
        "control_force",
        "qfrc_actuator",
        "g_x_u",
    )

    def __init__(self, epsilon: float = 1e-12) -> None:
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        self.epsilon = float(epsilon)

    def load_expert_trajectory(self, npz_path: str | Path) -> ForceTrajectory:
        """Load an exported expert trajectory from a NumPy NPZ file."""
        path = Path(npz_path)
        if not path.exists():
            raise FileNotFoundError(path)

        with np.load(path, allow_pickle=False) as payload:
            drift = self._read_first_array(payload, self._DRIFT_KEYS)
            control = self._read_first_array(payload, self._CONTROL_KEYS)
            time = self._optional_array(payload, "time")

        return self._build_trajectory(drift=drift, control=control, time=time)

    def compute_ratio(self, trajectory: ForceTrajectory) -> FloatArray:
        """Compute rho(t)=||f(x)||/||g(x)u|| for each sample."""
        self._validate_trajectory(trajectory)
        drift_norm = np.linalg.norm(trajectory.drift_generalized_force, axis=1)
        control_norm = np.linalg.norm(trajectory.control_generalized_force, axis=1)
        denominator = np.maximum(control_norm, self.epsilon)
        return np.asarray(drift_norm / denominator, dtype=np.float64)

    def compute_ratio_from_arrays(
        self,
        drift_generalized_force: Any,
        control_generalized_force: Any,
    ) -> FloatArray:
        trajectory = self._build_trajectory(
            drift=np.asarray(drift_generalized_force, dtype=np.float64),
            control=np.asarray(control_generalized_force, dtype=np.float64),
            time=None,
        )
        return self.compute_ratio(trajectory)

    def summarize_ratio(self, ratio: Any) -> dict[str, float | int]:
        values = np.asarray(ratio, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("ratio must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(values)):
            raise ValueError("ratio values must be finite")
        return {
            "sample_count": int(values.size),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "mean": float(np.mean(values)),
        }

    def _build_trajectory(
        self,
        drift: Any,
        control: Any,
        time: Any | None,
    ) -> ForceTrajectory:
        trajectory = ForceTrajectory(
            drift_generalized_force=self._as_force_matrix(drift, "drift"),
            control_generalized_force=self._as_force_matrix(control, "control"),
            time=None if time is None else np.asarray(time, dtype=np.float64),
        )
        self._validate_trajectory(trajectory)
        return trajectory

    def _validate_trajectory(self, trajectory: ForceTrajectory) -> None:
        if (
            trajectory.drift_generalized_force.shape
            != trajectory.control_generalized_force.shape
        ):
            raise ValueError(
                "drift and control generalized forces must have same shape"
            )
        if trajectory.sample_count == 0:
            raise ValueError("trajectory must contain at least one sample")
        if trajectory.time is not None and trajectory.time.shape != (
            trajectory.sample_count,
        ):
            raise ValueError("time must have one value per trajectory sample")
        if not np.all(np.isfinite(trajectory.drift_generalized_force)):
            raise ValueError("drift generalized forces must be finite")
        if not np.all(np.isfinite(trajectory.control_generalized_force)):
            raise ValueError("control generalized forces must be finite")

    @staticmethod
    def _as_force_matrix(value: Any, name: str) -> FloatArray:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise ValueError(f"{name} generalized forces must be a 1-D or 2-D array")
        return np.asarray(array, dtype=np.float64)

    @staticmethod
    def _read_first_array(payload: Any, keys: tuple[str, ...]) -> FloatArray:
        for key in keys:
            if key in payload:
                return np.asarray(payload[key], dtype=np.float64)
        raise KeyError(f"NPZ file must contain one of: {', '.join(keys)}")

    @staticmethod
    def _optional_array(payload: Any, key: str) -> FloatArray | None:
        if key not in payload:
            return None
        return np.asarray(payload[key], dtype=np.float64)
