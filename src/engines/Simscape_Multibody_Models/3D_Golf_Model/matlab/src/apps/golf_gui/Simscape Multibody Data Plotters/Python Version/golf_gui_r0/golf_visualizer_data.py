# mypy: disable-error-code="no-redef,var-annotated"
"""MATLAB data loading helpers for the legacy golf swing visualizer."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import scipy.io

try:
    from .golf_visualizer_models import FrameData
except ImportError:
    from golf_visualizer_models import FrameData

logger = logging.getLogger(__name__)


class DataProcessor:
    """Optimized data loading and processing with Numba acceleration"""

    def __init__(self) -> None:
        self.cache = {}
        self.max_force_magnitude = 1.0
        self.max_torque_magnitude = 1.0

    def load_matlab_data(
        self, baseq_file: str, ztcfq_file: str, delta_file: str
    ) -> tuple[pd.DataFrame, ...]:
        """Fast MATLAB data loading with error handling"""
        if not (baseq_file is not None):
            raise ValueError("baseq_file must be provided")
        datasets: dict[str, pd.DataFrame] = {}
        files = {"BASEQ": baseq_file, "ZTCFQ": ztcfq_file, "DELTAQ": delta_file}
        for name, filepath in files.items():
            try:
                mat_data = scipy.io.loadmat(filepath)
                var_name = self._find_table_variable(mat_data, name)
                datasets[name] = self._extract_dataframe(mat_data[var_name])
                logger.info(f"Loaded {name}: {len(datasets[name])} frames")
            except (RuntimeError, TypeError, ValueError) as e:
                raise RuntimeError(f"Failed to load {filepath}: {e}") from e
        self._calculate_scaling_factors(datasets["BASEQ"])
        return datasets["BASEQ"], datasets["ZTCFQ"], datasets["DELTAQ"]

    def _find_table_variable(self, mat_data: dict, dataset_name: str) -> str:
        """Intelligently find the table variable in MAT file"""
        candidates = [f"{dataset_name}_table", dataset_name, dataset_name.lower()]
        for var_name in candidates:
            if var_name in mat_data:
                return var_name
        vars_found = [k for k in mat_data if not k.startswith("__")]
        if vars_found:
            return vars_found[0]
        raise ValueError(f"No valid table found in {dataset_name}")

    def _extract_dataframe(self, mat_table: np.ndarray) -> pd.DataFrame:
        """Convert a MATLAB table-like structure into a pandas DataFrame."""
        if not hasattr(mat_table, "dtype") or mat_table.dtype.names is None:
            raise ValueError("Invalid MATLAB table structure")

        data: dict[str, list[object]] = {}
        for column in mat_table.dtype.names:
            column_values: list[object] = []
            for raw_value in mat_table[column]:
                value = (
                    raw_value.squeeze() if hasattr(raw_value, "squeeze") else raw_value
                )
                if isinstance(value, np.ndarray):
                    if value.size == 1:
                        column_values.append(value.item())
                    else:
                        column_values.append(value.tolist())
                else:
                    column_values.append(value)
            data[column] = column_values

        return pd.DataFrame(data)

    def _calculate_scaling_factors(self, baseq_data: pd.DataFrame) -> None:
        """Calculate scaling factors from data"""
        try:
            self.max_force_magnitude = 2000.0
            self.max_torque_magnitude = 200.0
            logger.info(
                f"Scaling factors set: Force={self.max_force_magnitude}N, "
                f"Torque={self.max_torque_magnitude}Nm"
            )
        except (RuntimeError, ValueError, OSError) as e:
            logger.info(f"Error calculating scaling factors: {e}")
            self.max_force_magnitude = 1000.0
            self.max_torque_magnitude = 100.0

    def extract_frame_data(
        self, frame_idx: int, datasets: dict[str, pd.DataFrame]
    ) -> FrameData:
        """Extract and process single frame data efficiently"""
        if not (frame_idx is not None):
            raise ValueError("frame_idx must be provided")
        if frame_idx in self.cache:
            return self.cache[frame_idx]
        frame_data = FrameData(
            frame_idx=frame_idx,
            time=frame_idx * 0.001,
            butt=self._safe_extract_point(datasets["BASEQ"], frame_idx, "Butt"),
            clubhead=self._safe_extract_point(datasets["BASEQ"], frame_idx, "Clubhead"),
            midpoint=self._safe_extract_point(datasets["BASEQ"], frame_idx, "MidPoint"),
            left_wrist=self._safe_extract_point(
                datasets["BASEQ"], frame_idx, "LeftWrist"
            ),  # noqa: E501
            left_elbow=self._safe_extract_point(
                datasets["BASEQ"], frame_idx, "LeftElbow"
            ),  # noqa: E501
            left_shoulder=self._safe_extract_point(
                datasets["BASEQ"], frame_idx, "LeftShoulder"
            ),  # noqa: E501
            right_wrist=self._safe_extract_point(
                datasets["BASEQ"], frame_idx, "RightWrist"
            ),  # noqa: E501
            right_elbow=self._safe_extract_point(
                datasets["BASEQ"], frame_idx, "RightElbow"
            ),  # noqa: E501
            right_shoulder=self._safe_extract_point(
                datasets["BASEQ"], frame_idx, "RightShoulder"
            ),  # noqa: E501
            hub=self._safe_extract_point(datasets["BASEQ"], frame_idx, "Hub"),
            forces={
                "BASEQ": self._safe_extract_vector(
                    datasets["BASEQ"], frame_idx, "TotalHandForceGlobal"
                ),
                "ZTCFQ": self._safe_extract_vector(
                    datasets["ZTCFQ"], frame_idx, "TotalHandForceGlobal"
                ),
                "DELTAQ": self._safe_extract_vector(
                    datasets["DELTAQ"], frame_idx, "TotalHandForceGlobal"
                ),
            },
            torques={
                "BASEQ": self._safe_extract_vector(
                    datasets["BASEQ"], frame_idx, "EquivalentMidpointCoupleGlobal"
                ),
                "ZTCFQ": self._safe_extract_vector(
                    datasets["ZTCFQ"], frame_idx, "EquivalentMidpointCoupleGlobal"
                ),
                "DELTAQ": self._safe_extract_vector(
                    datasets["DELTAQ"], frame_idx, "EquivalentMidpointCoupleGlobal"
                ),
            },
        )
        self.cache[frame_idx] = frame_data
        return frame_data

    def _safe_extract_point(
        self, dataset: pd.DataFrame, frame_idx: int, column: str
    ) -> np.ndarray:  # noqa: E501
        """Safely extract 3D point with fallbacks"""
        if not (dataset is not None):
            raise ValueError("dataset must be provided")
        try:
            point = dataset[column].iloc[frame_idx]
            if isinstance(point, list | np.ndarray) and len(point) == 3:
                return np.array(point, dtype=np.float32)
        except (TypeError, ValueError, IndexError):
            pass
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def _safe_extract_vector(
        self, dataset: pd.DataFrame, frame_idx: int, column: str
    ) -> np.ndarray:  # noqa: E501
        """Safely extract 3D vector with fallbacks"""
        if not (dataset is not None):
            raise ValueError("dataset must be provided")
        try:
            vector = dataset[column].iloc[frame_idx]
            if isinstance(vector, list | np.ndarray) and len(vector) == 3:
                return np.array(vector, dtype=np.float32)
        except (TypeError, ValueError, IndexError):
            pass
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
