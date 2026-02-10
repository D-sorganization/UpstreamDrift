"""Principal Component Analysis for kinematic data.

Includes PCA computation, principal movements, and kinematic sequence analysis.
"""

from __future__ import annotations

import numpy as np

from src.shared.python.analysis.dataclasses import (
    KinematicSequenceInfo,
    PCAResult,
)


class PCAAnalysisMixin:
    """Mixin for PCA and kinematic sequence analysis.

    Expects the following attributes to be available on the instance:
    - times: np.ndarray
    - joint_positions: np.ndarray
    - joint_velocities: np.ndarray
    """

    times: np.ndarray
    joint_positions: np.ndarray
    joint_velocities: np.ndarray

    def compute_principal_component_analysis(
        self,
        n_components: int | None = None,
        data_type: str = "position",
    ) -> PCAResult | None:
        """Compute Principal Component Analysis (PCA) on joint data.

        Also known as "Principal Movements" when applied to kinematic data.
        Identifies the main modes of variation in the movement.

        Args:
            n_components: Number of components to retain (default: all)
            data_type: 'position', 'velocity'

        Returns:
            PCAResult object or None
        """
        if data_type == "position":
            data = self.joint_positions
        else:
            data = self.joint_velocities

        if data.shape[1] == 0 or len(data) == 0:
            return None

        mean = np.mean(data, axis=0)
        centered_data = data - mean

        try:
            U, S, Vt = np.linalg.svd(centered_data, full_matrices=False)
        except np.linalg.LinAlgError:
            return None

        n_samples = data.shape[0]
        explained_variance = (S**2) / (n_samples - 1)
        total_var = np.sum(explained_variance)
        explained_variance_ratio = (
            explained_variance / total_var if total_var > 0 else np.zeros_like(S)
        )

        components = Vt
        projected_data = U * S

        if n_components is not None:
            n_components = min(n_components, len(explained_variance))
            components = components[:n_components]
            explained_variance = explained_variance[:n_components]
            explained_variance_ratio = explained_variance_ratio[:n_components]
            projected_data = projected_data[:, :n_components]

        return PCAResult(
            components=components,
            explained_variance=explained_variance,
            explained_variance_ratio=explained_variance_ratio,
            projected_data=projected_data,
            mean=mean,
        )

    def compute_principal_movements(
        self, n_modes: int = 3
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Compute Principal Movements (PMs) from position data.

        Wrapper around PCA specifically for position data to analyze
        primary coordination modes.

        Args:
            n_modes: Number of PMs to return

        Returns:
            (eigenvectors, scores) or None
        """
        result = self.compute_principal_component_analysis(
            n_components=n_modes, data_type="position"
        )
        if result:
            return result.components, result.projected_data
        return None

    def analyze_kinematic_sequence(
        self,
        segment_indices: dict[str, int],
    ) -> tuple[list[KinematicSequenceInfo], float]:
        """Analyze the kinematic sequence of the swing.

        The kinematic sequence refers to the proximal-to-distal sequencing of
        peak rotational velocities.

        Args:
            segment_indices: Dictionary mapping segment names to joint indices.

        Returns:
            Tuple of:
            - List of KinematicSequenceInfo objects sorted by peak time
            - Sequence efficiency score (0.0 to 1.0)
        """
        sequence_info = []

        for segment_name, joint_idx in segment_indices.items():
            if joint_idx >= self.joint_velocities.shape[1]:
                continue

            velocities = np.abs(self.joint_velocities[:, joint_idx])

            max_idx = np.argmax(velocities)
            peak_val = float(velocities[max_idx])
            peak_time = float(self.times[max_idx])

            sequence_info.append(
                KinematicSequenceInfo(
                    segment_name=segment_name,
                    peak_velocity=peak_val,
                    peak_time=peak_time,
                    peak_index=int(max_idx),
                    order_index=0,
                ),
            )

        sequence_info.sort(key=lambda x: x.peak_time)

        for i, info in enumerate(sequence_info):
            info.order_index = i

        expected_order = list(segment_indices.keys())
        actual_order = [info.segment_name for info in sequence_info]

        matches = sum(
            1 for e, a in zip(expected_order, actual_order, strict=False) if e == a
        )
        efficiency_score = matches / len(expected_order) if expected_order else 0.0

        return sequence_info, efficiency_score
