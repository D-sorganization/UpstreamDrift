from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ...utils.logging import get_logger
from ._c3d_models import (
    BIOMECHANICAL_MARKER_MAX_M,
    BIOMECHANICAL_MARKER_MIN_M,
    C3DMetadata,
)

logger = get_logger(__name__)


def validate_marker_positions(
    coordinates: np.ndarray,
    source_units: str,
    target_units: str | None,
) -> None:
    """Validate marker positions per Guideline P1 (biomechanical range check).

    Raises:
        ValueError: If positions exceed the 10m sanity threshold.
    """
    if coordinates.size == 0:
        return

    min_pos = np.nanmin(coordinates)
    max_pos = np.nanmax(coordinates)

    if np.isnan(min_pos) or np.isnan(max_pos):
        logger.warning(
            "All marker coordinates are NaN or non-finite; skipping unit "
            "range validation (Guideline P1). Verify upstream data quality "
            "and missing-data handling."
        )
        return

    # Heuristic: only flag suspiciously small positive coordinates. Negative
    # values up to ~2 m are normal for swing data when the world origin is
    # placed at the target (clubs and body sweep behind the player on the
    # backswing). Filtering on |min_pos| < 1mm AND min_pos >= 0 avoids the
    # false positive seen on Tour-average driver/iron files where
    # min_pos ~= -1.97 m.
    if 0.0 <= min_pos < BIOMECHANICAL_MARKER_MIN_M:
        logger.warning(
            "Suspiciously small marker positions detected (< 1mm). "
            "Min position: %.6f m. Source units: %s, target: %s. "
            "Guideline P1: Verify unit conversion is correct to avoid 1000x errors.",
            min_pos,
            source_units,
            target_units or "unchanged",
        )

    if max_pos > BIOMECHANICAL_MARKER_MAX_M:
        logger.error(
            "\u274c Unrealistic marker positions detected (> 10m). "
            f"Max position: {max_pos:.2f}m. "
            f"Source units: {source_units}, target: "
            f"{target_units or 'unchanged'}. "
            "Guideline P1 VIOLATION: Likely unit conversion error."
        )
        raise ValueError(
            f"Marker positions exceed {BIOMECHANICAL_MARKER_MAX_M}m "
            f"(max: {max_pos:.2f}m) - likely unit error. "
            f"Check that source units '{source_units}' are correct. "
            "Common issue: mm labeled as m or vice versa."
        )


def build_points_dataframe(
    c3d_data: Any,
    metadata: C3DMetadata,
    file_name: str,
    scale: float,
    include_time: bool,
    markers: Sequence[str] | None,
    residual_nan_threshold: float | None,
    target_units: str | None,
) -> pd.DataFrame:
    """Build a tidy DataFrame of marker trajectories."""
    points = c3d_data["data"]["points"]
    marker_labels = np.array(metadata.marker_labels)

    if markers:
        mask = np.isin(marker_labels, list(markers))
        marker_labels = marker_labels[mask]
        points = points[:, mask, :]

    sort_indices = np.argsort(marker_labels)
    sorted_labels = marker_labels[sort_indices]
    points = points[:, sort_indices, :]

    raw_coordinates = np.transpose(points[:3, :, :], axes=(2, 1, 0)).reshape(-1, 3)
    coordinates = raw_coordinates * scale

    validate_marker_positions(coordinates, metadata.units, target_units)

    residuals = points[3, :, :].T.reshape(-1)

    if residual_nan_threshold is not None:
        too_noisy = residuals > residual_nan_threshold
        coordinates[too_noisy, :] = np.nan

    current_marker_count = len(sorted_labels)
    frame_indices = np.repeat(np.arange(metadata.frame_count), current_marker_count)
    marker_names = np.tile(sorted_labels, metadata.frame_count)

    data: dict[str, Any] = {
        "frame": frame_indices,
        "marker": marker_names,
        "x": coordinates[:, 0],
        "y": coordinates[:, 1],
        "z": coordinates[:, 2],
        "residual": residuals,
    }

    if include_time:
        if metadata.frame_rate > 0:
            data["time"] = frame_indices / metadata.frame_rate
        else:
            logger.warning(
                "Frame rate is 0. Time column will be omitted "
                "despite include_time=True."
            )

    dataframe = pd.DataFrame(data)
    dataframe = dataframe.reset_index(drop=True)

    logger.info(
        "Loaded %s frames for %s markers from %s",
        metadata.frame_count,
        current_marker_count,
        file_name,
    )
    return dataframe
