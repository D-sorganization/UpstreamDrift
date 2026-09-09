"""Estimate one reference placement from explicitly paired world landmarks."""

import numpy as np

from src.shared.python.body_part_viz.fitters._kabsch import kabsch_rotation

from .registration import ReferenceTransform


def estimate_reference_transform(
    reference_world: np.ndarray,
    golfer_world: np.ndarray,
    *,
    scale: bool = False,
) -> ReferenceTransform:
    """Return a fixed proper similarity from paired (N, 3) world points.

    Both inputs already use the camera world's Y-up convention. Missing pairs
    are excluded. At least three non-collinear correspondences are required.
    This is pose placement, not camera calibration. Apply the same result to
    every playback frame so differences after the anchor pose remain visible.
    """
    source = np.asarray(reference_world, dtype=float)
    target = np.asarray(golfer_world, dtype=float)
    if source.ndim != 2 or source.shape[-1] != 3 or source.shape != target.shape:
        raise ValueError("Correspondences must have matching (N, 3) shapes")
    valid = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source, target = source[valid], target[valid]
    if len(source) < 3:
        raise ValueError("At least three non-collinear pairs are required")
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    centered_source, centered_target = source - source_center, target - target_center
    if (
        min(
            np.linalg.matrix_rank(centered_source),
            np.linalg.matrix_rank(centered_target),
        )
        < 2
    ):
        raise ValueError("At least three non-collinear pairs are required")
    rotation = kabsch_rotation(centered_source, centered_target)
    factor = (
        float(
            np.sum((centered_source @ rotation.T) * centered_target)
            / np.sum(centered_source**2)
        )
        if scale
        else 1.0
    )
    translation = target_center - factor * (rotation @ source_center)
    return ReferenceTransform.model_validate(
        {
            "rotation": rotation.tolist(),
            "translation_m": translation.tolist(),
            "scale": factor,
            "body_size_normalized": scale,
        }
    )
