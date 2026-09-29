"""Relaxed rigid-body attachment residuals, independent of engine dynamics."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from src.shared.python.body_part_viz.fitters._kabsch import kabsch_rotation


def rigid_attachment_residuals(
    offsets_m: NDArray[np.float64],
    observed_m: NDArray[np.float64],
    body_names: Sequence[str],
) -> NDArray[np.float64]:
    """Return (frames, markers) Euclidean residuals of independent proper SE(3) fits.

    Every body receives an unconstrained pose at every frame. Thus aggregate RMS
    is a lower bound conditional on these fixed offsets; it is not a bound after
    attachment recalibration. Missing points must contain three NaNs and remain
    NaN in the result. One-point bodies fit exactly but do not identify rotation.
    Joint connectivity, dynamics, torque bounds and temporal continuity are all
    relaxed. This diagnostic must never be substituted for forward simulation.
    """
    offsets = np.asarray(offsets_m, dtype=float)
    observed = np.asarray(observed_m, dtype=float)
    if (
        offsets.ndim != 2
        or offsets.shape[1] != 3
        or not len(offsets)
        or not np.isfinite(offsets).all()
    ):
        raise ValueError("offsets must be finite with shape (markers, 3)")
    if observed.ndim != 3 or observed.shape[1:] != offsets.shape or not len(observed):
        raise ValueError("observations must have shape (frames, markers, 3)")
    if len(body_names) != len(offsets) or any(
        not isinstance(name, str) or not name.strip() for name in body_names
    ):
        raise ValueError("one nonempty body name is required per marker")
    missing = np.isnan(observed).all(axis=2)
    if not np.isfinite(observed[~missing]).all() or missing.all():
        raise ValueError(
            "observations must be finite or whole missing points, with some observations"
        )
    names = np.asarray(body_names)
    residuals = np.full(observed.shape[:2], np.nan)
    for body in np.unique(names):
        indices = np.flatnonzero(names == body)
        for frame in range(len(observed)):
            chosen = indices[~missing[frame, indices]]
            if not len(chosen):
                continue
            source = offsets[chosen] - offsets[chosen].mean(axis=0)
            target = observed[frame, chosen] - observed[frame, chosen].mean(axis=0)
            rotation = kabsch_rotation(source, target)
            residuals[frame, chosen] = np.linalg.norm(
                source @ rotation.T - target, axis=1
            )
    return residuals
