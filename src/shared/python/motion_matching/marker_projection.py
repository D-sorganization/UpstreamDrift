"""Engine-independent projection of fixed body-local marker attachments."""

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def project_markers(
    frames: Mapping[str, ArrayLike], bodies: Sequence[str], offsets: ArrayLike
) -> NDArray[np.float64]:
    """Return world marker metres from body-to-world rigid transforms."""
    local = np.asarray(offsets, dtype=float)
    if (
        local.shape != (len(bodies), 3)
        or not len(bodies)
        or not np.isfinite(local).all()
    ):
        raise ValueError("Provide finite marker-by-xyz offsets and matching bodies")
    if any(body not in frames for body in bodies):
        raise ValueError("Marker body is missing from the frame inventory")
    transforms = np.asarray([frames[body] for body in bodies], dtype=float)
    if transforms.shape != (len(bodies), 4, 4) or not np.isfinite(transforms).all():
        raise ValueError("Each transform must be a finite 4-by-4 matrix")
    rotation = transforms[:, :3, :3]
    if (
        not np.allclose(transforms[:, 3, :], [0, 0, 0, 1], atol=1e-10, rtol=0)
        or not np.allclose(
            rotation.transpose(0, 2, 1) @ rotation, np.eye(3), atol=1e-8, rtol=0
        )
        or not np.allclose(np.linalg.det(rotation), 1, atol=1e-8, rtol=0)
    ):
        raise ValueError("Each transform must describe a proper rigid frame")
    return np.einsum("nij,nj->ni", rotation, local) + transforms[:, :3, 3]
