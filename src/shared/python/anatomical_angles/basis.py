"""Explicit basis maps from an engine world frame to the ISB world frame.

ISB (Wu et al. 2002/2005) uses X anterior, Y up (cranial), Z right. An engine
declares how its world axes relate to those by a :class:`BasisMap` instead of
hiding the permutation inside a segment-frame builder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BasisMap:
    """Rows of ``matrix`` are the ISB axes (X, Y, Z) expressed in engine axes.

    Preconditions: ``matrix`` is a proper rotation (orthonormal, det +1), so a
    mirror can never be introduced by a basis change; ``ValueError`` otherwise.
    A segment frame whose local axes are labelled like the engine world axes
    (``MyoFullBody``'s ``x`` right, ``y`` up, ``z`` posterior) maps by
    ``P R P^T``, which :meth:`apply_rotation` computes.
    """

    matrix: np.ndarray
    name: str

    def __post_init__(self) -> None:
        m = np.array(self.matrix, dtype=float)
        if m.shape != (3, 3) or not np.all(np.isfinite(m)):
            raise ValueError("basis matrix must be a finite 3x3 matrix")
        if not np.allclose(m @ m.T, np.eye(3), atol=1e-12) or not np.isclose(
            np.linalg.det(m), 1.0, atol=1e-12
        ):
            raise ValueError(
                f"basis {self.name!r} must be a proper rotation (no mirror)"
            )
        m.setflags(write=False)
        object.__setattr__(self, "matrix", m)

    def apply_vector(self, vector: object) -> np.ndarray:
        """``vector`` (engine axes, shape ``(..., 3)``) in ISB axes."""
        return np.einsum("ij,...j->...i", self.matrix, np.asarray(vector, float))

    def apply_rotation(self, rotation: object) -> np.ndarray:
        """``P R P^T`` of ``(..., 3, 3)`` rotations: world and local relabelled."""
        r = np.asarray(rotation, dtype=float)
        if r.shape[-2:] != (3, 3):
            raise ValueError("rotation must have trailing shape (3, 3)")
        return np.asarray(self.matrix @ r @ self.matrix.T)


#: Engine frame already ISB-labelled (X anterior, Y up, Z right).
ISB_IDENTITY = BasisMap(np.eye(3), "isb_identity")
#: ``MyoFullBody`` (``myofullbody/anatomy.py``): x right, y up, z posterior.
#: ISB X (anterior) = -z, Y = y, Z (right) = x.
ISB_FROM_MYOFULLBODY = BasisMap(
    np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    "myofullbody_x_right_y_up_z_posterior",
)
