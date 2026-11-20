"""Soft stance anchor of the pose solve (#15296).

The legacy anchor ties a planted contact sphere to the point where it first
touched, in all three directions, at the ground weight (100 times a marker).
A planted foot in a swing still rolls and pivots by a few millimetres, so that
anchor costs marker fit; no anchor lets the feet creep by centimetres. The soft
anchor keeps the sphere on the ground at the ground weight and lets it slide
freely inside ``slide_m`` of its anchor; only the slide beyond that is
penalised, at ``slide_weight``. Engine-free: a kinematics passes the sphere's
gap to its anchor and the sphere's position Jacobian.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class SoftAnchor:
    """Rule of a soft anchor. Weights are those of a squared residual in
    metres, so 1 is the weight of a marker; ``slide_weight`` 0 frees the slide
    and ``slide_m`` 0 with ``slide_weight == normal_weight`` is the legacy
    anchor. Raises ``ValueError`` on a negative or non-finite value or a zero
    ``normal``."""

    normal: Sequence[float]
    normal_weight: float
    slide_weight: float
    slide_m: float

    def __post_init__(self) -> None:
        values = (self.normal_weight, self.slide_weight, self.slide_m)
        if not all(np.isfinite(value) and value >= 0 for value in values):
            raise ValueError("SoftAnchor weights and slide must be finite and >= 0")
        normal = np.asarray(self.normal, dtype=float)
        if normal.shape != (3,) or not np.linalg.norm(normal) > 0:
            raise ValueError("SoftAnchor normal must be a nonzero 3-vector")


def soft_anchor_rows(gap: Array, jac: Array, rule: SoftAnchor) -> tuple[Array, Array]:
    """Residual rows ``(4,)`` and their Jacobian ``(4, n)`` of one sphere.

    ``gap`` is the sphere centre minus its anchor target, ``jac`` ``(3, n)`` the
    centre's position Jacobian. Row 0 is the gap along the ground normal; rows
    1 to 3 are the in-plane slide beyond ``rule.slide_m``, zero inside it. The
    rows are continuous and so is their slope except on the allowance circle.
    """
    gap = np.asarray(gap, dtype=float)
    jac = np.asarray(jac, dtype=float)
    if gap.shape != (3,) or jac.ndim != 2 or jac.shape[0] != 3:
        raise ValueError("gap must be a 3-vector and jac (3, n)")
    normal = np.asarray(rule.normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    up, along = np.sqrt(rule.normal_weight), np.sqrt(rule.slide_weight)
    in_plane = np.eye(3) - np.outer(normal, normal)
    slide = in_plane @ gap
    distance = float(np.linalg.norm(slide))
    rows, slope = np.zeros(4), np.zeros((4, jac.shape[1]))
    rows[0], slope[0] = up * (normal @ gap), up * (normal @ jac)
    excess = distance - rule.slide_m
    if excess > 0.0:
        heading = slide / distance
        share = excess / distance
        turn = np.outer(heading, heading)
        rows[1:] = along * excess * heading
        slope[1:] = along * (turn + share * (in_plane - turn)) @ jac
    return rows, slope
