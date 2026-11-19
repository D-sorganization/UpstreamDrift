"""Visual neck of the CC0 head: a tapered, forward-leaning solid (#15292).

The CC0 head ends a hand's breadth under the jaw, and the neck point (the head
body's origin) sits only a few centimetres below the chin, so without a neck
solid the shirt collar meets the jaw. The neck here is its own solid in the
canonical head frame (origin at the neck point, x forward, y left, z up): it
reaches down into the trunk top, so the collar hides its base, and up into the
head's own neck stub, so the jaw hides its top. Every size is a fraction of
the head length, so it scales with the subject.

Visual only: nothing here carries mass, moves a joint or resizes the head.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.torso_profile import (
    TorsoFrame,
    TorsoProfile,
    TorsoStation,
    torso_mesh,
)


@dataclass(frozen=True)
class NeckSettings:
    """Neck solid, in head lengths from the neck point (z up, x forward)."""

    base_height: float = -0.46  # axis start (inside the slanted collar)
    top_height: float = 0.13  # axis end (inside the head's neck stub)
    base_forward: float = -0.02  # axis start, forward of the neck point
    top_forward: float = 0.14  # axis end: the forward lean of the neck
    half_width: float = 0.27  # base half width: 13 cm across for a 1.75 m man
    half_depth: float = 0.24  # base half depth: 11.7 cm front to back
    #: (along, width, depth): sections as fractions of the base half extents. The
    #: neck narrows up to the head's own neck stub (0.35 head lengths at the jaw)
    #: and eases wider towards the collar; it is 11 to 12.5 cm across mid-height.
    stations: tuple[tuple[float, float, float], ...] = (
        (0.0, 1.0, 1.0),
        (0.35, 0.93, 0.93),
        (0.55, 0.86, 0.86),
        (0.69, 0.83, 0.83),
        (0.78, 0.78, 0.78),
        (0.86, 0.69, 0.69),
        (1.0, 0.6, 0.6),
    )
    squareness: float = 2.0

    def __post_init__(self) -> None:
        values = (self.base_height, self.top_height, self.base_forward)
        if not (np.isfinite(values).all() and self.top_height > self.base_height):
            raise ValueError("neck must rise from its base to its top")
        if min(self.half_width, self.half_depth) <= 0.0:
            raise ValueError("neck half extents must be positive")


@dataclass(frozen=True)
class NeckAcceptance:
    """Acceptance floors of the look of the neck (fractions of head length).

    ``neck_width`` / ``neck_depth`` (bands) and ``max_base_bulge`` (base over
    mid width) describe the neck column, 12 by 11 cm for a 1.75 m man.
    ``max_neck_fill``: section area over bounding box (round, not boxy);
    ``neck_over_head_width`` at ``neck_mid_height``; ``max_neck_past_head_m``:
    how far the neck may stand outside the head skin above
    ``neck_head_overlap_from`` (no step at the jaw).
    """

    neck_width: tuple[float, float] = (0.45, 0.55)
    neck_depth: tuple[float, float] = (0.38, 0.50)
    max_base_bulge: float = 1.12
    max_neck_fill: float = 0.79  # neck section area / bounding box (ellipse 0.785)
    neck_mid_height: float = -0.12  # where the neck width over head width is read
    neck_over_head_width: tuple[float, float] = (0.64, 0.72)
    max_neck_past_head_m: float = 0.002  # neck above the jaw, outside the head skin
    neck_head_overlap_from: float = 0.04  # head lengths up from the neck point


NECK_ACCEPTANCE = NeckAcceptance()


def build_neck(length_m: float, settings: NeckSettings | None = None) -> Mesh:
    """The neck solid in the canonical head frame for a head of ``length_m``.

    Preconditions: positive finite head length. Postconditions: closed,
    positive volume, mirror-symmetric about the mid-sagittal plane; every
    coordinate scales with ``length_m``; it reaches below the neck point.
    """
    if not math.isfinite(length_m) or length_m <= 0:
        raise ValueError("Head length must be a positive finite number")
    cfg = settings or NeckSettings()
    frame = TorsoFrame(
        length_m * np.array([cfg.base_forward, 0.0, cfg.base_height]),
        length_m * np.array([cfg.top_forward, 0.0, cfg.top_height]),
        np.array([0.0, 1.0, 0.0]),
        length_m * cfg.half_width,
        length_m * cfg.half_depth,
    )
    profile = TorsoProfile(
        tuple(TorsoStation(*row) for row in cfg.stations),
        cfg.squareness,
        rings=16,
        sides=36,
    )
    return torso_mesh(frame, profile)


__all__ = ["NECK_ACCEPTANCE", "NeckAcceptance", "NeckSettings", "build_neck"]
