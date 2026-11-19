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

    The collar is slanted: highest at the back, at the neck point (C7), and
    lower at the front (suprasternal notch). ``min_front_visible_neck``: chin
    to the front of the trunk top (anthropometry: 8 to 11 cm of a 24 cm head);
    ``max_back_top_above_neck_point``: the back of the trunk top;
    ``min_collar_slant``: back top minus front top; ``max_opening_over_neck_width`` and
    ``max_collar_area_over_neck``: the crew-neck opening hugs the neck (no
    scoop, no flat lid); ``min_cap_below_neck_point``: shoulder cap and yoke
    tops below the neck point at the address pose (head frame);
    ``max_cap_behind_joint``:
    how far the upper arm may extend behind its shoulder joint, in arm radii
    (the joint is 4 cm below C7, so a cap on the joint is 3 cm or more below
    the neck point once the arm hangs); ``min_cap_below_chin`` and
    ``neck_zone_half_width``: the yoke near the neck stays below the chin;
    ``slope_deg_band``: the trapezius, shoulder line up to the neck base;
    ``neck_width`` / ``neck_depth`` (bands) and ``max_base_bulge`` (base over
    mid width) describe the neck column, 12 by 11 cm for a 1.75 m man.
    """

    min_front_visible_neck: float = 0.30
    max_back_top_above_neck_point: float = 0.05
    min_collar_slant: float = 0.20
    max_opening_over_neck_width: float = 1.4  # collar opening / neck, across
    max_collar_area_over_neck: float = 1.8  # collar opening / neck, area (no lid)
    shoulder_line_height: float = -0.45  # rings above this, head lengths below C7
    min_collar_side_angle_deg: float = 15.0  # side meridian above it, from level
    max_collar_kink_deg: float = 20.0  # slope change between ring steps
    rounded_collar_rings: int = 1  # top rings that must be round, not boxy
    max_collar_fill: float = 0.82  # ring area / bounding box (ellipse 0.785)
    collar_rings: int = 5  # rings from the shoulder line up to the opening
    min_collar_wall_slope: float = 0.27  # rise over run of the shirt there (15 deg)
    min_cap_below_neck_point: float = 0.04  # caps and yoke at address (1 cm)
    max_cap_behind_joint: float = 0.1  # arm radii the cap may rise past its joint
    min_cap_below_chin: float = 0.08
    neck_zone_half_width: float = 0.5
    slope_deg_band: tuple[float, float] = (10.0, 40.0)
    neck_width: tuple[float, float] = (0.45, 0.55)
    neck_depth: tuple[float, float] = (0.38, 0.50)
    max_base_bulge: float = 1.12
    max_neck_fill: float = 0.79  # neck section area / bounding box (ellipse 0.785)
    neck_mid_height: float = -0.12  # where the neck width over head width is read
    neck_over_head_width: tuple[float, float] = (0.64, 0.72)
    max_neck_past_head_m: float = 0.002  # neck above the jaw, outside the head skin
    neck_head_overlap_from: float = 0.04  # head lengths up from the neck point
    max_opening_off_axis_m: float = 0.003  # top ring centre to neck axis (default)
    opening_gap_m: tuple[float, float] = (0.002, 0.009)  # ring to neck skin, all round
    max_neck_ahead_of_opening_m: float = 0.015  # at address (28 mm before round six)


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
