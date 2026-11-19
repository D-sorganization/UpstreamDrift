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

    base_height: float = -0.34  # axis start (inside the trunk top)
    top_height: float = 0.13  # axis end (inside the head's neck stub)
    base_forward: float = -0.04  # axis start, forward of the neck point
    top_forward: float = 0.12  # axis end: the forward lean of the neck
    half_width: float = 0.27  # base half width (left-right)
    half_depth: float = 0.27  # base half depth (front-back)
    #: (along, width, depth): sections as fractions of the base half extents.
    stations: tuple[tuple[float, float, float], ...] = (
        (0.0, 1.0, 1.0),
        (0.5, 1.0, 1.0),
        (0.72, 1.02, 1.0),  # the neck point: covers the head asset's flared base
        (1.0, 0.64, 0.64),
    )
    squareness: float = 2.2

    def __post_init__(self) -> None:
        values = (self.base_height, self.top_height, self.base_forward)
        if not (np.isfinite(values).all() and self.top_height > self.base_height):
            raise ValueError("neck must rise from its base to its top")
        if min(self.half_width, self.half_depth) <= 0.0:
            raise ValueError("neck half extents must be positive")


@dataclass(frozen=True)
class NeckAcceptance:
    """Acceptance floors of the look of the neck (fractions of head length).

    ``min_visible_neck_head_fraction``: chin to the highest trunk vertex;
    ``min_cap_below_chin_head_fraction``: the highest yoke or trunk vertex
    below the chin (yoke: within ``neck_zone_half_width_head_fraction`` of the
    neck; the deltoid cap sits on the fixed shoulder joint); ``slope_deg_band``: the trapezius, from the shoulder line
    up to the neck base. Anthropometry (visible neck 8 to 11 cm, a third of
    the head height) is the aim; the floors are what the default model's
    fixed neck joint allows (see the REPORT for #15292).
    """

    min_visible_neck_head_fraction: float = 0.15
    min_cap_below_chin_head_fraction: float = 0.08
    neck_zone_half_width_head_fraction: float = 0.5  # where the yoke meets the neck
    slope_deg_band: tuple[float, float] = (10.0, 40.0)


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
        rings=10,
        sides=28,
    )
    return torso_mesh(frame, profile)


__all__ = ["NECK_ACCEPTANCE", "NeckAcceptance", "NeckSettings", "build_neck"]
