"""Distributed pad layout of the contact grip model (issue #11739, OSV-7 phase 3).

One engine-agnostic description of the ``contact`` grip: per hand, rings of
spherical finger and palm pads that press on a rigid cylindrical grip of the
club.  Every engine builds its contact from this one layout, so the engines
differ in their contact solver only.

Frames and geometry.  All pad positions are in the *grip frame* of their hand
(x along the grip axis, y and z across it, origin on the axis at that hand's
grip point; the hand and club grip frames coincide when the club is held).

* ``rows`` pad rings at axial offsets ``+-axial_offset_m``; each ring has
  ``pads_per_ring`` pads at equal angles about the axis.  The right-hand rings
  are turned by half a pad spacing, so the two hands' pads do not coincide
  where the hands abut.
* The grip frame origin of each hand sits on the hand's contact surface, one
  grip radius from the shaft axis (the spec's standoff): the axis is at
  ``+r`` along the grip-frame z axis for the right hand and ``-r`` for the
  left hand (the two origins are ``2 r`` apart across the grip).
* A pad centre rests at radial distance ``grip_radius_m + pad_radius_m -
  preload_penetration_m`` from the axis: the interference is the closure
  (squeeze) of the hand.

Matching to the bushing (:func:`matched_pad_parameters`, no fitting).  For
``n`` pads at equal angles the linearised pad stiffness of a hand against a
translation across the axis is ``k_pad * n / 2`` (``sum cos^2 = n/2``) and
against a rotation about a transverse axis ``k_pad * (n / 2) * a^2`` for rings
at ``+-a``.  So ``k_pad = 2 K_t / n + f_pad / rho`` (the second term restores the negative
geometric stiffness of a preloaded pad, ``rho`` = pad-centre distance from the
axis) reproduces the bushing translational
stiffness across the grip, and ``a = sqrt(K_r / K_t)`` (which is the bushing's
own ``R_EFF`` of 0.04 m) its bending stiffness.  The Hunt-Crossley dissipation
``c`` gives a damping coefficient ``k_pad * delta0 * c = f_pad * c`` per pad, so
``c = c_t / (N_total / 2)`` reproduces the bushing translational damping across
the grip.  There is no stiffness against translation along the axis or against
rotation about it: those loads are carried by Coulomb friction only (a
documented difference from the bushing, not a defect).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.shared.python.grip_contact.parameters import BushingParameters

#: Grip radius [m]: the spec's own hand standoff (``RHandStandoff`` placement
#: z = -0.0127 m, half an inch).
GRIP_RADIUS_M = 0.0127
#: Pad (finger and palm patch) sphere radius [m]: an engineering default.
PAD_RADIUS_M = 0.008

__all__ = [
    "GRIP_RADIUS_M",
    "PAD_RADIUS_M",
    "MatchedPadParameters",
    "PadLayout",
    "matched_pad_parameters",
    "required_squeeze_n",
]


@dataclass(frozen=True)
class PadLayout:
    """Pad rings of one hand on the grip cylinder.

    Preconditions: positive radii, ``pads_per_ring >= 3`` (so the ring is
    isotropic across the axis), ``rows`` of one or two, ``0 <= penetration <
    pad_radius``.
    """

    grip_radius_m: float = GRIP_RADIUS_M
    pad_radius_m: float = PAD_RADIUS_M
    axial_offset_m: float = 0.04
    pads_per_ring: int = 6
    rows: int = 2
    preload_penetration_m: float = 5.0e-4

    def __post_init__(self) -> None:
        for name in ("grip_radius_m", "pad_radius_m", "axial_offset_m"):
            v = getattr(self, name)
            if not math.isfinite(v) or v <= 0.0:
                raise ValueError(f"{name} must be positive and finite, got {v}")
        if self.pads_per_ring < 3:
            raise ValueError("pads_per_ring must be at least 3 (isotropic ring)")
        if self.rows not in (1, 2):
            raise ValueError("rows must be 1 or 2")
        if not 0.0 <= self.preload_penetration_m < self.pad_radius_m:
            raise ValueError("preload_penetration_m must be in [0, pad_radius_m)")

    @property
    def pad_count(self) -> int:
        """Pads per hand."""
        return self.rows * self.pads_per_ring

    @property
    def centre_radius_m(self) -> float:
        """Radial distance of a pad centre from the grip axis at rest."""
        return self.grip_radius_m + self.pad_radius_m - self.preload_penetration_m

    def axial_offsets_m(self) -> tuple[float, ...]:
        """Axial ring offsets (a single ring sits at the grip point)."""
        a = self.axial_offset_m
        return (-a, a) if self.rows == 2 else (0.0,)

    def axis_offset_grip_frame(self, side: str) -> np.ndarray:
        """Grip-axis point nearest the grip-frame origin of ``side``."""
        if side not in ("L", "R"):
            raise ValueError(f"side must be 'L' or 'R', got {side!r}")
        return np.array(
            [0.0, 0.0, self.grip_radius_m if side == "R" else -self.grip_radius_m]
        )

    def positions_grip_frame(self, side: str) -> np.ndarray:
        """Pad centres ``(pad_count, 3)`` in the grip frame of ``side``."""
        centre = self.axis_offset_grip_frame(side)
        phase = 0.0 if side == "L" else math.pi / self.pads_per_ring
        angles = (
            phase + 2.0 * math.pi * np.arange(self.pads_per_ring) / self.pads_per_ring
        )
        ring = self.centre_radius_m * np.column_stack(
            [np.zeros_like(angles), np.cos(angles), np.sin(angles)]
        )
        out = []
        for a in self.axial_offsets_m():
            shifted = ring.copy()
            shifted[:, 0] = a
            out.append(shifted + centre)
        return np.vstack(out)


@dataclass(frozen=True)
class MatchedPadParameters:
    """Per-pad Hunt-Crossley stiffness and dissipation matched to a bushing.

    ``preload_penetration_m`` is the interference ``f_pad / k_pad`` that gives
    the requested squeeze; build the layout with it
    (``dataclasses.replace(layout, preload_penetration_m=...)``).
    """

    stiffness_n_m: float
    dissipation_s_m: float
    normal_force_per_pad_n: float
    squeeze_per_hand_n: float
    preload_penetration_m: float
    axial_offset_m: float


def matched_pad_parameters(
    bushing: BushingParameters, layout: PadLayout, squeeze_per_hand_n: float
) -> MatchedPadParameters:
    """Pad stiffness and dissipation reproducing the bushing across the grip.

    ``squeeze_per_hand_n`` is the total pad normal force of one hand at rest.
    Postcondition: ``pad_count * stiffness * preload_penetration`` equals the
    squeeze.

    Raises:
        ValueError: if the squeeze is not positive and finite.
    """
    if not math.isfinite(squeeze_per_hand_n) or squeeze_per_hand_n <= 0.0:
        raise ValueError("squeeze_per_hand_n must be positive and finite")
    n = layout.pad_count
    k_t = float(np.mean(bushing.translational_stiffness_n_m[1:]))
    c_t = float(np.mean(bushing.translational_damping_ns_m[1:]))
    k_r = float(np.mean(bushing.rotational_stiffness_nm_rad[1:]))
    f_pad = squeeze_per_hand_n / n
    # Preloaded pads also turn with the offset (geometric, negative stiffness
    # f_pad / rho per pad); add it back so the net hand stiffness is K_t.
    rho = layout.grip_radius_m + layout.pad_radius_m
    k_pad = 2.0 * k_t / n + f_pad / rho
    return MatchedPadParameters(
        stiffness_n_m=k_pad,
        dissipation_s_m=c_t / (0.5 * squeeze_per_hand_n),
        normal_force_per_pad_n=f_pad,
        squeeze_per_hand_n=squeeze_per_hand_n,
        preload_penetration_m=f_pad / k_pad,
        axial_offset_m=math.sqrt(k_r / k_t),
    )


def required_squeeze_n(
    peak_axial_force_n: float,
    peak_axial_torque_nm: float,
    peak_transverse_force_n: float,
    friction: float,
    grip_radius_m: float,
) -> float:
    """Smallest total pad normal force of one hand that can carry the demand.

    The demand is the peak per-hand wrench of the bushing run, so the squeeze is
    derived, not tuned.  Three limits, the largest governs:

    * axial load by friction: ``N >= F_axial / mu``;
    * torque about the axis by friction at the grip radius:
      ``N >= tau_axial / (mu r)``;
    * transverse load without a ring losing contact: a ring of equal preload
      carries ``N / 2`` before the opposite pads unload, so ``N >= 2 F_perp``.

    Raises:
        ValueError: on a negative demand or non-positive friction or radius.
    """
    demands = (peak_axial_force_n, peak_axial_torque_nm, peak_transverse_force_n)
    if min(demands) < 0.0 or not all(math.isfinite(d) for d in demands):
        raise ValueError("demands must be finite and non-negative")
    if friction <= 0.0 or grip_radius_m <= 0.0:
        raise ValueError("friction and grip_radius_m must be positive")
    return max(
        peak_axial_force_n / friction,
        peak_axial_torque_nm / (friction * grip_radius_m),
        2.0 * peak_transverse_force_n,
    )
