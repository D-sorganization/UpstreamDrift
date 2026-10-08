"""Grip compliance and contact material parameters (issue #11739, OSV-7).

Engine-agnostic value objects.  Axis convention: components are expressed in
the *grip frame*, whose x axis is the grip (shaft) axis and whose y, z axes are
across the grip.  Translational stiffness is N/m, rotational stiffness N*m/rad.

Provenance of the default numbers (read before changing them):

* **Engineering defaults, not fitted.**  No published study reports a
  six-axis stiffness of a two-hand golf grip.  The defaults model a
  stiff-but-finite *surrogate* of the rigid grip so that the bushing is a
  determinate replacement of the weld.  They were fixed a priori from the
  relations below and are never tuned to a target; the deflection criterion is
  checked afterwards and the sensitivity is recorded in the modelling
  reference.
* Soft-tissue scale (lower bracket for sensitivity runs): fingertip pulp
  compresses at roughly 0.5 to 3 N/mm (Serina, Mote and Rempel, "Force
  response of the fingertip pulp to repeated compression", J. Biomech. 30(10),
  1997; Wu, Dong, Rakheja, Schopper and Smutz, "A structural fingertip model
  for simulating of the biomechanics of tactile sensation", Med. Eng. Phys.
  26(2), 2004).  A wrapped hand adds several such patches in parallel, so
  10 to 100 N/mm per hand is the soft-tissue range; the default is the stiff
  end of the bracket and beyond because squeeze preload stiffens the contact.
* Rotational stiffness follows a lever relation ``k_r = k_t * r_eff**2`` with
  ``r_eff = 0.04 m`` (half of a ~80 mm palm width).
* Damping uses a damping ratio ``zeta = 0.3`` (engineering default for
  tissue-dominated contact): translational ``c = 2 zeta sqrt(k m)`` on the club
  mass, rotational ``c = 2 zeta sqrt(k_r I_ref)`` per axis with
  ``I_ref = m L**2`` (``L = 0.6 m``, club centre of mass to the grip) for the
  pitch and yaw axes and the driver-head inertia about the shaft for the
  twist axis.  A fixed damping on the translational lever alone
  leaves the pitch mode with ``zeta ~ 0.01`` (measured), which does not settle.
* Hand-grip force magnitudes for plausibility: Komi, Roberts and Rothberg,
  "Evaluation of thin, flexible sensors for time-resolved grip force
  measurement", Proc. IMechE Part C 222, 2008, and the FingerTPS golf study
  at scitepress.org/Papers/2018/69623.

The citations above are given from the literature record and were not
re-verified against the full texts in this change; treat the numeric
defaults as owner-reviewable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Vec3 = tuple[float, float, float]

R_EFF_M = 0.04
ZETA = 0.3
_REFERENCE_CLUB_MASS_KG = 0.313
# Reference rotational inertia per grip axis (x = about the shaft, y/z = pitch
# and yaw about the hands): driver head inertia about the shaft; m * L**2.
_REFERENCE_CLUB_INERTIA_KG_M2 = (
    3.2e-4,
    _REFERENCE_CLUB_MASS_KG * 0.6**2,
    _REFERENCE_CLUB_MASS_KG * 0.6**2,
)


def _check_vec3(name: str, value: object, *, positive: bool) -> Vec3:
    try:
        vec = tuple(float(c) for c in value)  # type: ignore[attr-defined]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be three numbers") from exc
    if len(vec) != 3:
        raise ValueError(f"{name} must have exactly 3 components, got {len(vec)}")
    if not all(math.isfinite(c) for c in vec):
        raise ValueError(f"{name} must be finite, got {vec}")
    if positive and not all(c > 0.0 for c in vec):
        raise ValueError(f"{name} must be strictly positive, got {vec}")
    if not positive and not all(c >= 0.0 for c in vec):
        raise ValueError(f"{name} must be non-negative, got {vec}")
    return (vec[0], vec[1], vec[2])


@dataclass(frozen=True)
class BushingParameters:
    """Six-axis compliant hand-club interface, one instance per hand.

    Preconditions: stiffnesses strictly positive, dampings non-negative, all
    finite.  Postcondition: every field is a float 3-tuple (grip-frame axes).
    """

    translational_stiffness_n_m: Vec3
    rotational_stiffness_nm_rad: Vec3
    translational_damping_ns_m: Vec3
    rotational_damping_nms_rad: Vec3
    source: str = "engineering default (see module docstring)"

    def __post_init__(self) -> None:
        for name, positive in (
            ("translational_stiffness_n_m", True),
            ("rotational_stiffness_nm_rad", True),
            ("translational_damping_ns_m", False),
            ("rotational_damping_nms_rad", False),
        ):
            object.__setattr__(
                self, name, _check_vec3(name, getattr(self, name), positive=positive)
            )

    def scaled(self, factor: float) -> BushingParameters:
        """Return a copy with all stiffnesses scaled (sensitivity studies).

        Damping is rescaled by ``sqrt(factor)`` to hold the damping ratio.
        """
        if not math.isfinite(factor) or factor <= 0.0:
            raise ValueError(f"factor must be positive and finite, got {factor}")
        root = math.sqrt(factor)
        return BushingParameters(
            tuple(factor * k for k in self.translational_stiffness_n_m),  # type: ignore[arg-type]
            tuple(factor * k for k in self.rotational_stiffness_nm_rad),  # type: ignore[arg-type]
            tuple(root * c for c in self.translational_damping_ns_m),  # type: ignore[arg-type]
            tuple(root * c for c in self.rotational_damping_nms_rad),  # type: ignore[arg-type]
            source=f"{self.source}; stiffness x{factor:g}",
        )


def default_bushing() -> BushingParameters:
    """Stiff-surrogate engineering defaults (never tuned to a target)."""
    k_t = 1.0e6
    c_t = 2.0 * ZETA * math.sqrt(k_t * _REFERENCE_CLUB_MASS_KG)
    k_r = k_t * R_EFF_M**2
    c_r = tuple(2.0 * ZETA * math.sqrt(k_r * i) for i in _REFERENCE_CLUB_INERTIA_KG_M2)
    return BushingParameters(
        (k_t, k_t, k_t),
        (k_r, k_r, k_r),
        (c_t, c_t, c_t),
        c_r,  # type: ignore[arg-type]
    )


@dataclass(frozen=True)
class ContactMaterial:
    """Elastic-foundation contact parameters (placeholder for phase 2).

    Not consumed by any engine yet; values are engineering placeholders for
    rubber grip against skin or glove and must be replaced with cited values
    before the ``contact`` grip model is implemented.
    """

    stiffness_n_m2: float = 1.0e7
    dissipation_s_m: float = 1.0
    static_friction: float = 0.9
    dynamic_friction: float = 0.7
    viscous_friction: float = 0.0
    status: str = "placeholder (phase 2)"

    def __post_init__(self) -> None:
        for name in ("stiffness_n_m2", "dissipation_s_m", "viscous_friction"):
            v = getattr(self, name)
            if not math.isfinite(v) or v < 0.0:
                raise ValueError(f"{name} must be finite and non-negative, got {v}")
        if self.stiffness_n_m2 <= 0.0:
            raise ValueError("stiffness_n_m2 must be positive")
        if not 0.0 <= self.dynamic_friction <= self.static_friction:
            raise ValueError("require 0 <= dynamic_friction <= static_friction")
