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
* Soft-tissue scale (context only, the numbers are NOT taken from these
  papers).  Serina, Mote and Rempel, "Force response of the fingertip pulp
  to repeated compression: effects of loading rate, loading angle, and
  anthropometry", J. Biomech. 30(10), 1035-1040, 1997, report a viscoelastic,
  rate-dependent, nonlinear pulp that is compliant below about 1 N and
  stiffens rapidly above it.  Wu, Dong, Rakheja, Schopper and Smutz, "A
  structural fingertip model for simulating of the biomechanics of tactile
  sensation", Med. Eng. Phys. 26(2), 165-175, 2004, give a hyperelastic and
  viscoelastic finite-element finger model.  Both existence and subject
  matter were confirmed by search (2026-10); the per-patch stiffness
  figure of 0.5 to 3 N/mm quoted in the first version of this note could
  not be confirmed and is withdrawn.  The stiffness default is therefore a
  stiff surrogate chosen so the bushing replaces the weld, not a measured
  tissue stiffness; the sensitivity runs (x0.1, x10) bracket it.
* Rotational stiffness follows a lever relation ``k_r = k_t * r_eff**2`` with
  ``r_eff = 0.04 m`` (half of a ~80 mm palm width).
* Damping is *designed*, not tuned: ``design_damping`` (module ``damping``)
  sets the damping coefficients so that every one of the six club modes has
  the modal damping ratio ``zeta = 0.7`` (the value with about 5 % overshoot
  and the fastest settling without sustained ringing).  The derivation and
  the reason a per-axis ``2 zeta sqrt(k m)`` is wrong (the lowest modes are
  pendulum modes of the club carried by a force pair of the two hands, which
  that formula leaves at zeta ~ 0.01) are in ``damping.py``.  The default
  ``default_bushing`` therefore carries *zero* damping; use
  ``GripInterface.from_spec`` to obtain the designed values.
* Hand-grip force context: Komi, Roberts and Rothberg, "Evaluation of thin,
  flexible sensors for time-resolved grip force measurement", Proc. IMechE
  Part C, J. Mech. Eng. Sci. 222, 1687-1700, 2008 (confirmed to exist; it
  includes a golf-shot grip measurement).  A second FingerTPS golf reference
  given in the first version could not be identified and is removed.

Only the existence, authors and venue of the three papers above were
confirmed; no numeric value in this module is sourced from them.  Treat all
numeric defaults as owner-reviewable engineering defaults.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Vec3 = tuple[float, float, float]

R_EFF_M = 0.04


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
    """Stiff-surrogate engineering stiffness defaults, zero damping.

    Damping depends on the club and hand geometry and is designed by
    :func:`src.shared.python.grip_contact.damping.design_damping`
    (``GripInterface.from_spec`` does this).
    """
    k_t = 1.0e6
    k_r = k_t * R_EFF_M**2
    zero = (0.0, 0.0, 0.0)
    return BushingParameters((k_t, k_t, k_t), (k_r, k_r, k_r), zero, zero)


@dataclass(frozen=True)
class ContactMaterial:
    """Contact parameters for the rubber grip against skin or glove.

    ``static_friction`` and ``dynamic_friction`` are consumed by the pad
    contact grip (``pad_contact``, OSV-7 phase 3).  The pad stiffness is not
    taken from ``stiffness_n_m2``: it is matched to the bushing translational
    stiffness (``pad_layout.matched_pad_parameters``).  ``stiffness_n_m2`` and
    ``dissipation_s_m`` remain engineering placeholders for an
    elastic-foundation (OpenSim) contact, which is not implemented; all values
    still need cited measurements before they count as data.
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
