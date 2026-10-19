"""Elastic-foundation parameters matched to the shared pad law (issue #11739).

The OpenSim ``contact`` grip uses ``ElasticFoundationForce``: the closed grip
mesh is a Winkler foundation (each face pushes back with ``k * depth * area``,
times ``1 + c * depth_rate``) and each hand pad is a rigid sphere.  For a
sphere of radius ``R_p`` pressed ``delta`` into a cylinder of radius ``R_g``
the gap closes as a quadratic form with principal curvatures
``A = 1/R_p + 1/R_g`` (around the grip) and ``B = 1/R_p`` (along it), the
contact patch is the ellipse of semi-axes ``sqrt(2 delta / A)`` and
``sqrt(2 delta / B)``, and integrating the depth over it gives

    F(delta) = k * pi * delta**2 / sqrt(A * B).

The shared pad law is linear, ``F = k_pad * delta`` (Hunt-Crossley).  The two
cannot agree everywhere, so they are matched at the operating point, with no
fitting:

* the same squeeze: ``F(delta0) = f_pad`` per pad;
* the same tangent stiffness: ``F'(delta0) = k_pad``.  For ``F ~ delta**2``
  this fixes the preload interference at ``delta0 = 2 f_pad / k_pad`` (twice
  the shared preload) and the foundation stiffness at
  ``k = f_pad / (pi delta0**2 / sqrt(A B))``.

Friction (static, dynamic, viscous, Stribeck transition speed) and the
dissipation coefficient ``c`` are those of the shared law.  The documented
differences are the quadratic (not linear) force-penetration curve away from
the preload and the doubled preload interference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from src.shared.python.grip_contact.pad_contact import PadContactModel
from src.shared.python.grip_contact.pad_layout import PadLayout
from src.shared.python.motion_matching.contact_law import ContactParameters

__all__ = [
    "ElasticFoundationParameters",
    "elastic_foundation_for_law",
    "elastic_foundation_parameters",
    "winkler_patch_factor_m",
]


def winkler_patch_factor_m(pad_radius_m: float, grip_radius_m: float) -> float:
    """``pi / sqrt(A B)`` [m]: ``F = k * factor * delta**2`` for one pad.

    Raises:
        ValueError: if a radius is not positive and finite.
    """
    for name, v in (("pad_radius_m", pad_radius_m), ("grip_radius_m", grip_radius_m)):
        if not (math.isfinite(v) and v > 0.0):
            raise ValueError(f"{name} must be positive and finite, got {v}")
    around = 1.0 / pad_radius_m + 1.0 / grip_radius_m
    along = 1.0 / pad_radius_m
    return math.pi / math.sqrt(around * along)


@dataclass(frozen=True)
class ElasticFoundationParameters:
    """``ElasticFoundationForce`` parameters of one pad and the matching layout.

    ``stiffness_n_m3`` is OpenSim's ``stiffness`` (force per unit area per unit
    depth).  ``layout`` is the shared layout with the elastic-foundation
    preload interference.  ``analytic_force_n`` is the Winkler estimate of a
    pad force at a given penetration; the OpenSim contact evaluates the real
    mesh, so a calibration against it (``calibrate_stiffness``) can refine
    ``stiffness_n_m3``.
    """

    stiffness_n_m3: float
    dissipation_s_m: float
    static_friction: float
    dynamic_friction: float
    viscous_friction: float
    transition_velocity_m_s: float
    preload_penetration_m: float
    force_per_pad_n: float
    pad_stiffness_n_m: float
    patch_factor_m: float
    layout: PadLayout

    def analytic_force_n(self, penetration_m: float) -> float:
        """Winkler force of one pad at ``penetration_m`` (zero if not pressed)."""
        d = max(penetration_m, 0.0)
        return self.stiffness_n_m3 * self.patch_factor_m * d * d

    def with_stiffness(self, stiffness_n_m3: float) -> ElasticFoundationParameters:
        """Copy with a calibrated stiffness.

        Raises:
            ValueError: if the stiffness is not positive and finite.
        """
        if not (math.isfinite(stiffness_n_m3) and stiffness_n_m3 > 0.0):
            raise ValueError("stiffness_n_m3 must be positive and finite")
        return replace(self, stiffness_n_m3=stiffness_n_m3)


def elastic_foundation_for_law(
    law: ContactParameters,
    grip_radius_m: float,
    pad_radius_m: float,
    preload_m: float,
    layout: PadLayout | None = None,
) -> ElasticFoundationParameters:
    """Match the elastic foundation to contact law and pad geometry at preload.

    Preconditions:
        ``law`` must be a :class:`ContactParameters` instance with positive stiffness.
        ``grip_radius_m``, ``pad_radius_m``, and ``preload_m`` must be positive finite floats.
        ``layout``, if provided, must be a :class:`PadLayout`.

    Postconditions:
        ``analytic_force_n(preload_penetration_m) == force_per_pad_n``.
        The analytic tangent stiffness at the operating point matches ``law.stiffness_n_m``.
    """
    if not isinstance(law, ContactParameters):
        raise TypeError(f"law must be ContactParameters, got {type(law).__name__}")
    for name, v in (
        ("grip_radius_m", grip_radius_m),
        ("pad_radius_m", pad_radius_m),
        ("preload_m", preload_m),
    ):
        if not (math.isfinite(v) and v > 0.0):
            raise ValueError(f"{name} must be positive and finite, got {v}")
    if layout is not None and not isinstance(layout, PadLayout):
        raise TypeError(
            f"layout must be PadLayout or None, got {type(layout).__name__}"
        )

    f_pad = law.stiffness_n_m * preload_m
    delta0 = 2.0 * f_pad / law.stiffness_n_m
    factor = winkler_patch_factor_m(pad_radius_m, grip_radius_m)
    if layout is None:
        target_layout = PadLayout(
            grip_radius_m=grip_radius_m,
            pad_radius_m=pad_radius_m,
            preload_penetration_m=delta0,
        )
    else:
        target_layout = replace(
            layout,
            grip_radius_m=grip_radius_m,
            pad_radius_m=pad_radius_m,
            preload_penetration_m=delta0,
        )
    return ElasticFoundationParameters(
        stiffness_n_m3=f_pad / (factor * delta0 * delta0),
        dissipation_s_m=law.dissipation_s_m,
        static_friction=law.static_friction,
        dynamic_friction=law.dynamic_friction,
        viscous_friction=law.viscous_friction,
        transition_velocity_m_s=law.transition_velocity_m_s,
        preload_penetration_m=delta0,
        force_per_pad_n=f_pad,
        pad_stiffness_n_m=law.stiffness_n_m,
        patch_factor_m=factor,
        layout=target_layout,
    )


def elastic_foundation_parameters(pads: PadContactModel) -> ElasticFoundationParameters:
    """Match the elastic foundation to ``pads`` at the preload operating point.

    Postconditions: ``analytic_force_n(preload) == force_per_pad_n`` and the
    analytic tangent stiffness at the preload equals the shared pad stiffness.
    """
    return elastic_foundation_for_law(
        pads.law,
        pads.layout.grip_radius_m,
        pads.layout.pad_radius_m,
        pads.layout.preload_penetration_m,
        layout=pads.layout,
    )
