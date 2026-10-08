"""Fail-closed optional gateway to the Tools delivery / D-plane physics.

Tools owns ``swing_sim.impact.delivery.derive_delivery`` and
``swing_sim.impact.dplane.analyze_dplane`` (and
``rate_of_closure.simulation.delivery.delivery_at``).  This module owns no
delivery physics; it maps UpstreamDrift's target-line frame into the Tools
app frame and returns the Tools estimates, labelled as model estimates.

Frame mapping (UD world -> Tools app frame ``x`` target, ``y`` up, ``z`` right)::

    x_app = v . x_t
    y_app = v . z_t
    z_app = -s * (v . y_t)      s = +1 right-handed, -1 left-handed

Tools' ``delivery_at`` then uses ``path = atan2(z_app, x_app)`` and
``attack = atan2(y_app, hypot(x_app, z_app))``, which equal the UD
definitions ``atan2(-s v.y_t, v.x_t)`` and ``atan2(v.z_t, |v_h|)``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module

import numpy as np

from .target_frame import TargetFrame

IMPACT_MODULE = "shared.python.swing_sim.impact"
_REQUIRED = ("derive_delivery", "DeliveryParameters", "analyze_dplane")
_MAX_ANGLE_DEG = 89.0
Importer = Callable[[str], object]


class ToolsDeliveryUnavailableError(ImportError):
    """Tools delivery physics could not be imported."""


class ToolsDeliveryCompatibilityError(ValueError):
    """Tools delivery façade lacks a required export."""


@dataclass(frozen=True)
class ToolsDeliveryEstimate:
    """Tools values (degrees), a labelled model estimate."""

    club_path_deg: float | None
    attack_angle_deg: float | None
    face_angle_deg: float | None
    dynamic_loft_deg: float
    face_to_path_deg: float | None
    spin_loft_3d_deg: float
    planar_spin_loft_deg: float | None
    spin_axis_tilt_deg: float | None
    dplane_status: str
    label: str = "model estimate (Tools D-plane, spin axis ~ unit(v x n))"


def to_app_frame(vector: object, frame: TargetFrame) -> np.ndarray:
    """Map a UD world vector into the Tools app frame (see module docstring)."""
    x, y, z = frame.components(vector)
    return np.array([x, z, -frame.lateral_sign * y])


class ToolsDeliveryGateway:
    """Thin consumer of the Tools impact façade."""

    def __init__(self, module: object) -> None:
        if module is None:
            raise TypeError("module must not be None")
        for name in _REQUIRED:
            if not callable(getattr(module, name, None)):
                raise ToolsDeliveryCompatibilityError(
                    f"Tools impact façade is missing required export {name}"
                )
        self._module = module

    def estimate(
        self,
        velocity: object,
        face_normal: object,
        frame: TargetFrame,
    ) -> ToolsDeliveryEstimate:
        """Run Tools ``derive_delivery`` / ``analyze_dplane`` on mapped vectors.

        Raises:
            ValueError: for zero speed or angles outside Tools' +/-89 deg domain.
        """
        v_app = to_app_frame(velocity, frame)
        n_app = to_app_frame(face_normal, frame)
        speed = float(np.linalg.norm(v_app))
        if speed <= 1e-6:
            raise ValueError("clubhead speed must be > 0 for Tools delivery")
        n_unit = n_app / float(np.linalg.norm(n_app))
        params = self._module.DeliveryParameters(  # type: ignore[attr-defined]
            clubhead_speed_mps=speed,
            club_path_deg=_clamp(math.degrees(math.atan2(v_app[2], v_app[0]))),
            attack_angle_deg=_clamp(
                math.degrees(math.atan2(v_app[1], math.hypot(v_app[0], v_app[2])))
            ),
            face_angle_deg=_clamp(math.degrees(math.atan2(n_unit[2], n_unit[0]))),
            dynamic_loft_deg=_clamp(
                math.degrees(math.atan2(n_unit[1], math.hypot(n_unit[0], n_unit[2])))
            ),
        )
        derived = self._module.derive_delivery(params)  # type: ignore[attr-defined]
        # Independent D-plane on the exact (unclamped) vectors.
        plane = self._module.analyze_dplane(v_app, n_unit)  # type: ignore[attr-defined]
        return ToolsDeliveryEstimate(
            club_path_deg=plane.club_path_deg,
            attack_angle_deg=plane.attack_angle_deg,
            face_angle_deg=plane.face_angle_deg,
            dynamic_loft_deg=float(plane.dynamic_loft_deg),
            face_to_path_deg=plane.face_to_path_deg,
            spin_loft_3d_deg=float(derived.spin_loft_deg),
            planar_spin_loft_deg=plane.planar_spin_loft_deg,
            spin_axis_tilt_deg=plane.dplane_tilt_deg,
            dplane_status=str(plane.status),
        )


def _clamp(value_deg: float) -> float:
    return max(-_MAX_ANGLE_DEG, min(_MAX_ANGLE_DEG, value_deg))


def load_tools_delivery_gateway(
    importer: Importer = import_module,
) -> ToolsDeliveryGateway:
    """Import and validate the Tools impact façade (fail closed)."""
    if not callable(importer):
        raise TypeError("importer must be callable")
    try:
        module = importer(IMPACT_MODULE)
    except ImportError as exc:
        raise ToolsDeliveryUnavailableError(
            f"Tools delivery physics not available: {exc}"
        ) from exc
    return ToolsDeliveryGateway(module)
