"""Frozen Impact Model Capability & Passivity Matrix (#9550).

This module makes no new physics claims. It restates, as a typed and
test-enforced contract, what each shipping impact model in
``src/shared/python/physics/impact_model`` and the pinned Tools
impact-interval solver (see ``impact_interval_audit.py``) can and cannot
report, so that a caller can never attribute a capability - most importantly
3D tangential spin - to a model that does not compute it.

The ``spring_damper`` model is the deliberate trap this module closes: it
integrates only the face-normal Kelvin-Voigt contact and never advances
tangential spin, so any spin value it returns is the unmodified pre-impact
input, not a prediction. ``qualify_spin_prediction`` refuses to report that
value - including a literal ``0.0`` - as a physical result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

#: Whether the model advances contact over a reported duration.
ALLOWED_DURATION_MODES = frozenset({"instantaneous", "finite_duration"})

#: Whether the model integrates tangential (friction/gear-effect) spin
#: dynamics, or can only ever echo the caller's pre-impact spin.
ALLOWED_SPIN_MODES = frozenset({"supported_3d_tangential", "unavailable_normal_only"})


@dataclass(frozen=True)
class ImpactModelCapability:
    """Declared capabilities of one impact model.

    Preconditions:
        ``model_id`` is non-empty; ``duration_mode`` is a member of
        ``ALLOWED_DURATION_MODES``; ``tangential_spin_mode`` is a member of
        ``ALLOWED_SPIN_MODES``.
    """

    model_id: str
    duration_mode: str
    tangential_spin_mode: str
    supports_off_center_gear_effect: bool
    uses_full_3d_inertia_tensor: bool
    supports_velocity_dependent_cor: bool
    enforces_passivity_bound: bool

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("model_id must be a non-empty string")
        if self.duration_mode not in ALLOWED_DURATION_MODES:
            raise ValueError(
                f"duration_mode must be one of {sorted(ALLOWED_DURATION_MODES)}, "
                f"got {self.duration_mode!r}"
            )
        if self.tangential_spin_mode not in ALLOWED_SPIN_MODES:
            raise ValueError(
                f"tangential_spin_mode must be one of {sorted(ALLOWED_SPIN_MODES)}, "
                f"got {self.tangential_spin_mode!r}"
            )


IMPACT_MODEL_CAPABILITY_MATRIX: Mapping[str, ImpactModelCapability] = {
    "rigid_body": ImpactModelCapability(
        model_id="rigid_body",
        duration_mode="instantaneous",
        tangential_spin_mode="supported_3d_tangential",
        supports_off_center_gear_effect=True,
        uses_full_3d_inertia_tensor=False,
        supports_velocity_dependent_cor=False,
        enforces_passivity_bound=True,
    ),
    "spring_damper": ImpactModelCapability(
        model_id="spring_damper",
        duration_mode="finite_duration",
        tangential_spin_mode="unavailable_normal_only",
        supports_off_center_gear_effect=True,
        uses_full_3d_inertia_tensor=False,
        supports_velocity_dependent_cor=False,
        enforces_passivity_bound=True,
    ),
    "finite_time": ImpactModelCapability(
        model_id="finite_time",
        duration_mode="finite_duration",
        tangential_spin_mode="supported_3d_tangential",
        supports_off_center_gear_effect=True,
        uses_full_3d_inertia_tensor=False,
        supports_velocity_dependent_cor=False,
        enforces_passivity_bound=True,
    ),
    "contact_interval": ImpactModelCapability(
        model_id="contact_interval",
        duration_mode="finite_duration",
        tangential_spin_mode="supported_3d_tangential",
        supports_off_center_gear_effect=True,
        uses_full_3d_inertia_tensor=True,
        supports_velocity_dependent_cor=False,
        enforces_passivity_bound=True,
    ),
}


def get_impact_model_capability(model_id: str) -> ImpactModelCapability:
    """Look up the declared capability record for ``model_id``.

    Preconditions:
        ``model_id`` is a non-empty string present in
        ``IMPACT_MODEL_CAPABILITY_MATRIX``.
    Postconditions:
        Returns the matching :class:`ImpactModelCapability`, never ``None``.
    """
    if not isinstance(model_id, str):
        raise TypeError(f"model_id must be a str, got {type(model_id).__name__}")
    if not model_id:
        raise ValueError("model_id must not be empty")
    try:
        return IMPACT_MODEL_CAPABILITY_MATRIX[model_id]
    except KeyError:
        raise ValueError(
            f"unknown model_id {model_id!r}; known models are "
            f"{sorted(IMPACT_MODEL_CAPABILITY_MATRIX)}"
        ) from None


def qualify_spin_prediction(
    model_id: str, predicted_spin_rpm: float | None
) -> dict[str, object]:
    """Qualify a spin value against the originating model's declared capability.

    Preconditions:
        ``model_id`` resolves via :func:`get_impact_model_capability`. When
        the model supports 3D tangential spin, ``predicted_spin_rpm`` must be
        a finite number.
    Postconditions:
        For a model with ``tangential_spin_mode == "unavailable_normal_only"``,
        always returns ``predicted_spin_rpm=None`` and
        ``claim_qualified=False`` - the input value is discarded unread,
        because that model never advances tangential spin and any value
        attached to its output (including ``0.0``) is the unmodified
        pre-impact input, not a prediction. Otherwise returns the validated
        float spin value with ``claim_qualified=True``.
    """
    capability = get_impact_model_capability(model_id)
    if capability.tangential_spin_mode == "unavailable_normal_only":
        return {
            "spin_status": "unavailable_normal_only",
            "predicted_spin_rpm": None,
            "claim_qualified": False,
        }
    if predicted_spin_rpm is None or not math.isfinite(predicted_spin_rpm):
        raise ValueError(
            "predicted_spin_rpm must be a finite number for a model with "
            "tangential_spin_mode='supported_3d_tangential', got "
            f"{predicted_spin_rpm!r}"
        )
    return {
        "spin_status": "supported_3d_tangential",
        "predicted_spin_rpm": float(predicted_spin_rpm),
        "claim_qualified": True,
    }


def verify_impact_passivity(
    ke_in_j: float,
    ke_out_j: float,
    work_external_j: float = 0.0,
    atol_j: float = 1e-9,
) -> bool:
    """Verify an impact did not spuriously create kinetic energy.

    Preconditions:
        ``ke_in_j`` and ``ke_out_j`` are finite and non-negative;
        ``work_external_j`` is finite; ``atol_j`` is finite and non-negative.
    Postconditions:
        Returns ``True`` iff
        ``ke_out_j <= ke_in_j + work_external_j + atol_j``.
    """
    for name, value in (
        ("ke_in_j", ke_in_j),
        ("ke_out_j", ke_out_j),
        ("work_external_j", work_external_j),
        ("atol_j", atol_j),
    ):
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite number, got {value!r}")
    if ke_in_j < 0.0:
        raise ValueError(f"ke_in_j must be non-negative, got {ke_in_j!r}")
    if ke_out_j < 0.0:
        raise ValueError(f"ke_out_j must be non-negative, got {ke_out_j!r}")
    if atol_j < 0.0:
        raise ValueError(f"atol_j must be non-negative, got {atol_j!r}")
    return ke_out_j <= ke_in_j + work_external_j + atol_j


__all__ = [
    "ALLOWED_DURATION_MODES",
    "ALLOWED_SPIN_MODES",
    "IMPACT_MODEL_CAPABILITY_MATRIX",
    "ImpactModelCapability",
    "get_impact_model_capability",
    "qualify_spin_prediction",
    "verify_impact_passivity",
]
