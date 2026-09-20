"""Pydantic models for ground-support forward dynamics and tracking stages (HO-2 #10156)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .receipt_components import RomFlag

# -----------------------------------------------------------------------------
# Dynamics Stage Models
# -----------------------------------------------------------------------------


class ControllerReceipt(BaseModel):
    """Computed-torque tracking controller parameters."""

    model_config = ConfigDict(extra="ignore")

    type: str = Field(
        ...,
        description="Controller architecture and control mode",
        json_schema_extra={"unit": "text", "stage": "dynamics"},
    )
    omega_rad_s: float = Field(
        ...,
        description="Natural frequency of PD tracking controller",
        json_schema_extra={"unit": "rad/s", "stage": "dynamics"},
    )
    zeta: float = Field(
        ...,
        description="Damping ratio of PD tracking controller",
        json_schema_extra={"unit": "ratio", "stage": "dynamics"},
    )
    balance: list[float] = Field(
        ...,
        description="Root regulation and balance stabilization gains",
        json_schema_extra={"unit": "gains", "stage": "dynamics"},
    )
    tracking_cutoff_hz: float | None = Field(
        None,
        description="Pre-filter cutoff frequency applied to reference commands",
        json_schema_extra={"unit": "Hz", "stage": "dynamics"},
    )


class ContactParametersReceipt(BaseModel):
    """Shared regularized Hunt-Crossley and Coulomb ground contact law."""

    model_config = ConfigDict(extra="ignore")

    stiffness_n_m: float = Field(
        ...,
        description="Ground sole contact normal stiffness",
        json_schema_extra={"unit": "N/m", "stage": "dynamics"},
    )
    dissipation_s_m: float = Field(
        ...,
        description="Hunt-Crossley contact dissipation coefficient",
        json_schema_extra={"unit": "s/m", "stage": "dynamics"},
    )
    static_friction: float = Field(
        ...,
        description="Coulomb static friction coefficient",
        json_schema_extra={"unit": "dimensionless", "stage": "dynamics"},
    )
    dynamic_friction: float = Field(
        ...,
        description="Coulomb dynamic friction coefficient",
        json_schema_extra={"unit": "dimensionless", "stage": "dynamics"},
    )
    viscous_friction: float = Field(
        ...,
        description="Viscous friction velocity damping coefficient",
        json_schema_extra={"unit": "N s/m", "stage": "dynamics"},
    )
    transition_velocity_m_s: float = Field(
        ...,
        description="Transition velocity threshold between static and dynamic friction",
        json_schema_extra={"unit": "m/s", "stage": "dynamics"},
    )


class ReferenceZmpReceipt(BaseModel):
    """Zero-moment point required by the reference kinematics."""

    model_config = ConfigDict(extra="ignore")

    description: str = Field(
        ...,
        description="Explanatory summary of ZMP evaluation meaning",
        json_schema_extra={"unit": "text", "stage": "dynamics"},
    )
    outside_fraction: float = Field(
        ...,
        description="Fraction of trajectory frames where ZMP leaves support polygon",
        json_schema_extra={"unit": "ratio", "stage": "dynamics"},
    )
    outside_fraction_1s_to_1_5s: float = Field(
        ...,
        description="Fraction of downswing frames (1.0-1.5s) where ZMP is outside",
        json_schema_extra={"unit": "ratio", "stage": "dynamics"},
    )
    outside_max_m: float = Field(
        ...,
        description="Maximum distance of ZMP beyond support polygon boundary",
        json_schema_extra={"unit": "m", "stage": "dynamics"},
    )
    unloaded_fraction: float = Field(
        ...,
        description="Fraction of frames where both feet are completely unloaded",
        json_schema_extra={"unit": "ratio", "stage": "dynamics"},
    )
    vertical_grf_over_weight: list[float] = Field(
        ...,
        description="Min and max normalized vertical ground reaction force (BW)",
        json_schema_extra={"unit": "BW", "stage": "dynamics"},
    )
    horizontal_grf_over_weight_max: float = Field(
        ...,
        description="Peak normalized horizontal ground reaction force (BW)",
        json_schema_extra={"unit": "BW", "stage": "dynamics"},
    )


class WeightFractionReceipt(BaseModel):
    """Total vertical contact reaction force normalized by body weight."""

    model_config = ConfigDict(extra="ignore")

    min: float = Field(
        ...,
        description="Minimum weight fraction observed during forward simulation",
        json_schema_extra={"unit": "BW", "stage": "dynamics"},
    )
    max: float = Field(
        ...,
        description="Peak weight fraction observed during forward simulation",
        json_schema_extra={"unit": "BW", "stage": "dynamics"},
    )
    mean: float = Field(
        ...,
        description="Mean weight fraction observed during forward simulation",
        json_schema_extra={"unit": "BW", "stage": "dynamics"},
    )
    by_phase: dict[str, dict[str, float]] | None = Field(
        None,
        description="Phase-specific weight fraction breakdown (address, backswing, downswing, impact, follow-through)",
        json_schema_extra={"unit": "compound", "stage": "dynamics"},
    )


class BackswingReceipt(BaseModel):
    """Dynamic tracking performance up to top of backswing (0 to 1.0 s)."""

    model_config = ConfigDict(extra="ignore")

    root_error_max_m: float = Field(
        ...,
        description="Maximum floating root displacement error during backswing",
        json_schema_extra={"unit": "m", "stage": "dynamics"},
    )
    marker_rms_m: float = Field(
        ...,
        ge=0,
        description="Simulated marker tracking RMS error during backswing",
        json_schema_extra={"unit": "m", "stage": "dynamics"},
    )
    weight_fraction_min: float = Field(
        ...,
        description="Minimum normalized ground support reaction during backswing",
        json_schema_extra={"unit": "BW", "stage": "dynamics"},
    )
    weight_fraction_max: float = Field(
        ...,
        description="Maximum normalized ground support reaction during backswing",
        json_schema_extra={"unit": "BW", "stage": "dynamics"},
    )


class ZmpFilterPass(BaseModel):
    """Iteration pass of cart-table zero-moment-point correction."""

    model_config = ConfigDict(extra="ignore")

    com_shift_max_m: float | None = Field(
        None,
        description="Maximum center-of-mass shift applied",
        json_schema_extra={"unit": "m", "stage": "dynamics"},
    )
    marker_rms_m: float | None = Field(
        None,
        ge=0,
        description="Marker RMS error after shift",
        json_schema_extra={"unit": "m", "stage": "dynamics"},
    )
    closure_error_max_m: float | None = Field(
        None,
        description="Max closure residual after shift",
        json_schema_extra={"unit": "m", "stage": "dynamics"},
    )
    outside_fraction: float | None = Field(
        None,
        description="ZMP outside fraction after pass",
        json_schema_extra={"unit": "ratio", "stage": "dynamics"},
    )
    outside_fraction_1s_to_1_5s: float | None = Field(
        None,
        description="ZMP outside fraction in downswing after pass",
        json_schema_extra={"unit": "ratio", "stage": "dynamics"},
    )
    outside_max_m: float | None = Field(
        None,
        description="Max ZMP excursion after pass",
        json_schema_extra={"unit": "m", "stage": "dynamics"},
    )


class ZmpFilterReport(BaseModel):
    """Optional cart-table dynamics filter evaluation (MM-7b)."""

    model_config = ConfigDict(extra="ignore")

    margin_m: float | None = Field(
        None,
        description="Support polygon boundary safety margin",
        json_schema_extra={"unit": "m", "stage": "dynamics"},
    )
    com_weight: float | None = Field(
        None,
        description="Weight of CoM row in IK optimization",
        json_schema_extra={"unit": "weight", "stage": "dynamics"},
    )
    passes: list[Any] | None = Field(
        None,
        description="Iteration passes of CoM modification",
        json_schema_extra={"unit": "list", "stage": "dynamics"},
    )
    before: dict[str, Any] | None = Field(
        None,
        description="ZMP summary metrics prior to filter passes",
        json_schema_extra={"unit": "compound", "stage": "dynamics"},
    )


class ShootingFitReport(BaseModel):
    """Optional contact-aware shooting fit evaluation (FB-5)."""

    model_config = ConfigDict(extra="ignore")

    best_iteration: int | None = Field(
        None,
        description="Index of iteration that achieved best replay error",
        json_schema_extra={"unit": "index", "stage": "dynamics"},
    )
    iterations: list[dict[str, Any]] | int | None = Field(
        None,
        description="Iteration passes or total count of shooting fit",
        json_schema_extra={"unit": "list", "stage": "dynamics"},
    )
    relaxation: float | None = Field(
        None,
        description="Iterative learning relaxation update gain",
        json_schema_extra={"unit": "gain", "stage": "dynamics"},
    )
    locked: list[str] | None = Field(
        None,
        description="Pelvis coordinates pinned during re-solve",
        json_schema_extra={"unit": "names", "stage": "dynamics"},
    )


class DynamicsReceipt(BaseModel):
    """Forward dynamics tracking simulation results and diagnostics."""

    model_config = ConfigDict(extra="ignore")

    duration_s: float = Field(
        ...,
        description="Total duration of simulated trajectory",
        json_schema_extra={"unit": "s", "stage": "dynamics"},
    )
    dt_s: float = Field(
        ...,
        description="Integration numerical time step",
        json_schema_extra={"unit": "s", "stage": "dynamics"},
    )
    controller: ControllerReceipt = Field(
        ...,
        description="Computed torque controller settings",
        json_schema_extra={"unit": "compound", "stage": "dynamics"},
    )
    contact_parameters: ContactParametersReceipt | None = Field(
        None,
        description="Ground contact stiffness, dissipation, and friction coefficients",
        json_schema_extra={"unit": "compound", "stage": "dynamics"},
    )
    reference_zmp: ReferenceZmpReceipt | None = Field(
        None,
        description="Zero-moment point metrics demanded by reference trajectory",
        json_schema_extra={"unit": "compound", "stage": "dynamics"},
    )
    marker_rms_m: float = Field(
        ...,
        ge=0,
        description="Simulated full-body motion marker tracking RMS error",
        json_schema_extra={"unit": "m", "stage": "dynamics"},
    )
    segment_rms_m: dict[str, float] = Field(
        ...,
        description="Per-segment simulated marker tracking RMS error",
        json_schema_extra={"unit": "m", "stage": "dynamics"},
    )
    joint_tracking_rms_rad: float = Field(
        ...,
        description="RMS joint coordinate tracking error between sim and reference",
        json_schema_extra={"unit": "rad", "stage": "dynamics"},
    )
    root_tracking_rms_m: float = Field(
        ...,
        description="RMS floating base position tracking error between sim and ref",
        json_schema_extra={"unit": "m", "stage": "dynamics"},
    )
    weight_fraction: WeightFractionReceipt = Field(
        ...,
        description="Normalized vertical ground support reaction stats",
        json_schema_extra={"unit": "BW", "stage": "dynamics"},
    )
    inside_support_polygon_fraction: float = Field(
        ...,
        description="Fraction of simulated frames where CoM remains inside support",
        json_schema_extra={"unit": "ratio", "stage": "dynamics"},
    )
    range_of_motion_flags: dict[str, RomFlag] | None = Field(
        None,
        description="Range-of-motion excursions observed during simulation",
        json_schema_extra={"unit": "compound", "stage": "dynamics"},
    )
    root_error_timeline_m: dict[str, float] = Field(
        ...,
        description="Floating pelvis tracking position error sampled across timeline",
        json_schema_extra={"unit": "m", "stage": "dynamics"},
    )
    backswing_to_1s: BackswingReceipt | None = Field(
        None,
        description="Dynamic tracking metrics restricted to backswing (0 to 1.0 s)",
        json_schema_extra={"unit": "compound", "stage": "dynamics"},
    )
    peak_joint_torque_n_m: float = Field(
        ...,
        description="Maximum absolute joint actuator torque exerted in simulation",
        json_schema_extra={"unit": "N m", "stage": "dynamics"},
    )
    lowest_sphere_height_min_m: float = Field(
        ...,
        description="Minimum elevation of lowest foot contact sphere in simulation",
        json_schema_extra={"unit": "m", "stage": "dynamics"},
    )
    lowest_sphere_height_max_m: float = Field(
        ...,
        description="Maximum elevation of lowest foot contact sphere in simulation",
        json_schema_extra={"unit": "m", "stage": "dynamics"},
    )
    zmp_filter: ZmpFilterReport | None = Field(
        None,
        description="Optional cart-table zero-moment-point filter report",
        json_schema_extra={"unit": "compound", "stage": "dynamics"},
    )
    shooting_fit: ShootingFitReport | None = Field(
        None,
        description="Optional contact-aware shooting fit report",
        json_schema_extra={"unit": "compound", "stage": "dynamics"},
    )
    mjx: dict[str, Any] | None = Field(
        None,
        description="Optional MJX differentiable trajectory optimization report",
        json_schema_extra={"unit": "compound", "stage": "dynamics"},
    )
