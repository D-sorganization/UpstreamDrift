"""Fail-closed gateway to the pinned Tools force-attribution contract."""

from __future__ import annotations

import numpy as np

from shared.python.swing_sim import (
    ATTRIBUTION_SCHEMA_VERSION,
    DoublePendulumAttributionProvider,
    PendulumParameters,
    TrajectoryAttribution,
    attribute_trajectory,
)

from src.shared.python.simulation_backends import GolfModelParams

REQUIRED_FORCE_ATTRIBUTION_SCHEMA = "force-attribution/v1"


def _require_schema() -> None:
    if ATTRIBUTION_SCHEMA_VERSION != REQUIRED_FORCE_ATTRIBUTION_SCHEMA:
        raise RuntimeError(
            "Tools force-attribution schema mismatch: expected "
            f"{REQUIRED_FORCE_ATTRIBUTION_SCHEMA}, got {ATTRIBUTION_SCHEMA_VERSION}"
        )


def _tools_parameters(params: GolfModelParams) -> PendulumParameters:
    rendered = params.to_double_pendulum_parameters()
    upper = rendered.upper_segment
    lower = rendered.lower_segment
    return PendulumParameters(
        m1=upper.mass_kg,
        l1=upper.length_m,
        lc1=upper.center_of_mass_distance,
        i1=upper.inertia_about_proximal_joint,
        m2=lower.total_mass,
        l2=lower.length_m,
        lc2=lower.center_of_mass_distance,
        i2=lower.inertia_about_proximal_joint,
        d1=rendered.damping_shoulder,
        d2=rendered.damping_wrist,
    )


def attribute_double_pendulum_trajectory(
    params: GolfModelParams,
    time_s: np.ndarray,
    q: np.ndarray,
    velocity: np.ndarray,
    controls_nm: np.ndarray,
) -> TrajectoryAttribution:
    """Attribute an achieved Upstream trajectory through pinned Tools physics.

    The model uses shoulder-absolute/wrist-relative coordinates and the wrist
    joint as the declared hand-path endpoint. The force-only endpoint mapping
    retains any joint-couple residual instead of relabeling it as hand force.
    """
    if not isinstance(params, GolfModelParams):
        raise TypeError("params must be a GolfModelParams instance")
    _require_schema()
    provider = DoublePendulumAttributionProvider(
        _tools_parameters(params),
        g_inplane=(0.0, -params.projected_gravity),
    )
    return attribute_trajectory(provider, time_s, q, velocity, controls_nm)


__all__ = [
    "REQUIRED_FORCE_ATTRIBUTION_SCHEMA",
    "attribute_double_pendulum_trajectory",
]
