"""Tools ball-flight bridge for the impact outcome map (NSX-83).

Ball launch and spin quantities are produced by the Tools delivery physics
through :class:`ToolsDeliveryGateway`, treated as a **black box**: sensitivities
are central finite differences of ``gateway.estimate(v, n, frame)`` over the
club state, then projected onto the event section with the same ``Pi`` as the
kinematic outcomes.  This module owns no delivery physics.  When Tools cannot
be imported (``ToolsDeliveryUnavailableError``) or the face is unobserved, every
quantity is ``UNAVAILABLE`` with the reason; nothing is invented.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.shared.python.impact_parameters import TargetFrame
from src.shared.python.impact_parameters.tools_gateway import (
    ToolsDeliveryGateway,
    ToolsDeliveryUnavailableError,
    load_tools_delivery_gateway,
)

from .evidence import (
    AllocationDependence,
    InputProvenance,
    ModelProvenance,
    QuantityRecord,
)
from .impact_outcome_map import (
    DEFAULT_STEP,
    STATE_DIM,
    ImpactOutcomeMap,
    RigidClubState,
    central_difference_jacobian,
)

Array = NDArray[np.float64]

TOOLS_FIELDS = (
    "face_to_path_deg",
    "spin_loft_3d_deg",
    "planar_spin_loft_deg",
    "spin_axis_tilt_deg",
)
_WRAPPED = ("face_to_path_deg",)
MODEL = ModelProvenance("impact_ball_flight_map", "1")


@dataclass(frozen=True)
class BallFlightMap:
    """Tools-derived quantities with black-box FD Jacobians.

    ``jacobian`` is event-section; ``jacobian_fixed_time`` is a diagnostic.
    """

    names: tuple[str, ...]
    values: Array
    jacobian: Array
    jacobian_fixed_time: Array
    records: dict[str, QuantityRecord]
    unavailable: dict[str, str]
    available: bool
    unavailable_reason: str
    label: str


def _record(name: str, value: float, reason: str | None, label: str) -> QuantityRecord:
    return QuantityRecord(
        values=np.array([np.nan if reason else value]),
        units="deg",
        allocation_dependence=(
            AllocationDependence.UNAVAILABLE
            if reason
            else AllocationDependence.INVARIANT
        ),
        model_provenance=MODEL,
        input_provenance=InputProvenance("rigid_club_state", "tools_gateway:" + name),
        conventions=(label,),
        reason=reason,
    )


def _all_unavailable(reason: str, n_dim: int) -> BallFlightMap:
    names = TOOLS_FIELDS
    nan_rows = np.full((len(names), n_dim), np.nan)
    return BallFlightMap(
        names=names,
        values=np.full(len(names), np.nan),
        jacobian=nan_rows,
        jacobian_fixed_time=nan_rows.copy(),
        records={n: _record(n, np.nan, reason, "Tools unavailable") for n in names},
        unavailable=dict.fromkeys(names, reason),
        available=False,
        unavailable_reason=reason,
        label="Tools unavailable",
    )


def _tools_vector(
    gateway: ToolsDeliveryGateway, state: RigidClubState, frame: TargetFrame
) -> tuple[Array, str]:
    series = state.series()
    if series.face_normal is None:
        raise ValueError("Tools ball-flight bridge requires an observed face normal")
    est = gateway.estimate(series.velocity_mps[1], series.face_normal[1], frame)
    vals = np.array(
        [np.nan if getattr(est, n) is None else getattr(est, n) for n in TOOLS_FIELDS]
    )
    return vals, est.label


def tools_ball_flight_map(
    state: RigidClubState,
    omap: ImpactOutcomeMap,
    frame: TargetFrame,
    *,
    gateway_loader: Callable[[], ToolsDeliveryGateway] = load_tools_delivery_gateway,
    step: float = DEFAULT_STEP,
) -> BallFlightMap:
    """Ball-flight quantities and event-section FD Jacobians via Tools.

    Postconditions: never raises for a missing Tools install or an unobserved
    face (returns an ``available=False`` map); ``jacobian = J_fd @ omap.projection``.

    Raises:
        ValueError: On a non-positive ``step`` or a Tools domain error.
    """
    if not step > 0.0:
        raise ValueError(f"step must be positive, got {step}")
    if state.face_unobservable_reason is not None:
        return _all_unavailable(
            f"face unobservable: {state.face_unobservable_reason}", STATE_DIM
        )
    try:
        gateway = gateway_loader()
    except ToolsDeliveryUnavailableError as exc:
        return _all_unavailable(f"Tools unavailable (fail closed): {exc}", STATE_DIM)
    base, label = _tools_vector(gateway, state, frame)
    wrap = np.array([n in _WRAPPED for n in TOOLS_FIELDS])
    fixed = central_difference_jacobian(
        lambda dx: _tools_vector(gateway, state.retract(dx), frame)[0],
        STATE_DIM,
        step,
        wrap,
    )
    unavailable = {
        n: "Tools returned no value for this quantity"
        for n, v in zip(TOOLS_FIELDS, base, strict=True)
        if not np.isfinite(v)
    }
    return BallFlightMap(
        names=TOOLS_FIELDS,
        values=base,
        jacobian=fixed @ omap.projection,
        jacobian_fixed_time=fixed,
        records={
            n: _record(n, v, unavailable.get(n), label)
            for n, v in zip(TOOLS_FIELDS, base, strict=True)
        },
        unavailable=unavailable,
        available=True,
        unavailable_reason="",
        label=label,
    )
