"""Event-location adapter for existing distributed-grip trajectories."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from scripts.research.proximal_distal_energy.articulated_contact_events import (
    ContactEventRecord,
    locate_contact_events,
)
from scripts.research.proximal_distal_energy.articulated_distributed_forward import (
    DistributedIntegrationCase,
)
from scripts.research.proximal_distal_energy.articulated_distributed_grip import (
    distributed_reference_lengths,
    distributed_signed_gaps,
)
from scripts.research.proximal_distal_energy.spatial_full_body import SpatialModel


def locate_distributed_trace_events(
    *,
    model: SpatialModel,
    case: DistributedIntegrationCase,
    trace: Mapping[str, Any],
    gap_tolerance_m: float = 1.0e-10,
    time_tolerance_s: float = 1.0e-12,
) -> tuple[ContactEventRecord, ...]:
    """Locate opening/reattachment roots on a retained distributed trace."""

    if not isinstance(model, SpatialModel):
        raise TypeError("model must be a SpatialModel")
    if not isinstance(case, DistributedIntegrationCase):
        raise TypeError("case must be a DistributedIntegrationCase")
    required = (
        "time_s",
        "q",
        "qd",
        "station_signed_gap_m",
        "station_active",
    )
    missing = tuple(name for name in required if name not in trace)
    if missing:
        raise ValueError(f"trace is missing event fields: {', '.join(missing)}")
    reference = distributed_reference_lengths(
        model,
        case.q,
        grip_span_m=case.grip_span_m,
        hand_contact_local_x_m=case.hand_contact_local_x_m,
        config=case.grip,
    )

    def evaluate_gap(position: np.ndarray) -> np.ndarray:
        return distributed_signed_gaps(
            model,
            position,
            grip_span_m=case.grip_span_m,
            hand_contact_local_x_m=case.hand_contact_local_x_m,
            reference_lengths_m=reference,
            config=case.grip,
        )

    return locate_contact_events(
        time_s=np.asarray(trace["time_s"], dtype=np.float64),
        positions=np.asarray(trace["q"], dtype=np.float64),
        velocities=np.asarray(trace["qd"], dtype=np.float64),
        station_signed_gap_m=np.asarray(
            trace["station_signed_gap_m"], dtype=np.float64
        ),
        station_active=np.asarray(trace["station_active"], dtype=bool),
        gap_evaluator=evaluate_gap,
        gap_tolerance_m=gap_tolerance_m,
        time_tolerance_s=time_tolerance_s,
    )


__all__ = ["locate_distributed_trace_events"]
