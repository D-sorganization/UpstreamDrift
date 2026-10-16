"""Anthro support for the NSX-72 closure-alternative comparison (#12590).

The anthro record is a full-body model whose spec carries exactly one hand-club
closure with weld semantics.  The rows-only grip models (shaft-twist revolute,
trail ball) are therefore re-derived as the same closure with fewer rows, which
the model's own velocity and acceleration residuals admit; the lead-hand ball
adds three coordinates and a topology the spec builder does not support, so it
is a stated limitation here and exists only on the synthetic spatial fixture.
The position level is not evaluated (no closure plant in the record, recorded
as ``require_position=False``).  Capacity prior, friction cones and hand
geometry are those of the NSX-26 support, identical for every model.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.shared.python.loop_nullspace.admission import AdmissionGates
from src.shared.python.loop_nullspace.closure_alternatives import (
    GRIP_MODEL_SPECS,
    GripModel,
    admit_model,
)
from src.shared.python.loop_nullspace.closure_alternatives_report import (
    GripModelResult,
    evaluate_grip_model,
)
from src.shared.python.loop_nullspace.restriction import restrict
from tests.integration.loop_nullspace_lower_body_support import AnthroBench, Frame
from tests.integration.loop_nullspace_restriction_support import (
    feasible_set_of,
    report_quantity_builders,
)

ANTHRO_MODELS = (
    GripModel.WELD,
    GripModel.SHAFT_TWIST_REVOLUTE,
    GripModel.TRAIL_BALL,
)
ANTHRO_REFUSED = {
    GripModel.LEAD_BALL: (
        "adds three coordinates and a ball-joint topology the full-body spec "
        "builder does not support (one weld closure per spec)"
    ),
}


def anthro_results(
    bench: AnthroBench, frames: Sequence[Frame]
) -> dict[GripModel, GripModelResult]:
    """Admission, dynamics and envelopes of each rows-only model on ``frames``."""
    times = np.asarray([bench.record["time_s"][f.index] for f in frames])
    gates = AdmissionGates(require_position=False)
    out = {}
    for model in ANTHRO_MODELS:
        kind = GRIP_MODEL_SPECS[model].closure_kind
        snaps = [restrict(f.snapshot, kind) for f in frames]
        adm = admit_model(model, snaps, times, gates)
        out[model] = evaluate_grip_model(
            adm,
            quantities=report_quantity_builders(bench),
            feasible_set_of=feasible_set_of(bench),
        )
    return out
