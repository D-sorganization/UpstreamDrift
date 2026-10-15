"""NSX-72 (#12590): rows-only grip models on the anthro record, against NSX-26.

On the welded anthro trajectory a re-derived rows-only closure (shaft-twist
revolute, trail ball) admits the trajectory with its own velocity and
acceleration residuals and gives exactly the NSX-26 restriction numbers; what
changes is the dimension of the motion it admits.  The default run uses every
tenth benchmark frame; ``LOOP_NULLSPACE_RESTRICTION_STRIDE=1`` runs all 200 and
``LOOP_NULLSPACE_ALTERNATIVES_JSON`` writes the measured record.  Nothing here
is evidence about the true grip mechanics.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("highspy")

from src.shared.python.loop_nullspace.closure_alternatives import (  # noqa: E402
    GRIP_MODEL_SPECS,
    GripModel,
)
from src.shared.python.loop_nullspace.closure_alternatives_report import (  # noqa: E402
    GRIP_MODEL_LABEL,
    median_width_ratio,
)
from tests.integration.loop_nullspace_closure_alternatives_support import (  # noqa: E402
    ANTHRO_MODELS,
    ANTHRO_REFUSED,
    anthro_results,
)
from tests.integration.loop_nullspace_lower_body_support import (  # noqa: E402
    AnthroBench,
    build_frames,
)
from tests.integration.loop_nullspace_restriction_support import (  # noqa: E402
    restriction_report,
    strided,
)

pytestmark = [pytest.mark.integration, pytest.mark.requires_pinocchio]
W, S, TB, LB = (
    GripModel.WELD,
    GripModel.SHAFT_TWIST_REVOLUTE,
    GripModel.TRAIL_BALL,
    GripModel.LEAD_BALL,
)


@pytest.fixture(scope="module")
def record() -> tuple:
    stride = int(os.environ.get("LOOP_NULLSPACE_RESTRICTION_STRIDE", "10"))
    bench = AnthroBench()
    indices = strided(stride)
    frames, refused = build_frames(bench, indices)
    results = anthro_results(bench, frames)
    nsx26, _, _ = restriction_report(bench, indices)
    return results, nsx26, frames, refused, stride


def test_every_rows_only_model_admits_the_welded_trajectory(record) -> None:
    results = record[0]
    for model in ANTHRO_MODELS:
        adm = results[model].admission
        assert adm.admitted, (model, adm.reasons)
        assert adm.max_phi_dot < 1e-6 and adm.max_phi_ddot < 1e-6
        assert np.isnan(adm.max_phi)  # no closure plant in the record: skipped


def test_admitted_motion_dimension_grows_with_the_dropped_rows(record) -> None:
    ks = {m: record[0][m].kinematic_nullity for m in ANTHRO_MODELS}
    assert len({lo for lo, _ in ks.values()}) == 3
    assert ks[S][0] - ks[W][0] == 1 and ks[S][1] - ks[W][1] == 1
    assert ks[TB][0] - ks[W][0] == 3 and ks[TB][1] - ks[W][1] == 3


def test_numbers_equal_the_nsx26_restriction_on_the_same_frames(record) -> None:
    results, nsx26 = record[0], record[1]
    for model in ANTHRO_MODELS:
        kind = GRIP_MODEL_SPECS[model].closure_kind
        own = results[model]
        assert own.counts() == {
            "feasible": nsx26.feasible_counts()[kind],
            "infeasible": nsx26.infeasible_counts()[kind],
            "balance_inconsistent": nsx26.balance_inconsistent_counts()[kind],
        }
        assert np.allclose(
            own.widths("tau_"),
            nsx26.series(kind).width()[:, : own.widths("tau_").shape[1]],
            equal_nan=True,
            rtol=1e-9,
            atol=1e-9,
        )


def test_lead_ball_is_a_stated_limitation_on_the_anthro_model(record) -> None:
    assert "topology" in ANTHRO_REFUSED[LB]
    assert GRIP_MODEL_SPECS[LB].extra_coordinates == 3
    assert LB not in record[0]


def test_record_is_labelled_and_written_on_request(record) -> None:
    results, _, frames, refused, stride = record
    out = {
        "label": GRIP_MODEL_LABEL,
        "stride": stride,
        "frames_used": [f.index for f in frames],
        "refused_frames": {str(k): v for k, v in refused.items()},
        "models": {m.value: results[m].to_dict() for m in ANTHRO_MODELS},
        "limitations": {m.value: why for m, why in ANTHRO_REFUSED.items()},
        "median_torque_width_ratio_vs_weld": {
            m.value: median_width_ratio(results[W], results[m], "tau_") for m in (S, TB)
        },
    }
    text = json.dumps(out, indent=2, default=str)
    target = os.environ.get("LOOP_NULLSPACE_ALTERNATIVES_JSON")
    if target:
        Path(target).write_text(text)
    assert json.loads(text)["label"] == GRIP_MODEL_LABEL
