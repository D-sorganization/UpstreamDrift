"""NSX-26 (#12569): reaction-restriction sensitivity on the anthro record.

The welded trajectory is fixed; only the hand-club reaction restriction changes
(WELD6 -> SHAFT_FREE5 -> BALL3).  The default run uses every tenth benchmark
frame; set ``LOOP_NULLSPACE_RESTRICTION_STRIDE=1`` for all 200 and
``LOOP_NULLSPACE_RESTRICTION_JSON`` to write the measured record.  Nothing here
is evidence about the true grip mechanics, and the ASSUMED capacity prior is
never relaxed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("highspy")

from src.shared.python.loop_nullspace.contracts import ClosureKind  # noqa: E402
from tests.integration.loop_nullspace_lower_body_support import (  # noqa: E402
    AnthroBench,
)
from tests.integration.loop_nullspace_restriction_support import (  # noqa: E402
    restriction_report,
    strided,
    summarise,
)

pytestmark = [pytest.mark.integration, pytest.mark.requires_pinocchio]
W6, S5, B3 = ClosureKind.WELD6, ClosureKind.SHAFT_FREE5, ClosureKind.BALL3
AIRBORNE_FRAME = 510


@pytest.fixture(scope="module")
def record() -> tuple:
    stride = int(os.environ.get("LOOP_NULLSPACE_RESTRICTION_STRIDE", "10"))
    bench = AnthroBench()
    report, frames, refused = restriction_report(bench, strided(stride))
    return report, frames, refused, stride


def test_report_is_labelled_and_accounts_for_every_frame(record) -> None:
    report, frames, refused, stride = record
    assert report.label == "REACTION_RESTRICTION_SENSITIVITY"
    assert "not evidence about the true grip mechanics" in report.non_claims
    assert report.n_frames == len(frames)
    assert set(refused) <= {AIRBORNE_FRAME}
    for kind in report.kinds:
        total = (
            report.feasible_counts()[kind]
            + report.infeasible_counts()[kind]
            + report.balance_inconsistent_counts()[kind]
        )
        assert total == len(frames)


def test_reference_infeasibility_names_the_capacity_prior_only(record) -> None:
    report = record[0]
    assert set(report.conflict_hypotheses(W6)) <= {"H-TORQUE-CAPACITY"}
    assert report.infeasible_counts()[W6] > 0  # the NSX-60 finding is present


def test_restricted_envelopes_nest_inside_the_weld_envelope(record) -> None:
    report = record[0]
    assert report.nesting_violations() == ()
    feasible = report.feasible_counts()
    assert feasible[W6] >= feasible[S5] >= feasible[B3]
    for kind in (S5, B3):
        assert set(report.lost_frames(kind)).isdisjoint(
            f.index for f in report.outcomes[W6].frames if f.outcome.name != "FEASIBLE"
        )


def test_conflicts_name_declared_hypotheses_only(record) -> None:
    report = record[0]
    for kind in report.kinds:
        assert set(report.conflict_hypotheses(kind)) <= {
            "H-TORQUE-CAPACITY",
            "H-FRICTION-CONE",
        }


def test_summary_is_json_serialisable_and_written_on_request(record) -> None:
    report, frames, refused, stride = record
    summary = summarise(report)
    summary["stride"] = stride
    summary["frames_used"] = [f.index for f in frames]
    summary["refused_frames"] = {str(k): v for k, v in refused.items()}
    text = json.dumps(summary, indent=2, default=str)
    target = os.environ.get("LOOP_NULLSPACE_RESTRICTION_JSON")
    if target:
        Path(target).write_text(text)
    assert json.loads(text)["label"] == "REACTION_RESTRICTION_SENSITIVITY"
