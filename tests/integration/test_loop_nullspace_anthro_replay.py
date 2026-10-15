"""NSX-31 (#12571) acceptance: open-loop versus feedback-assisted anthro replay.

Allocates the longest run of usable strided frames of the ground-supported
anthro record (see ``loop_nullspace_replay_support`` for the window rule) and
replays the allocated torques plus the allocated foot-ground loads through the
Pinocchio constrained provider, from the first frame's measured state only.

The tests pin what must hold whatever the outcome (labels, separation of the
feedback budget, the load map reproducing the measured accelerations, an
integrator-converged open-loop divergence) and record the numbers; they do not
assert that a replay succeeds.  Set ``LOOP_NULLSPACE_REPLAY_JSON=<path>`` to
write the measured record.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("highspy")
pytest.importorskip("clarabel")

from tests.integration.loop_nullspace_lower_body_support import (  # noqa: E402
    BENCHMARK_FRAMES,
    AnthroBench,
    build_frames,
    host_record,
)
from tests.integration.loop_nullspace_replay_support import (  # noqa: E402
    FEEDBACK_SWEEP,
    PROBE_TIMES_S,
    acceleration_consistency,
    build_case,
    run_feedback,
    run_open_loop,
    summarize,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.requires_pinocchio,
]


@pytest.fixture(scope="module")
def case():
    bench = AnthroBench()
    frames, _ = build_frames(bench, BENCHMARK_FRAMES)
    sets = [bench.feasible_set(f, None) for f in frames]
    return build_case(bench, frames, sets)


@pytest.fixture(scope="module")
def open_loop(case):
    return run_open_loop(case, substeps=3, interpolation="linear")


def test_window_is_the_longest_run_of_usable_frames(case) -> None:
    lengths = [n for _, _, n in case.runs_summary]
    assert len(case.window) == max(lengths) >= 40
    assert case.n_usable == 124  # 75 of the 199 decomposable frames are infeasible
    assert all(
        b.index - a.index == 3
        for a, b in zip(case.window, case.window[1:], strict=False)
    )


def test_allocation_of_the_window_is_optimal(case) -> None:
    assert case.allocation.status.name == "OPTIMAL"
    assert case.allocation.kkt_residuals.max() < 1e-6


def test_allocated_loads_reproduce_the_measured_accelerations(case) -> None:
    """The load map is right, so any divergence below is dynamics, not plumbing."""
    consistency = acceleration_consistency(case)
    assert consistency["max_relative_to_peak_a"] < 1e-6


def test_open_loop_replay_is_labelled_and_independent(case, open_loop) -> None:
    assert open_loop.mode.value == "open_loop"
    assert open_loop.evidence_label == "independent_open_loop_replay"
    assert open_loop.state_resets == 0
    assert open_loop.feedback_torque is None and open_loop.feedback_budget is None
    np.testing.assert_array_equal(open_loop.q[0], case.q_ref[0])
    assert not open_loop.may_present_as_validation


def test_open_loop_divergence_is_integrator_converged(case, open_loop) -> None:
    """Halving the RK4 step leaves the early divergence unchanged."""
    fine = run_open_loop(case, substeps=6, interpolation="linear")
    a, b = summarize(case, open_loop), summarize(case, fine)
    assert a["time_to_divergence_s"] is not None
    assert 0.005 < a["time_to_divergence_s"] < 0.1
    assert abs(a["time_to_divergence_s"] - b["time_to_divergence_s"]) <= 0.003
    for key in ("10ms", "30ms"):
        assert b["state_error_at"][key] == pytest.approx(
            a["state_error_at"][key], rel=0.02
        )


def test_feedback_replay_reports_its_budget_separately(case, open_loop) -> None:
    omega, substeps = FEEDBACK_SWEEP[2]
    fb = run_feedback(case, omega, substeps)
    assert fb.mode.value == "feedback_assisted"
    assert fb.evidence_label == "feedback_assisted_tracking"
    assert not fb.may_present_as_validation
    np.testing.assert_allclose(
        fb.applied_torque, fb.allocated_torque + fb.feedback_torque, atol=1e-9
    )
    # the allocated series is the open-loop one: feedback never rewrites it
    assert fb.feedback_budget.energy_ratio > 0.0
    assert np.max(np.abs(fb.q - fb.reference_q)) < np.max(
        np.abs(open_loop.q - open_loop.reference_q)
    )


def test_replay_record_is_written_when_requested(case, open_loop) -> None:
    """Runs the feedback sweep and writes the measured record (opt in by env)."""
    path = os.environ.get("LOOP_NULLSPACE_REPLAY_JSON")
    if not path:
        pytest.skip("set LOOP_NULLSPACE_REPLAY_JSON to record the replay report")
    sweep = {}
    for omega, substeps in FEEDBACK_SWEEP:
        sweep[f"omega_{omega:g}"] = summarize(case, run_feedback(case, omega, substeps))
    interpolation = {
        kind: summarize(case, run_open_loop(case, substeps=3, interpolation=kind))
        for kind in ("zoh", "linear", "cubic")
    }
    record = {
        "host": host_record(),
        "window": {
            "first_frame": case.window[0].index,
            "last_frame": case.window[-1].index,
            "frames": len(case.window),
            "start_time_s": float(case.times[0]),
            "end_time_s": float(case.times[-1]),
            "runs": case.runs_summary,
            "usable_frames": case.n_usable,
        },
        "allocation": {
            "status": case.allocation.status.name,
            "solve_time_s": case.allocation.solve_time_s,
            "kkt": str(case.allocation.kkt_residuals),
            "hypotheses": list(case.allocation.record.hypotheses),
        },
        "acceleration_consistency": acceleration_consistency(case),
        "probe_times_s": PROBE_TIMES_S,
        "open_loop": summarize(case, open_loop),
        "open_loop_by_interpolation": interpolation,
        "open_loop_fine_step": summarize(
            case, run_open_loop(case, substeps=6, interpolation="linear")
        ),
        "feedback_sweep": sweep,
    }
    Path(path).write_text(json.dumps(record, indent=2, default=str))
