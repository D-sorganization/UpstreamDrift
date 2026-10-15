"""NSX-71 (#12589): pre-registered held-out evaluation on the anthro record.

Executes the protocol fixed in
``docs/research/loop_nullspace/inverse_optimal_control_report.md`` (section
"Pre-Registration").  The gates are *not* asserted as passing: the test checks
that the protocol runs, that its constants equal the registered values, and
that the verdict logic is the registered one.  The measured record is written
to the file named by ``LOOP_NULLSPACE_INVERSE_JSON``.

Limitation, stated in the report: one swing record, no measured torques.  The
held-out windows are other time windows of that record, not other swings.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("highspy")
pytest.importorskip("clarabel")

from src.shared.python.loop_nullspace.allocation import (  # noqa: E402
    GrossPowerSpec,
    NotStrictlyConvexError,
    torque_scale_from_capacity,
)
from src.shared.python.loop_nullspace.inverse_allocation import (  # noqa: E402
    InverseProblem,
    block_bootstrap_log_ratio,
    fit_allocation_weights,
    relative_torque_error,
    replay_allocation,
    stationarity,
)
from src.shared.python.loop_nullspace.solvers import SolveStatus  # noqa: E402
from tests.integration.loop_nullspace_lower_body_support import (  # noqa: E402
    BENCHMARK_FRAMES,
    AnthroBench,
    build_frames,
    contiguous_runs,
    host_record,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.requires_pinocchio,
]

logger = logging.getLogger(__name__)
REPORT = (
    Path(__file__).resolve().parents[2]
    / "docs/research/loop_nullspace/inverse_optimal_control_report.md"
)

# ---- pre-registered constants (report, "Gates and Success Threshold") -------
FIT_FRAMES = 41
BUFFER_FRAMES = 3
RHO_KKT_MAX = 0.10
SV_RATIO_MIN = 1e-2
CI_WIDTH_MAX = 1.0
BOOT_BLOCK = 8
BOOT_RESAMPLES = 200
BOOT_SEED = 12589
GAIN_MAX = 0.9
ERROR_MAX = 0.25
DEFAULT_WEIGHTS = {"effort": 1.0, "rate": 0.1}
EFFORT_ONLY = {"effort": 1.0, "rate": 0.0}


@pytest.fixture(scope="module")
def bench() -> AnthroBench:
    return AnthroBench()


@pytest.fixture(scope="module")
def runs(bench):
    frames, _refused = build_frames(bench, BENCHMARK_FRAMES)
    sets = [bench.feasible_set(f, None) for f in frames]
    flags = [not fs.is_empty() for fs in sets]
    by_index = {f.index: fs for f, fs in zip(frames, sets, strict=True)}
    found = contiguous_runs(frames, flags)
    return [([f.index for f in r], [by_index[f.index] for f in r]) for r in found]


def _problem(bench, indices, sets, *, gross: bool = False) -> InverseProblem:
    topo_idx = [bench.order.index(n) for n in bench.actuated]
    rec = bench.record
    tau = np.array([rec["tau"][i][topo_idx] for i in indices])
    gp = None
    if gross:
        vel = np.array([rec["v"][i][topo_idx] for i in indices])
        ones = np.ones(len(topo_idx))
        gp = GrossPowerSpec(1.0, ones, ones, vel)
    return InverseProblem(
        feasible_sets=tuple(sets),
        times_s=np.array([rec["time_s"][i] for i in indices]),
        torque_scale=torque_scale_from_capacity(bench.capacity),
        observed_torque=tau,
        gross_power=gp,
    )


def _held_out_error(problem, weights, target) -> dict:
    """Relative error of the replayed allocation against ``target`` (fail closed).

    A replay that is not ``OPTIMAL`` at the allocator's own tolerance has no
    torque; its error is ``None`` and the status is recorded.
    """
    try:
        result = replay_allocation(problem, weights)
    except NotStrictlyConvexError:
        return {"error": None, "status": "NOT_STRICTLY_CONVEX"}
    if result.status is not SolveStatus.OPTIMAL:
        return {"error": None, "status": f"{result.status.name}: {result.reason}"}
    return {
        "error": relative_torque_error(problem, result.torque, target),
        "error_vs_raw_record": relative_torque_error(
            problem, result.torque, problem.observed_torque
        ),
        "status": "OPTIMAL",
    }


def _window_report(problem, fitted) -> dict:
    stat = stationarity(problem)
    target = stat.projected_torque
    return {
        "frames": len(problem.times_s),
        "family_residual_median": float(np.median(stat.family_residual)),
        "family_residual_max": float(np.max(stat.family_residual)),
        "frames_residual_le_0.10": int(np.sum(stat.family_residual <= 0.10)),
        "frames_binding": len(stat.dropped_binding),
        "frames_infeasible": len(stat.dropped_infeasible),
        "fitted": _held_out_error(problem, fitted, target),
        "default": _held_out_error(problem, DEFAULT_WEIGHTS, target),
        "effort_only": _held_out_error(problem, EFFORT_ONLY, target),
    }


def verdict(fit, ci, h1, h2) -> dict:
    """The registered decision rule (G1 adequacy, G2 identifiability, G3)."""
    g1 = bool(fit.kkt_residual <= RHO_KKT_MAX)
    g2 = bool(
        fit.second_smallest_sv_ratio >= SV_RATIO_MIN
        and (ci[1] - ci[0]) <= CI_WIDTH_MAX
        and not fit.unidentified_features
    )

    def window_ok(w) -> bool:
        e, d = w["fitted"]["error"], w["default"]["error"]
        return e is not None and d is not None and e <= GAIN_MAX * d and e <= ERROR_MAX

    g3 = window_ok(h1) and window_ok(h2)
    if not g1:
        text = "objective family rejected by these data (G1 failed)"
    elif not g2:
        text = "weights not identifiable from these data (G2 failed)"
    elif not g3:
        text = "fitted weights do not beat the defaults on held-out windows (G3 failed)"
    else:
        text = "supported for held-out windows of this one swing only"
    return {"G1": g1, "G2": g2, "G3": g3, "verdict": text}


def test_registered_constants_appear_in_the_report() -> None:
    text = REPORT.read_text()
    section = text.split("## Results")[0]
    for token in (
        "41 frames",
        "0.10",
        "1e-2",
        "200 resamples",
        "12589",
        "0.9",
        "0.25",
        "blocks of 8",
    ):
        assert token in section, token
    assert FIT_FRAMES == 41 and BUFFER_FRAMES == 3 and BOOT_BLOCK == 8
    assert re.search(r"a_e = 1.*a_r = 0\.1", section.replace("\n", " "))


def test_registered_protocol_runs_and_records_the_measurement(bench, runs) -> None:
    by_len = sorted(runs, key=lambda r: len(r[0]), reverse=True)
    run_b, run_a = by_len[0], by_len[1]
    assert (run_b[0][0], run_b[0][-1], len(run_b[0])) == (96, 339, 82)
    assert (run_a[0][0], run_a[0][-1], len(run_a[0])) == (0, 90, 31)

    idx_b, sets_b = run_b
    fit_p = _problem(bench, idx_b[:FIT_FRAMES], sets_b[:FIT_FRAMES])
    start = FIT_FRAMES + BUFFER_FRAMES
    h1_p = _problem(bench, idx_b[start:], sets_b[start:])
    h2_p = _problem(bench, *run_a)
    assert len(h1_p.times_s) == 38 and len(h2_p.times_s) == 31

    stat = stationarity(fit_p)
    fit = fit_allocation_weights(fit_p, precomputed=stat)
    ci = block_bootstrap_log_ratio(
        stat,
        "rate",
        "effort",
        resamples=BOOT_RESAMPLES,
        block=BOOT_BLOCK,
        seed=BOOT_SEED,
    )
    fitted = {k: v for k, v in fit.weight_dict().items() if np.isfinite(v)}
    h1 = _window_report(h1_p, fitted)
    h2 = _window_report(h2_p, fitted)
    in_sample = _window_report(fit_p, fitted)
    outcome = verdict(fit, ci, h1, h2)

    secondary = None
    gross_p = _problem(bench, idx_b[:FIT_FRAMES], sets_b[:FIT_FRAMES], gross=True)
    gross_fit = fit_allocation_weights(gross_p)
    secondary = {
        "weights": gross_fit.weight_dict(),
        "kkt_residual": gross_fit.kkt_residual,
        "identifiable": gross_fit.identifiable,
        "used_frames": len(gross_fit.used_frames),
        "dropped_nonsmooth": len(gross_fit.dropped_nonsmooth),
    }

    record = {
        "host": host_record(),
        "primary_fit": {
            "weights": fit.weight_dict(),
            "rate_over_effort": fit.ratio("rate", "effort"),
            "kkt_residual": fit.kkt_residual,
            "singular_values": fit.singular_values.tolist(),
            "second_smallest_sv_ratio": fit.second_smallest_sv_ratio,
            "tangent_curvature_ratio": fit.tangent_curvature_ratio,
            "identifiable": fit.identifiable,
            "null_directions": fit.null_directions.tolist(),
            "used_frames": len(fit.used_frames),
            "dropped_binding": len(fit.dropped_binding),
            "dropped_infeasible": len(fit.dropped_infeasible),
            "stationarity_rows": fit.n_rows,
            "log10_ratio_ci90": list(ci),
        },
        "fit_window": in_sample,
        "held_out_H1": h1,
        "held_out_H2": h2,
        "secondary_gross_power": secondary,
        "gates": outcome,
    }
    logger.info("NSX-71 measurement: %s", record)
    target = os.environ.get("LOOP_NULLSPACE_INVERSE_JSON")
    if target:
        Path(target).write_text(json.dumps(record, indent=2, default=str))
    assert outcome["verdict"]
    assert fit.feature_names == ("effort", "rate")
