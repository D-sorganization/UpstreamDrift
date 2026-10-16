"""NSX-23 (#12566) acceptance: anthro-fixture trajectory allocation.

"Anthro fixture allocation converges with recorded solve time and KKT
residuals."  The foot-contact Jacobian ``J_g`` and Coulomb cones come from
NSX-60 (#12585), so the ground-supported anthro frames decompose and their
feasible sets (``H-TORQUE-CAPACITY`` prior, unchanged, plus ``H-FRICTION-CONE``)
can be allocated.

A trajectory allocation is feasible only if every frame's set is non-empty.  On
the 199 decomposable benchmark frames, 75 are INFEASIBLE under the ASSUMED
capacity prior (recorded efforts exceed it), so the full window is infeasible
by construction and reports ``INFEASIBLE``; the capacity prior is never relaxed.
The acceptance window is the longest contiguous run of feasible frames.  The
record's frame spacing is 3 samples (0.009 s), so the rate term differences
strided frames.

Set ``LOOP_NULLSPACE_ALLOC_JSON`` to a file path to write the measured record.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("highspy")
pytest.importorskip("clarabel")

from src.shared.python.loop_nullspace.allocation import (  # noqa: E402
    ObjectiveSpec,
    allocate_trajectory,
    torque_scale_from_capacity,
)
from src.shared.python.loop_nullspace.evidence import (  # noqa: E402
    AllocationDependence,
)
from src.shared.python.loop_nullspace.solvers import SolveStatus  # noqa: E402
from tests.integration.loop_nullspace_lower_body_support import (  # noqa: E402
    BENCHMARK_FRAMES,
    AnthroBench,
    Frame,
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
KKT_TOL = 1e-6
EFFORT_WEIGHT = 1.0
RATE_WEIGHT = 0.1


@pytest.fixture(scope="module")
def bench() -> AnthroBench:
    return AnthroBench()


@pytest.fixture(scope="module")
def decomposed(bench):
    frames, _refused = build_frames(bench, BENCHMARK_FRAMES)
    sets = [bench.feasible_set(f, None) for f in frames]
    return frames, sets, [not fs.is_empty() for fs in sets]


@pytest.fixture(scope="module")
def window(decomposed) -> list[int]:
    """Positions (in ``decomposed``) of the longest contiguous feasible run."""
    frames, _sets, flags = decomposed
    runs = contiguous_runs(frames, flags)
    longest = max(runs, key=len)
    first = frames.index(longest[0])
    return list(range(first, first + len(longest)))


def _objective(bench, frames: list[Frame]) -> ObjectiveSpec:
    times = np.array([bench.record["time_s"][f.index] for f in frames])
    return ObjectiveSpec(
        times_s=times,
        torque_scale=torque_scale_from_capacity(bench.capacity),
        effort_weight=EFFORT_WEIGHT,
        rate_weight=RATE_WEIGHT,
    )


def _measure(bench, frames, sets, positions) -> tuple[dict, object]:
    chosen = [frames[i] for i in positions]
    result = allocate_trajectory(
        [sets[i] for i in positions], _objective(bench, chosen)
    )
    summary = {
        "frames": len(chosen),
        "record_frames": [chosen[0].index, chosen[-1].index],
        "status": result.status.name,
        "solve_time_s": result.solve_time_s,
        "iterations": result.iterations,
        "kkt": None
        if result.kkt_residuals is None
        else {
            "primal": result.kkt_residuals.primal,
            "dual": result.kkt_residuals.dual,
            "complementarity": result.kkt_residuals.complementarity,
            "max": result.kkt_residuals.max(),
        },
        "objective_terms": result.objective_terms,
        "power_reduction": None
        if result.power_reduction is None
        else result.power_reduction.kind,
        "gauge_dimension": None
        if result.convexity is None
        else result.convexity.gauge_dimension,
        "retained_rank": None if result.convexity is None else result.convexity.rank,
        "reason": result.reason,
    }
    return summary, result


def test_infeasible_frames_make_the_full_window_infeasible(bench, decomposed) -> None:
    """75 frames violate the ASSUMED capacity prior: the full window is empty."""
    frames, sets, flags = decomposed
    assert not all(flags) and any(flags)
    summary, result = _measure(bench, frames, sets, list(range(len(frames))))
    logger.info("full window: %s", summary)
    assert result.status is SolveStatus.INFEASIBLE
    assert result.record is None  # never a relaxed answer


def test_window_across_one_infeasible_frame_is_infeasible(bench, decomposed) -> None:
    frames, sets, flags = decomposed
    bad = flags.index(False)
    positions = [i for i in range(max(0, bad - 1), bad + 2) if i < len(frames)]
    _summary, result = _measure(bench, frames, sets, positions)
    assert result.status is SolveStatus.INFEASIBLE


def test_longest_feasible_run_allocation_converges(bench, decomposed, window) -> None:
    frames, sets, _flags = decomposed
    assert len(window) >= 60
    summary, result = _measure(bench, frames, sets, window)
    logger.info("longest feasible run: %s", summary)
    assert result.status is SolveStatus.OPTIMAL, result.reason
    assert result.kkt_residuals is not None and result.kkt_residuals.max() < KKT_TOL
    assert result.solve_time_s > 0.0 and result.iterations > 0
    record = result.record
    assert record is not None
    assert record.allocation_dependence is AllocationDependence.FAMILY_DEPENDENT
    assert "H-TORQUE-CAPACITY" in record.hypotheses
    assert "H-FRICTION-CONE" in record.hypotheses
    # Torques respect the capacity prior the feasible sets were built with.
    cap = bench.capacity
    tau = result.torque
    assert np.all(tau <= cap.tau_max_positive_nm + 1e-6)
    assert np.all(tau >= -cap.tau_max_negative_nm - 1e-6)

    short, short_result = _measure(bench, frames, sets, window[:10])
    assert short_result.status is SolveStatus.OPTIMAL, short_result.reason
    assert short["kkt"]["max"] < KKT_TOL

    full, _ = _measure(bench, frames, sets, list(range(len(frames))))
    target = os.environ.get("LOOP_NULLSPACE_ALLOC_JSON")
    if target:
        report = {
            "host": host_record(),
            "weights": {"effort": EFFORT_WEIGHT, "rate": RATE_WEIGHT},
            "feasible_frames": sum(_flag for _flag in decomposed[2]),
            "decomposed_frames": len(frames),
            "longest_run": summary,
            "first_10_of_run": short,
            "full_window": full,
        }
        Path(target).write_text(json.dumps(report, indent=2, default=str))
