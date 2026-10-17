"""NSX-22 (#12565) and NSX-60 (#12585): the real anthro envelope benchmark.

The Pinocchio anthro model (41 coordinates, 35 actuated, root = 6 unactuated)
is ground-supported: the six root rows of ``M a + h = S^T tau + J_c^T lambda``
are balanced by foot-ground loads.  Foot contacts enter through the adapter
seam (rigid contacts where the compliant sphere gap is within the declared
tolerance), with Coulomb cones of the spec's static friction (a declared
hypothesis).  The benchmark runs ``static_envelopes`` on the 200 evenly strided
frames ``0, 3, ..., 597`` of ``ground_support/dynamics_record.npz``.

Nothing here gates on wall time; set ``LOOP_NULLSPACE_BENCH_JSON`` to a file
path to write the measured record (host, wall times, per-frame statistics).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("highspy")

from src.shared.python.loop_nullspace.decomposition import (  # noqa: E402
    BalanceInconsistentError,
    decompose,
)
from tests.integration.loop_nullspace_lower_body_support import (  # noqa: E402
    BENCHMARK_FRAMES,
    AnthroBench,
    accounting_summary,
    accounting_table,
    bounded_frames,
    build_frames,
    compare_series,
    host_record,
    variant_report,
)

pytestmark = [pytest.mark.integration, pytest.mark.requires_pinocchio]

AIRBORNE_FRAME = 510


@pytest.fixture(scope="module")
def bench() -> AnthroBench:
    return AnthroBench()


@pytest.fixture(scope="module")
def frames(bench):
    return build_frames(bench, BENCHMARK_FRAMES)


def test_benchmark_frames_are_the_200_strided_frames() -> None:
    assert len(BENCHMARK_FRAMES) == 200
    assert BENCHMARK_FRAMES[:3] == (0, 3, 6) and BENCHMARK_FRAMES[-1] == 597


def test_capacity_rows_exist_for_every_actuated_coordinate(bench) -> None:
    assert len(bench.capacity.actuated_names) == 35


@pytest.mark.parametrize("frame", [0, 300])
def test_ground_supported_frames_decompose_with_foot_contacts(bench, frame) -> None:
    f = bench.frame(frame)
    snap = f.snapshot
    assert snap.J_g is not None and snap.J_g.shape[0] == 3 * len(f.contact_names)
    assert f.contact_names  # at least one rigid contact
    assert f.decomposition.n_actuated == 35
    assert f.decomposition.equality_residual < 1e-9
    assert [bench.order[i] for i in snap.topology.unactuated_indices][:3] == [
        "TranslationInputX",
        "TranslationInputY",
        "TranslationInputZ",
    ]


def test_airborne_frame_is_refused_and_not_repaired(bench) -> None:
    """Frame 510 has no sphere within the tolerance and a solver-noise root residual.

    The balance tolerance is not loosened for it.
    """
    with pytest.raises(BalanceInconsistentError) as err:
        bench.frame(AIRBORNE_FRAME)
    assert 1e-9 < err.value.residual_scaled < 1e-6


def test_airborne_frame_without_contacts_has_no_ground_rows(bench) -> None:
    flights = [i for i in (489, 492, 495) if not bench.frame(i).contact_names]
    for i in flights:
        f = bench.frame(i)
        assert f.snapshot.J_g is None
        assert bench.record_observation(f) is None


def test_decompose_without_foot_contacts_still_refuses_a_loaded_frame(bench) -> None:
    from dataclasses import replace

    snap = bench.frame(0).snapshot
    bare = replace(snap, J_g=None, foot_contacts=(), grf_observed=None)
    with pytest.raises(BalanceInconsistentError):
        decompose(bare, weights=np.eye(bare.topology.n_actuated))


@pytest.mark.slow
def test_anthro_200_frame_benchmark_and_ground_reaction_report(bench, frames) -> None:
    built, refused = frames
    assert len(built) + len(refused) == 200
    assert set(refused) <= {AIRBORNE_FRAME}

    accounts = accounting_table(bench, built)
    report: dict = {
        "benchmark_frames": [BENCHMARK_FRAMES[0], BENCHMARK_FRAMES[-1], 3],
        "decomposed_frames": len(built),
        "refused_frames": refused,
        "flight_frames": [f.index for f in built if not f.contact_names],
        "host": host_record(),
        "accounting": {k: accounting_summary(v) for k, v in accounts.items()},
    }
    report["free"], _ = variant_report(bench, built, "free")
    assert report["free"]["frames"] == len(built)
    assert sum(report["free"]["status_counts"].values()) == len(built)
    # Observation rows only add constraints: an empty free set stays empty, so
    # the observed variants run on the frames that have a feasible free set.
    open_frames = bounded_frames(built, report["free"])
    report["observed_frames"] = len(open_frames)
    series = {}
    for kind in ("free", "record", "plate"):
        report[f"{kind}_on_open_frames"], series[kind] = variant_report(
            bench, open_frames, kind
        )
        assert report[f"{kind}_on_open_frames"]["frames"] == len(open_frames)
    for kind in ("record", "plate"):
        report[f"compare_{kind}"] = compare_series(series["free"], series[kind])

    # Observation rows only add constraints: no envelope may widen.
    for kind in ("record", "plate"):
        assert all(c["widened_frames"] == 0 for c in report[f"compare_{kind}"].values())
    # Every infeasible frame names declared hypotheses only.
    for kind in ("free", "record_on_open_frames", "plate_on_open_frames"):
        assert set(report[kind]["conflict_hypotheses"]) <= {
            "H-TORQUE-CAPACITY",
            "H-FRICTION-CONE",
            "H-GRIP-PRELOAD",
            "H-GRF-OBSERVATION",
        }
    # Simulation-record observations agree with the balance of the same record.
    for kind in ("record", "plate"):
        worst = report["accounting"][kind]["max_normalised_residual"]["max"]
        assert worst < 1.0
    # Rank removed never exceeds what the selector can see.
    for kind, rows in (("record", 3), ("plate", 6)):
        assert (
            max(int(r) for r in report["accounting"][kind]["rank_visible_counts"])
            <= rows
        )

    target = os.environ.get("LOOP_NULLSPACE_BENCH_JSON")
    if target:
        Path(target).write_text(json.dumps(report, indent=2, default=str))
