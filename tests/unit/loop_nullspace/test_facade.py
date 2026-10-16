"""NSX-40: the single-entry facade composes NSX modules and stays honest (#12572)."""

from __future__ import annotations

import pytest

pytest.importorskip("highspy")

from src.shared.python.loop_nullspace import facade  # noqa: E402
from src.shared.python.loop_nullspace.decomposition import (  # noqa: E402
    BalanceInconsistentError,
)
from src.shared.python.loop_nullspace.facade import (  # noqa: E402
    AnalysisCancelledError,
    AnalysisOptions,
    RefusedFrame,
    analyze,
)
from tests.unit.loop_nullspace.report_support import (  # noqa: E402
    planar_engine,
    planar_trajectory,
)

pytestmark = pytest.mark.unit


def test_stage_order_has_invariants_before_envelopes() -> None:
    seen: list[tuple[str, bool]] = []
    analyze(
        planar_trajectory(),
        planar_engine(),
        "envelope_only",
        on_stage=lambda stage, pct, partial: seen.append((stage, partial is not None)),
    )
    names = [s for s, _ in seen]
    assert names == [
        "decomposition",
        "admission",
        "invariants",
        "feasible_sets",
        "envelopes",
    ]
    assert dict(seen)["invariants"] is True


def test_partial_invariants_payload_is_the_report_section() -> None:
    partials: list = []
    report = analyze(
        planar_trajectory(),
        planar_engine(),
        "envelope_only",
        on_stage=lambda s, p, partial: partials.append(partial) if partial else None,
    )
    assert partials[0]["invariants"] == report["invariants"]


def test_cancel_stops_between_stages() -> None:
    calls = {"n": 0}

    def cancel() -> bool:
        calls["n"] += 1
        return calls["n"] > 3

    with pytest.raises(AnalysisCancelledError):
        analyze(
            planar_trajectory(), planar_engine(), "envelope_only", should_cancel=cancel
        )


def test_unknown_preset_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="unknown preset"):
        analyze(planar_trajectory(), planar_engine(), "nope")


def test_grf_preset_without_an_observation_feed_is_refused() -> None:
    with pytest.raises(ValueError, match="H-GRF-OBSERVATION"):
        analyze(planar_trajectory(), planar_engine(), "envelope_grf_observed")


def test_allocation_requires_envelopes() -> None:
    with pytest.raises(ValueError):
        AnalysisOptions(include_envelopes=False, include_allocation=True)


def test_a_frame_that_cannot_balance_is_refused_not_repaired(monkeypatch) -> None:
    real = facade.decompose
    state = {"n": 0}

    def flaky(snap, **kw):
        state["n"] += 1
        if state["n"] == 2:
            raise BalanceInconsistentError("unbalanced", 3e-8)
        return real(snap, **kw)

    monkeypatch.setattr(facade, "decompose", flaky)
    report = analyze(
        planar_trajectory((0, 3, 6, 9, 12)), planar_engine(), "envelope_only"
    )
    refused = report["trajectory"]["refused_frames"]
    assert [r["index"] for r in refused] == [3]
    assert refused[0]["residual_scaled"] == pytest.approx(3e-8)
    assert 3 not in report["trajectory"]["analyzed_frame_indices"]
    assert report["trajectory"]["n_input_frames"] == 5
    assert [f["index"] for f in report["invariants"]["frames"]] == [0, 6, 9, 12]


def test_frames_refused_while_building_are_listed() -> None:
    traj = planar_trajectory()
    traj = type(traj)(
        snapshots=traj.snapshots,
        times_s=traj.times_s,
        frame_indices=traj.frame_indices,
        provenance=traj.provenance,
        refused=(RefusedFrame(510, "build", "no contact", None),),
    )
    report = analyze(traj, planar_engine(), "envelope_only")
    assert report["trajectory"]["refused_frames"][0]["index"] == 510
    assert 510 in report["trajectory"]["frame_indices"]


def test_too_few_frames_stop_honestly() -> None:
    report = analyze(planar_trajectory((0, 3)), planar_engine(), "envelope_only")
    assert report["status"] == "insufficient_frames"
    assert report["admission"] is None and report["invariants"] is None


def test_non_increasing_times_are_a_precondition_error() -> None:
    traj = planar_trajectory()
    bad = type(traj)(
        snapshots=traj.snapshots,
        times_s=(0.0, 0.01, 0.01, 0.02),  # admission needs strictly increasing times
        frame_indices=traj.frame_indices,
        provenance=traj.provenance,
    )
    with pytest.raises(ValueError):
        analyze(bad, planar_engine(), "envelope_only")


def test_infeasible_frames_are_listed_and_excluded_from_allocation() -> None:
    pytest.importorskip("clarabel")
    report = analyze(
        planar_trajectory(),
        planar_engine(limit_nm=1e-3),
        "capacity_min_effort",
        AnalysisOptions(include_allocation=True),
    )
    assert report["allocation"]["status"] == "not_run"
    assert len(report["allocation"]["excluded_frames"]) == 4
    assert all(not f["feasible"] for f in report["envelopes"]["frames"])


def test_registry_version_is_embedded() -> None:
    report = analyze(planar_trajectory(), planar_engine(), "envelope_only")
    assert report["registry_version"] == report["hypotheses"]["registry_version"]
    assert report["registry_version"]
