"""Tests for study aggregation (SPN-13c, #12768).

Every number is INVENTED. Receipts and ``_loads.npz`` artifacts are built
synthetically in ``tmp_path`` with the SPN-13a record types; hand-checked
peaks, impulses and group statistics, fail-closed hash checks and
order-independence are tested, not physiology.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.spine.study_aggregation import (
    AggregationIntegrityError,
    LevelSummaryTable,
    aggregate_receipts,
    compare,
    load_verified_run,
    summarize_run,
    write_csv,
    write_json,
)
from src.engines.physics_engines.opensim.python.spine.study_receipt import (
    FileRecord,
    SpineStudyReceipt,
    StageRecord,
    write_receipt,
)
from src.shared.python.spine.levels import SpinalLevel

pytestmark = pytest.mark.unit

LEVELS = (SpinalLevel.L5_S1, SpinalLevel.L4_L5)
TIME = np.array([0.0, 1.0, 2.0, 3.0])
SHA = "0123456789abcdef" * 4


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_study(
    folder: Path,
    study_id: str = "s1",
    status: str = "pass",
    source: str = "law-a",
    scale: float = 1.0,
    context: tuple[dict, ...] = (),
) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    force = np.zeros((4, 2, 3))
    force[:, 0, 1] = scale * np.array([0.0, 100.0, 300.0, 200.0])  # L5-S1 compression
    force[:, 1, 1] = scale * np.array([0.0, 10.0, 20.0, 10.0])
    force[:, 0, 0] = scale * np.array([0.0, -50.0, 40.0, 0.0])  # AP shear
    force[:, 0, 2] = scale * np.array([0.0, 5.0, 0.0, 0.0])
    moment = np.zeros((4, 2, 3))
    moment[:, 0, 1] = scale * np.array([0.0, 1.0, -4.0, 2.0])  # axial
    id_moment = np.zeros((4, 2, 3))
    id_moment[:, 0, 2] = scale * np.array([0.0, 7.0, 9.0, 1.0])
    npz = folder / f"{study_id}_loads.npz"
    np.savez(
        npz,
        time_s=TIME,
        frame_index=np.arange(4),
        levels=np.array([lv.name for lv in LEVELS]),
        id_time_s=TIME,
        id_moment_nm=id_moment,
        joint_reaction_force_n=force,
        joint_reaction_moment_nm=moment,
    )
    receipt = SpineStudyReceipt(
        study_id=study_id,
        status=status,  # type: ignore[arg-type]
        status_reasons=() if status == "pass" else ("invented reason",),
        request={},
        request_sha256=SHA,
        inputs=(),
        code_version="x",
        opensim_version="4.6",
        scipy_version="1",
        numpy_version="1",
        law="law-id",
        distribution_source=source,
        asset=None,
        stages=(StageRecord("drive", "ok"),),
        artifacts=(FileRecord("loads_npz", npz.name, _sha(npz)),),
        residual={},
        reserve={},
        context_comparisons=context,
    )
    return write_receipt(receipt, folder / f"{study_id}_receipt.json")


def _ctx(level: str = "L5_S1", ratio: float = 0.5) -> dict:
    return {
        "band_id": "b1",
        "level": level,
        "quantity": "compression",
        "ratio": ratio,
        "is_acceptance": False,
    }


def test_load_verified_run_reads_arrays(tmp_path: Path) -> None:
    run = load_verified_run(_write_study(tmp_path))
    assert run.levels == LEVELS
    assert run.joint_reaction_force_n.shape == (4, 2, 3)


def test_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    path = _write_study(tmp_path)
    (tmp_path / "s1_loads.npz").write_bytes(b"tampered")
    with pytest.raises(AggregationIntegrityError, match="sha256"):
        load_verified_run(path)


def test_missing_artifact_fails_closed(tmp_path: Path) -> None:
    path = _write_study(tmp_path)
    (tmp_path / "s1_loads.npz").unlink()
    with pytest.raises(AggregationIntegrityError, match="missing"):
        load_verified_run(path)


@pytest.mark.parametrize(
    "name", ["../outside/s1_loads.npz", "sub/s1_loads.npz", "/abs/s1_loads.npz"]
)
def test_artifact_name_must_be_a_basename(tmp_path: Path, name: str) -> None:
    outside = tmp_path / "outside"
    _write_study(outside)
    inside = tmp_path / "inside"
    inside.mkdir()
    text = (outside / "s1_receipt.json").read_text()
    receipt = inside / "s1_receipt.json"
    receipt.write_text(text.replace('"s1_loads.npz"', json.dumps(name)))
    with pytest.raises(ValueError, match="basename"):
        load_verified_run(receipt)


def test_peaks_and_impulse_hand_checked(tmp_path: Path) -> None:
    rows = summarize_run(load_verified_run(_write_study(tmp_path)))
    jr = {
        (r.level, q.name): q
        for r in rows
        if r.observable == "joint_reaction"
        for q in r.stats
    }
    comp = jr[("L5_S1", "compression")]
    assert comp.peak == 300.0 and comp.t_peak_s == 2.0
    # trapezoid of [0,100,300,200] at dt=1 -> 50 + 200 + 250
    assert comp.impulse == pytest.approx(500.0)
    ap = jr[("L5_S1", "ap_shear")]
    assert ap.peak == -50.0 and ap.t_peak_s == 1.0
    assert jr[("L5_S1", "moment_y")].peak == -4.0


def test_window_limits_impulse_only(tmp_path: Path) -> None:
    run = load_verified_run(_write_study(tmp_path))
    rows = summarize_run(run, window_s=(1.0, 2.0))
    row = next(r for r in rows if r.observable == "joint_reaction")
    comp = next(q for q in row.stats if q.name == "compression")
    assert comp.impulse == pytest.approx(200.0)
    assert comp.peak == 300.0
    assert row.window_s == (1.0, 2.0)


def test_bad_window_rejected(tmp_path: Path) -> None:
    run = load_verified_run(_write_study(tmp_path))
    with pytest.raises(ValueError, match="window"):
        summarize_run(run, window_s=(2.0, 1.0))
    with pytest.raises(ValueError, match="window"):
        summarize_run(run, window_s=(-1.0, 1.0))


def test_body_weight_normalisation(tmp_path: Path) -> None:
    run = load_verified_run(_write_study(tmp_path))
    rows = summarize_run(run, body_mass_kg=30.0)
    row = next(
        r for r in rows if r.observable == "joint_reaction" and r.level == "L5_S1"
    )
    assert row.peak_compression_bw == pytest.approx(300.0 / (30.0 * 9.80665))
    with pytest.raises(ValueError, match="body_mass_kg"):
        summarize_run(run, body_mass_kg=-1.0)


def test_observables_stay_separate_and_labelled(tmp_path: Path) -> None:
    rows = summarize_run(load_verified_run(_write_study(tmp_path, source="law-a")))
    kinds = {r.observable for r in rows}
    assert kinds == {"generalized_effort", "joint_reaction"}
    assert all(r.distribution_source == "law-a" and r.law == "law-id" for r in rows)
    ge = next(r for r in rows if r.observable == "generalized_effort")
    assert {q.name for q in ge.stats} == {"moment_x", "moment_y", "moment_z"}


def test_context_annotation_not_acceptance(tmp_path: Path) -> None:
    path = _write_study(tmp_path, context=(_ctx("L5_S1"), _ctx("L4_L5", 0.2)))
    rows = summarize_run(load_verified_run(path))
    row = next(
        r for r in rows if r.observable == "joint_reaction" and r.level == "L5_S1"
    )
    assert len(row.context) == 1 and row.context[0]["is_acceptance"] is False
    ge = next(r for r in rows if r.observable == "generalized_effort")
    assert ge.context == ()


def test_acceptance_context_fails_closed(tmp_path: Path) -> None:
    bad = dict(_ctx(), is_acceptance=True)
    path = _write_study(tmp_path, context=(bad,))
    with pytest.raises(AggregationIntegrityError, match="is_acceptance"):
        summarize_run(load_verified_run(path))


def test_unqualified_and_failed_excluded_and_counted(tmp_path: Path) -> None:
    paths = [
        _write_study(tmp_path / "a", "a", "pass"),
        _write_study(tmp_path / "b", "b", "unqualified", scale=100.0),
        _write_study(tmp_path / "c", "c", "failed"),
    ]
    table = aggregate_receipts(paths)
    assert table.n_included == 1 and table.n_excluded == 2
    assert {e.study_id: e.status for e in table.excluded} == {
        "b": "unqualified",
        "c": "failed",
    }
    assert {r.study_id for r in table.rows} == {"a"}
    assert all(g.n_runs == 1 for g in table.groups)


def test_groups_split_by_distribution_source(tmp_path: Path) -> None:
    paths = [
        _write_study(tmp_path / "a", "a", source="law-a", scale=1.0),
        _write_study(tmp_path / "b", "b", source="law-a", scale=3.0),
        _write_study(tmp_path / "c", "c", source="law-b", scale=2.0),
    ]
    table = aggregate_receipts(paths)
    key = ("joint_reaction", "L5_S1", "compression")
    groups = {
        g.distribution_source: g
        for g in table.groups
        if (g.observable, g.level, g.quantity) == key
    }
    assert set(groups) == {"law-a", "law-b"}
    assert groups["law-a"].n_runs == 2
    assert groups["law-a"].min_peak == 300.0 and groups["law-a"].max_peak == 900.0
    assert groups["law-a"].median_peak == 600.0
    assert groups["law-b"].median_peak == 600.0


def test_order_independent(tmp_path: Path) -> None:
    paths = [
        _write_study(tmp_path / s, s, scale=float(i + 1)) for i, s in enumerate("abc")
    ]
    one = aggregate_receipts(paths)
    two = aggregate_receipts(list(reversed(paths)))
    assert one == two


def test_duplicate_study_id_rejected(tmp_path: Path) -> None:
    a = _write_study(tmp_path / "a", "dup")
    b = _write_study(tmp_path / "b", "dup")
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_receipts([a, b])


def test_empty_input_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        aggregate_receipts([])


def test_tampered_included_run_fails_whole_aggregate(tmp_path: Path) -> None:
    path = _write_study(tmp_path / "a", "a")
    (tmp_path / "a" / "a_loads.npz").write_bytes(b"x")
    with pytest.raises(AggregationIntegrityError):
        aggregate_receipts([path])


def test_compare_against_baseline(tmp_path: Path) -> None:
    base = aggregate_receipts([_write_study(tmp_path / "a", "a", source="law-a")])
    var = aggregate_receipts(
        [_write_study(tmp_path / "b", "b", source="law-b", scale=2.0)]
    )
    result = compare(base, {"v": var})
    delta = next(
        d
        for d in result.deltas
        if (d.observable, d.level, d.quantity)
        == ("joint_reaction", "L5_S1", "compression")
    )
    assert delta.variant == "v"
    assert delta.delta == 300.0 and delta.ratio == pytest.approx(2.0)
    assert delta.baseline_source == "law-a" and delta.variant_source == "law-b"
    assert delta.baseline_law == delta.variant_law == "law-id"


def test_compare_skips_ambiguous_sources(tmp_path: Path) -> None:
    base = aggregate_receipts(
        [
            _write_study(tmp_path / "a", "a", source="law-a"),
            _write_study(tmp_path / "b", "b", source="law-b"),
        ]
    )
    var = aggregate_receipts([_write_study(tmp_path / "c", "c")])
    result = compare(base, {"v": var})
    assert not result.deltas
    assert result.unmatched
    with pytest.raises(ValueError, match="variants"):
        compare(base, {})


def test_writers_deterministic_with_units(tmp_path: Path) -> None:
    table = aggregate_receipts([_write_study(tmp_path / "a", "a", context=(_ctx(),))])
    csv_path = write_csv(table, tmp_path / "out" / "t.csv")
    json_path = write_json(table, tmp_path / "out" / "t.json")
    again = write_csv(table, tmp_path / "out" / "t2.csv")
    assert csv_path.read_bytes() == again.read_bytes()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    quantity_cols = [
        h for h in header if h.startswith(("peak_", "impulse_", "t_peak_"))
    ]
    assert quantity_cols and all(h.endswith("]") for h in quantity_cols)
    assert "peak_compression[N]" in header
    assert "impulse_compression[N*s]" in header
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["n_included"] == 1 and data["scientific_status"] == "unqualified"
    assert isinstance(table, LevelSummaryTable)
