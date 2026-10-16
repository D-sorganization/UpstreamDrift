"""Distribution sensitivity within one model (SPN-14a, #12769).

Every model and motion number is INVENTED: the five-level SPN-13a fixture
spine (lumped muscles, tendon slack retuned test-locally) and its five-sample
synthetic golf motion.  The laws are the shipped ``coupling_data`` laws,
restricted to the fixture's lumbar range.  Only the distribution law varies;
the tests check that contract, the per-level spread bookkeeping, provenance,
fail-closed exclusion and determinism, never physiology.  Spreads are
descriptive and never acceptance.
"""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.spine.distribution_sensitivity import (
    KINEMATICS,
    LOADS,
    SensitivityReport,
    build_sensitivity_report,
    classify_spread,
    kinematic_spreads,
    load_spreads,
    run_distribution_sensitivity,
    write_report_csv,
    write_report_json,
)
from src.engines.physics_engines.opensim.python.spine.study import SpineStudyRequest
from src.engines.physics_engines.opensim.python.spine.study_aggregation import (
    GroupStat,
    LevelSummaryTable,
)
from src.shared.python.spine.levels import SpinalLevel
from tests.opensim.spine.test_study import GOLF_MODEL, _request
from tests.opensim.spine.test_study import inputs as _inputs  # noqa: F401
from tests.helpers.opensim_support import require_opensim

LAWS = ("bruno2015_regional", "fujii2007_lumbar", "uniform_reference")
LEVELS = (SpinalLevel.L5_S1, SpinalLevel.L4_L5)
N_LUMBAR = 5
N_AXES = 3

inputs = _inputs


# ------------------------------------------------------------- pure helpers


@pytest.mark.unit
def test_classify_spread_robust_and_sensitive() -> None:
    spread, rel, label = classify_spread({"a": 10.0, "b": 9.5}, 0.1)
    assert spread == pytest.approx(0.5)
    assert rel == pytest.approx(0.05)
    assert label == "robust"
    spread, rel, label = classify_spread({"a": -10.0, "b": 5.0, "c": 0.0}, 0.1)
    assert spread == pytest.approx(15.0)
    assert rel == pytest.approx(1.5)
    assert label == "sensitive"
    assert classify_spread({"a": 0.0, "b": 0.0}, 0.1) == (0.0, 0.0, "robust")


@pytest.mark.unit
def test_classify_spread_preconditions() -> None:
    with pytest.raises(TypeError, match="Mapping"):
        classify_spread([1.0, 2.0], 0.1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="two laws"):
        classify_spread({"a": 1.0}, 0.1)
    with pytest.raises(ValueError, match="finite"):
        classify_spread({"a": 1.0, "b": float("nan")}, 0.1)
    with pytest.raises(ValueError, match="robust_threshold"):
        classify_spread({"a": 1.0, "b": 2.0}, 0.0)


def _group(level: str, quantity: str, law: str, peak: float) -> GroupStat:
    return GroupStat(
        observable="joint_reaction",
        level=level,
        quantity=quantity,
        units="N",
        distribution_source=f"coupling:{law}@1",
        law=f"{law}@1",
        n_runs=1,
        min_peak=peak,
        median_peak=peak,
        max_peak=peak,
        mean_impulse=0.0,
    )


def _table(law: str, scale: float) -> LevelSummaryTable:
    groups = (
        _group("L5_S1", "compression", law, 300.0 * scale),
        _group("L4_L5", "compression", law, 200.0),
    )
    return LevelSummaryTable(rows=(), groups=groups, excluded=())


@pytest.mark.unit
def test_load_spreads_per_level_with_law_labels() -> None:
    tables = {"a": _table("a", 1.0), "b": _table("b", 0.5), "c": _table("c", 1.0)}
    out = load_spreads(tables, 0.1)
    by_level = {s.level: s for s in out}
    assert set(by_level) == {"L5_S1", "L4_L5"}
    l5 = by_level["L5_S1"]
    assert l5.kind == LOADS
    assert dict(l5.values) == {"a": 300.0, "b": 150.0, "c": 300.0}
    assert l5.spread == pytest.approx(150.0)
    assert l5.relative_spread == pytest.approx(0.5)
    assert l5.classification == "sensitive"
    assert l5.max_pairwise_rms is None
    l4 = by_level["L4_L5"]
    assert l4.spread == 0.0 and l4.classification == "robust"
    assert [s.level for s in out] == ["L5_S1", "L4_L5"]


@pytest.mark.unit
def test_load_spreads_needs_two_laws() -> None:
    with pytest.raises(ValueError, match="two laws"):
        load_spreads({"a": _table("a", 1.0)}, 0.1)
    with pytest.raises(TypeError, match="LevelSummaryTable"):
        load_spreads({"a": _table("a", 1.0), "b": "x"}, 0.1)  # type: ignore[dict-item]


@pytest.mark.unit
def test_kinematic_spreads_peaks_and_pairwise_rms() -> None:
    base = np.zeros((4, 2, 3))
    base[:, 0, 2] = [0.0, 0.1, 0.2, 0.1]
    other = base.copy()
    other[:, 0, 2] *= 2.0
    out = kinematic_spreads(LEVELS, {"a": base, "b": other}, 0.1)
    assert len(out) == len(LEVELS) * N_AXES
    ar = next(s for s in out if s.level == "L5_S1" and s.quantity == "axial_rotation")
    assert ar.kind == KINEMATICS and ar.units == "rad"
    assert dict(ar.values) == pytest.approx({"a": 0.2, "b": 0.4})
    assert ar.spread == pytest.approx(0.2)
    expected = float(np.sqrt(np.mean((other[:, 0, 2] - base[:, 0, 2]) ** 2)))
    assert ar.max_pairwise_rms == pytest.approx(expected)
    assert ar.classification == "sensitive"
    fe = next(
        s for s in out if s.level == "L4_L5" and s.quantity == "flexion_extension"
    )
    assert fe.spread == 0.0 and fe.classification == "robust"


@pytest.mark.unit
def test_kinematic_spreads_preconditions() -> None:
    a = np.zeros((4, 2, 3))
    with pytest.raises(ValueError, match="two laws"):
        kinematic_spreads(LEVELS, {"a": a}, 0.1)
    with pytest.raises(ValueError, match="shape"):
        kinematic_spreads(LEVELS, {"a": a, "b": np.zeros((3, 2, 3))}, 0.1)
    with pytest.raises(ValueError, match="shape"):
        kinematic_spreads(LEVELS, {"a": a, "b": np.zeros((4, 3, 3))}, 0.1)


@pytest.mark.unit
def test_restrict_flag_on_request(tmp_path) -> None:
    ok = {
        "study_id": "s",
        "motion_path": tmp_path / "m.sto",
        "golf_model_path": tmp_path / "g.osim",
        "model_path": tmp_path / "x.osim",
        "model_id": "fixture",
        "output_dir": tmp_path,
    }
    plain = SpineStudyRequest(**ok)
    assert "restrict_law_to_model" not in plain.canonical_dict()
    flagged = SpineStudyRequest(
        **ok, coupling_law_id="uniform_reference", restrict_law_to_model=True
    )
    assert flagged.canonical_dict()["restrict_law_to_model"] is True
    with pytest.raises(ValueError, match="coupling_law_id"):
        SpineStudyRequest(**ok, restrict_law_to_model=True)
    with pytest.raises(TypeError, match="restrict_law_to_model"):
        SpineStudyRequest(**ok, coupling_law_id="x", restrict_law_to_model=1)


@pytest.mark.unit
def test_run_preconditions(tmp_path) -> None:
    base = SpineStudyRequest(
        study_id="s",
        motion_path=tmp_path / "m.sto",
        golf_model_path=GOLF_MODEL,
        model_path=tmp_path / "x.osim",
        model_id="fixture",
        output_dir=tmp_path,
    )
    with pytest.raises(TypeError, match="SpineStudyRequest"):
        run_distribution_sensitivity("nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="two"):
        run_distribution_sensitivity(base, ("uniform_reference",))
    with pytest.raises(ValueError, match="duplicate"):
        run_distribution_sensitivity(base, ("uniform_reference",) * 2)
    with pytest.raises(ValueError, match="pending"):
        run_distribution_sensitivity(base, ("uniform_reference", "pearcy1984_lumbar"))
    with pytest.raises(ValueError, match="model_path"):
        run_distribution_sensitivity(
            replace(base, model_path=None, asset_id="bruno2015_fullbody_os4_v2"),
            LAWS,
            env={},
        )


@pytest.mark.unit
def test_report_model_path_resolves_the_asset_environment(tmp_path) -> None:
    from src.engines.physics_engines.opensim.python.spine import (
        distribution_sensitivity as ds,
    )

    base = SpineStudyRequest(
        study_id="s",
        motion_path=tmp_path / "m.sto",
        golf_model_path=GOLF_MODEL,
        model_path=None,
        asset_id="bruno2015_fullbody_os4_v2",
        model_id="fixture",
        output_dir=tmp_path,
    )
    env = {"UD_SPINE_BRUNO_SOURCE": str(tmp_path / "m.osim")}
    assert ds._report_model_path(base, env) == tmp_path / "m.osim"


# ------------------------------------------------------------ native pipeline


@pytest.fixture(scope="module")
def report_dir(inputs, tmp_path_factory) -> tuple[SensitivityReport, Path]:
    require_opensim()
    out = tmp_path_factory.mktemp("sensitivity")
    base = _request(inputs, out, study_id="sens")
    report = run_distribution_sensitivity(base, LAWS, code_version="c")
    return report, out


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_only_the_law_varies(report_dir) -> None:
    report, _ = report_dir
    assert [law.law_id for law in report.laws] == sorted(LAWS)
    assert {r.role for r in report.held_fixed} >= {
        "motion",
        "golf_model",
        "segmental_model",
    }
    assert report.scientific_status == "unqualified"
    assert report.is_acceptance is False
    assert report.model_id == "fixture"


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_every_law_carries_provenance(report_dir) -> None:
    report, _ = report_dir
    by_id = {law.law_id: law for law in report.laws}
    for law_id, law in by_id.items():
        assert law.status == "pass", law.status_reasons
        assert law.citation.strip()
        assert law.distribution_source.startswith(f"coupling:{law_id}@")
        assert "levels:L5_S1..L1_L2" in law.law
        assert law.drive_max_rad >= 0.0
    assert by_id["uniform_reference"].verified_from_primary is False
    assert by_id["bruno2015_regional"].verified_from_primary is True
    # a one-coordinate axial law cannot reach a three-axis target
    assert by_id["fujii2007_lumbar"].drive_max_rad > 0.1
    assert by_id["bruno2015_regional"].drive_max_rad < 1e-9
    assert report.excluded == ()


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_per_level_kinematic_and_load_spreads(report_dir) -> None:
    report, _ = report_dir
    kin = [s for s in report.spreads if s.kind == KINEMATICS]
    assert len(kin) == N_LUMBAR * N_AXES
    for item in kin:
        values = dict(item.values)
        assert set(values) == set(LAWS)
        # within-region uniform Bruno equals the uniform law on a lumbar-only model
        assert values["bruno2015_regional"] == pytest.approx(
            values["uniform_reference"], abs=1e-9
        )
    assert any(s.classification == "sensitive" for s in kin)
    loads = [s for s in report.spreads if s.kind == LOADS]
    comp = [
        s
        for s in loads
        if s.observable == "joint_reaction" and s.quantity == "compression"
    ]
    assert len(comp) == N_LUMBAR
    assert all(np.isfinite(s.spread) and s.spread >= 0.0 for s in loads)
    assert all(set(dict(s.values)) == set(LAWS) for s in loads)


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_report_is_deterministic_and_written(report_dir, inputs, tmp_path) -> None:
    report, out = report_dir
    receipts = {law.law_id: out / law.receipt_name for law in report.laws}
    rebuilt = build_sensitivity_report("sens", receipts, inputs["model"], "fixture")
    one = write_report_json(report, tmp_path / "a.json").read_bytes()
    two = write_report_json(rebuilt, tmp_path / "b.json").read_bytes()
    assert one == two
    payload = json.loads(one)
    assert payload["scientific_status"] == "unqualified"
    assert payload["is_acceptance"] is False
    assert str(tmp_path) not in one.decode()
    path = write_report_csv(report, tmp_path / "spreads.csv")
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header = rows[0]
    assert "spread" in header and "classification" in header
    assert all(f"value[{law}]" in header for law in LAWS)
    assert len(rows) - 1 == len(report.spreads)


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_unreproduced_input_is_excluded_not_tuned(inputs, tmp_path) -> None:
    require_opensim()
    base = _request(inputs, tmp_path, study_id="tol", drive_tolerance_rad=1e-6)
    report = run_distribution_sensitivity(base, LAWS, code_version="c")
    excluded = {e.law_id: e for e in report.excluded}
    assert set(excluded) == {"fujii2007_lumbar"}
    assert excluded["fujii2007_lumbar"].status == "unqualified"
    assert any("drive" in r for r in excluded["fujii2007_lumbar"].status_reasons)
    for item in report.spreads:
        assert set(dict(item.values)) == {"bruno2015_regional", "uniform_reference"}
