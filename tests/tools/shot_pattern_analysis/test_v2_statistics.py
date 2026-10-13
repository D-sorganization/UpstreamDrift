"""Publication statistics layer: bootstrap, rescoring and head-assumption bands."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from src.tools.shot_pattern_analysis import assumption_band as band
from src.tools.shot_pattern_analysis import headline_selection as headline
from src.tools.shot_pattern_analysis import rescored_statistics as rescore
from src.tools.shot_pattern_analysis.matched_config import matched_preset
from src.tools.shot_pattern_analysis.matched_runner import (
    ExperimentConfig,
    paired_endpoint_statistics,
)
from src.tools.shot_pattern_analysis.nominal_matching import NominalMatchError
from src.tools.shot_pattern_analysis.noise import NoiseSpec
from src.tools.shot_pattern_analysis.uncertainty import (
    HISTORICAL_RESAMPLES,
    PUBLICATION_RESAMPLES,
    _bootstrap_indices,
    paired_mean_difference,
)

from .test_matched_sensitivity import saved_cell

pytestmark = pytest.mark.unit


def _reference_bootstrap(curved, straight, *, resamples, seed):
    """The original sequential loop the vectorized bootstrap must reproduce."""
    differences = np.asarray(curved, float) - np.asarray(straight, float)
    rng = np.random.default_rng(seed)
    estimates = [
        np.mean(differences[rng.integers(0, differences.size, size=differences.size)])
        for _ in range(resamples)
    ]
    return np.quantile(estimates, [0.025, 0.975])


def test_publication_resamples_replace_the_historical_default():
    assert PUBLICATION_RESAMPLES == 2000
    assert HISTORICAL_RESAMPLES == 500


@pytest.mark.parametrize("size,resamples", [(7, 13), (997, 2000), (10_000, 450)])
def test_batched_index_stream_equals_sequential_draws(size, resamples, monkeypatch):
    monkeypatch.setattr(
        "src.tools.shot_pattern_analysis.uncertainty._MAX_BATCH_DRAWS", 5000
    )
    batched = np.concatenate(
        list(_bootstrap_indices(np.random.default_rng(3), size, resamples))
    )
    rng = np.random.default_rng(3)
    sequential = np.stack([rng.integers(0, size, size=size) for _ in range(resamples)])
    np.testing.assert_array_equal(batched, sequential)


def test_vectorized_interval_matches_reference_loop_to_rounding():
    rng = np.random.default_rng(11)
    curved, straight = rng.normal(1.0, 2.0, 3000), rng.normal(0.8, 2.0, 3000)
    result = paired_mean_difference(curved, straight, resamples=500, seed=5)
    lower, upper = _reference_bootstrap(curved, straight, resamples=500, seed=5)
    assert result["lower_95"] == pytest.approx(lower, rel=1e-12, abs=1e-15)
    assert result["upper_95"] == pytest.approx(upper, rel=1e-12, abs=1e-15)
    assert result["lower_95"] <= result["estimate"] <= result["upper_95"]


def _request(club_id="seven_iron"):
    return ExperimentConfig(
        matched_preset(club_id),
        NoiseSpec((1, 0.5, 0, 0, 0, 0, 0)),
        n_shots=4,
        match_face_bounds_deg=(-6.0, 6.0),
    )


@pytest.mark.parametrize("club_id", ["driver", "seven_iron", "pitching_wedge"])
def test_experiment_config_json_round_trip(club_id):
    request = _request(club_id)
    record = json.loads(json.dumps(asdict(request)))
    assert ExperimentConfig.from_dict(record) == request


def test_experiment_config_from_incomplete_record_fails():
    record = json.loads(json.dumps(asdict(_request())))
    del record["noise"]
    with pytest.raises(KeyError):
        ExperimentConfig.from_dict(record)


def _endpoints(n=40, seed=2):
    rng = np.random.default_rng(seed)
    return {
        name: [
            (i, 150.0 + offset + rng.normal(0, 3), rng.normal(0, 5)) for i in range(n)
        ]
        for name, offset in (("Straight", 0.0), ("Draw", 0.5), ("Fade", -0.5))
    }


def test_paired_endpoint_statistics_records_resamples_and_population():
    scoring, comparisons = paired_endpoint_statistics(
        _endpoints(),
        club_id="seven_iron",
        target_x_m=150.0,
        target_radius_m=10.0,
        n_shots=50,
        seed=1,
        include_scoring=False,
        resamples=300,
    )
    assert comparisons["bootstrap_resamples"] == 300
    assert comparisons["paired_n"] == 40
    assert comparisons["conditioning"] == "common_supported_indices_only"
    assert scoring["status"] == "not_requested"
    for kind in (
        "mean_radial_error_difference_vs_straight",
        "mean_squared_target_error_difference_m2",
        "containment_fraction_difference",
    ):
        for pattern in ("Draw", "Fade"):
            interval = comparisons[kind][pattern]
            assert interval["lower_95"] <= interval["estimate"] <= interval["upper_95"]


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda e: e.pop("Fade"), "exactly"),
        (lambda e: e["Draw"].pop(), "same ordered"),
    ],
)
def test_paired_endpoint_statistics_rejects_unpaired_inputs(mutate, message):
    endpoints = _endpoints()
    mutate(endpoints)
    with pytest.raises(ValueError, match=message):
        paired_endpoint_statistics(
            endpoints,
            club_id="seven_iron",
            target_x_m=150.0,
            target_radius_m=10.0,
            n_shots=50,
            seed=1,
            include_scoring=False,
        )


def test_paired_endpoint_statistics_rejects_overfull_population():
    with pytest.raises(ValueError, match="exceed"):
        paired_endpoint_statistics(
            _endpoints(n=10),
            club_id="seven_iron",
            target_x_m=150.0,
            target_radius_m=10.0,
            n_shots=5,
            seed=1,
            include_scoring=False,
        )


def test_head_variants_are_one_factor_and_club_specific():
    iron = matched_preset("seven_iron")
    names = [name for name, _ in band.head_variants(iron)]
    assert names == [
        "cg_depth_x0.8",
        "inertia_x0.8",
        "cg_depth_x1.2",
        "inertia_x1.2",
        "newton_contact_law",
    ]
    driver = dict(band.head_variants(matched_preset("driver")))
    assert set(driver) - set(names) == {"face_radii_x0.8", "face_radii_x1.2"}
    nominal = matched_preset("driver")
    varied = driver["inertia_x1.2"]
    assert varied.inertia_diagonal_kg_m2 == pytest.approx(
        tuple(v * 1.2 for v in nominal.inertia_diagonal_kg_m2)
    )
    assert varied.cg_depth_m == nominal.cg_depth_m
    assert driver["newton_contact_law"].contact_law == band.ALTERNATIVE_CONTACT_LAW
    with pytest.raises(TypeError):
        band.head_variants({"club_id": "driver"})


def _interval(estimate, half=0.01):
    return {
        "estimate": estimate,
        "lower_95": estimate - half,
        "upper_95": estimate + half,
    }


def _fake_result(draw, fade):
    return SimpleNamespace(
        target_x_m=150.0,
        course_scoring={
            "status": "available",
            "paired_n": 4,
            "paired_benefit_vs_straight": {
                "Draw": _interval(draw),
                "Fade": _interval(fade),
            },
        },
        paired_comparisons={"mean_radial_error_difference_vs_straight": {}},
        summary={name: {"supported_n": 4} for name in ("Straight", "Draw", "Fade")},
    )


def test_run_head_variant_applies_variant_and_resamples():
    config = json.loads(json.dumps(asdict(_request())))
    seen = {}

    def analysis(request, *, include_scoring, resamples):
        seen.update(request=request, include_scoring=include_scoring, n=resamples)
        return _fake_result(0.02, -0.01)

    record = band.run_head_variant(
        config, "cg_depth_x1.2", resamples=700, analysis=analysis
    )
    assert seen["n"] == 700 and seen["include_scoring"]
    assert seen["request"].club.cg_depth_m == pytest.approx(
        matched_preset("seven_iron").cg_depth_m * 1.2
    )
    assert record["status"] == "available"
    assert record["paired_benefit_vs_straight"]["Draw"]["estimate"] == 0.02
    json.dumps(record, allow_nan=False)


def test_run_head_variant_records_domain_failures_and_rejects_unknown_names():
    config = json.loads(json.dumps(asdict(_request())))

    def failing(request, **_):
        raise NominalMatchError("no nominal inside the face bounds")

    record = band.run_head_variant(config, "inertia_x0.8", analysis=failing)
    assert record["status"] == "unavailable"
    assert "face bounds" in record["reason"]
    with pytest.raises(ValueError, match="unknown head variant"):
        band.run_head_variant(config, "loft_x2", analysis=failing)


def test_assumption_envelope_spans_every_interval_and_counts_failures():
    nominal = {"Draw": _interval(0.010), "Fade": _interval(-0.030)}
    runs = [
        {
            "variant": "a",
            "paired_benefit_vs_straight": {
                "Draw": _interval(0.002),
                "Fade": _interval(-0.050),
            },
        },
        {
            "variant": "b",
            "paired_benefit_vs_straight": {
                "Draw": _interval(0.030),
                "Fade": _interval(-0.020),
            },
        },
        {"variant": "c", "status": "unavailable"},
    ]
    result = band.assumption_envelope(nominal, runs)
    assert result["variants_attempted"] == 3
    assert result["variants_available"] == 2
    assert result["unavailable_variants"] == ["c"]
    assert result["Draw"]["band_lower_95"] == pytest.approx(-0.008)
    assert result["Draw"]["band_upper_95"] == pytest.approx(0.040)
    assert result["Draw"]["direction"] == "spans_zero"
    assert result["Fade"]["direction"] == "negative"
    assert result["Fade"]["estimate_min"] == -0.050


def test_assumption_envelope_rejects_disordered_intervals():
    nominal = {"Draw": _interval(0.01), "Fade": _interval(0.0)}
    nominal["Draw"]["lower_95"] = math.nan
    with pytest.raises(ValueError, match="finite and ordered"):
        band.assumption_envelope(nominal, [])


def test_plan_tasks_skips_published_runs(tmp_path):
    config = json.loads(json.dumps(asdict(_request())))
    (tmp_path / "cell__cg_depth_x0.8.json").write_text("{}")
    tasks = band.plan_tasks({"cell": config}, tmp_path, 100)
    assert [task[2] for task in tasks] == [
        "inertia_x0.8",
        "cg_depth_x1.2",
        "inertia_x1.2",
        "newton_contact_law",
    ]
    assert {task[5] for task in tasks} == {None}
    shots = tmp_path / "shots.csv.gz"
    with_nominal = band.plan_tasks({"cell": config}, tmp_path, 100, {"cell": shots})
    assert with_nominal[-1][2] == band.NOMINAL_REPRODUCTION
    assert with_nominal[-1][5] == str(shots)


def _shot(pattern, index, status, x=None, y=None):
    return SimpleNamespace(
        pattern=pattern, shot_index=index, status=status, carry_x_m=x, carry_y_m=y
    )


def test_nominal_reproduction_compares_every_archived_shot(tmp_path):
    cell, plan_path = saved_cell(tmp_path, "seven_iron")
    config = json.loads(plan_path.read_text())["cells"][cell.name]
    shots = [
        _shot(p, i, "off_face")
        if p == "Draw" and i == 2
        else _shot(p, i, "supported", 180 + i + 1e-9, (i - 1) * 2)
        for p in ("Straight", "Draw", "Fade")
        for i in range(3)
    ]

    def analysis(request, *, include_scoring):
        assert not include_scoring and request.n_shots == 3
        return SimpleNamespace(shots=shots, target_x_m=180.0)

    record = band.nominal_reproduction(config, cell / "shots.csv.gz", analysis=analysis)
    assert record["shots_compared"] == 9
    assert record["status_mismatches"] == 0
    assert record["max_abs_endpoint_difference_m"] == pytest.approx(1e-9, rel=1e-3)
    shots[0] = _shot("Straight", 0, "off_face")
    assert (
        band.nominal_reproduction(config, cell / "shots.csv.gz", analysis=analysis)[
            "status_mismatches"
        ]
        == 1
    )
    with pytest.raises(ValueError, match="counts differ"):
        band.nominal_reproduction(
            config,
            cell / "shots.csv.gz",
            analysis=lambda *_, **__: SimpleNamespace(shots=shots[:2], target_x_m=1),
        )


def test_headline_cells_are_the_24_zero_lean_conditions_in_order():
    names = headline.headline_cell_names()
    assert len(names) == len(set(names)) == 24
    assert names[0] == "driver_lean0_curve1_rotation1_shaft_only"
    assert names[-1] == "pitching_wedge_lean0_curve2_rotation2_combined2"
    supplemental = [
        n for n in names if headline.headline_source(n) != headline.PRIMARY_PROTOCOL
    ]
    assert supplemental == [
        f"pitching_wedge_lean0_curve2_rotation{sd}_{p}"
        for sd in (1, 2)
        for p in ("shaft_only", "combined2")
    ]
    assert (
        headline.readable_condition(names[0])
        == "Driver, Curve ×1, 1° Twist SD, Shaft Only"
    )
    with pytest.raises(ValueError):
        headline.headline_source("driver_lean5_curve1_rotation1_shaft_only")


def _publish_saved_comparisons(cell, plan_path):
    """Give the synthetic saved cell its own 500-resample paired statistics."""
    plan = json.loads(plan_path.read_text())
    config = plan["cells"][cell.name]
    endpoints, _ = rescore._supported_endpoints(
        cell / "shots.csv.gz", config["n_shots"]
    )
    scoring, comparisons = paired_endpoint_statistics(
        endpoints,
        club_id=config["club"]["club_id"],
        target_x_m=180.0,
        target_radius_m=config["target_radius_m"],
        n_shots=config["n_shots"],
        seed=config["seed"],
        resamples=HISTORICAL_RESAMPLES,
    )
    comparisons.pop("bootstrap_resamples")
    summary = json.loads((cell / "summary.json").read_text())
    summary.update(course_scoring=scoring, paired_comparisons=comparisons)
    (cell / "summary.json").write_text(json.dumps(summary))
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in cell.iterdir()
        if p.name != "manifest.json"
    }
    (cell / "manifest.json").write_text(json.dumps({"files_sha256": hashes}))
    return comparisons


def test_rescore_reproduces_saved_intervals_at_historical_resamples(tmp_path):
    cell, plan_path = saved_cell(tmp_path, "seven_iron")
    saved = _publish_saved_comparisons(cell, plan_path)
    record = rescore.rescore_matrix(tmp_path, resamples=HISTORICAL_RESAMPLES)
    item = record["cells"][cell.name]
    assert item["status"] == "completed"
    recomputed = dict(item["paired_comparisons"])
    assert recomputed.pop("bootstrap_resamples") == HISTORICAL_RESAMPLES
    assert recomputed == saved
    assert set(item["input_sha256"]) == set(rescore.INPUT_FILES)
    assert record["resamples"] == HISTORICAL_RESAMPLES
    json.dumps(record, allow_nan=False)


def test_rescore_lists_missing_cells_and_rejects_unknown_names(tmp_path):
    cell, plan_path = saved_cell(tmp_path, "seven_iron")
    _publish_saved_comparisons(cell, plan_path)
    plan = json.loads(plan_path.read_text())
    plan["cells"]["never_run"] = plan["cells"][cell.name]
    plan_path.write_text(json.dumps(plan))
    record = rescore.rescore_matrix(tmp_path, resamples=200)
    assert record["cells"]["never_run"] == {"status": "not_run"}
    with pytest.raises(ValueError, match="not in the matrix plan"):
        rescore.rescore_matrix(tmp_path, cells=("ghost",))


def test_rescore_fails_closed_on_tampered_shots(tmp_path):
    cell, plan_path = saved_cell(tmp_path, "seven_iron")
    _publish_saved_comparisons(cell, plan_path)
    (cell / "shots.csv.gz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        rescore.rescore_matrix(tmp_path, resamples=200)


def test_rescore_is_identical_across_worker_counts(tmp_path):
    cell, plan_path = saved_cell(tmp_path, "seven_iron")
    _publish_saved_comparisons(cell, plan_path)
    serial = rescore.rescore_matrix(tmp_path, resamples=200)
    parallel = rescore.rescore_matrix(tmp_path, resamples=200, workers=2)
    assert serial == parallel
    with pytest.raises(ValueError, match="workers"):
        rescore.rescore_matrix(tmp_path, workers=0)


def test_interaction_contrasts_follow_the_article_key_order():
    contrasts = headline.INTERACTION_CONTRASTS
    assert [key for key, *_ in contrasts] == [f"E{i:02d}" for i in range(1, 13)]
    assert contrasts[0] == (
        "E01",
        "shaft_rotation_sd",
        "driver_lean0_curve1_rotation1_shaft_only",
        "driver_lean0_curve1_rotation2_shaft_only",
    )
    assert contrasts[-1][1:] == (
        "curve_strength",
        "pitching_wedge_lean0_curve1_rotation2_combined2",
        "pitching_wedge_lean0_curve2_rotation2_combined2",
    )
    assert [key for key, *_ in headline.FIXED_TARGET_CONTRASTS] == ["F1", "F2"]


def _spec(kind, first, second):
    return {
        "comparison_kind": kind,
        "first": {"cell": first},
        "second": {"cell": second},
    }


def test_rescore_contrasts_selects_exactly_one_pair_each(monkeypatch, tmp_path):
    key = ("E01", "curve_strength", "a", "b")
    specs = [_spec("curve_strength", "a", "b"), _spec("curve_strength", "a", "c")]
    seen = []

    def analyze(pair, root, *, resamples):
        seen.append((pair["second"]["cell"], root, resamples))
        return {
            "source_protocols": ["original_control/1", "original_control/1"],
            "status": "available",
            "target_x_m": 150.0,
            "common_supported_n": 10,
            "total_n_per_population": 10,
            "course_scoring_effect_modification": {"Draw": _interval(0.01)},
            "sources": [],
        }

    monkeypatch.setattr(rescore.pairs, "explicit_pair_specs", lambda root: specs)
    monkeypatch.setattr(rescore.pairs, "analyze_pair_spec", analyze)
    record = rescore.rescore_contrasts(tmp_path, (key,), resamples=321)
    assert seen == [("b", tmp_path, 321)]
    assert record["contrasts"]["E01"]["common_six_n"] == 10
    assert record["resamples"] == 321
    specs.append(_spec("curve_strength", "a", "b"))
    with pytest.raises(ValueError, match="selects 2"):
        rescore.rescore_contrasts(tmp_path, (key,))
