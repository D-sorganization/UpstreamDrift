"""Contracts added by the post-integration shot-pattern review.

Each test pins one defect found in review: platform-dependent provenance keys,
a clean-checkout crash, mid-run delivery failures, unmatched comparisons,
unbounded putting extrapolation, path escapes and silent progress gaps.
"""

import math
from pathlib import Path

import pytest

from src.tools.shot_pattern_analysis import comparison, core, matched_export
from src.tools.shot_pattern_analysis import matched_matrix, provenance, scoring
from src.tools.shot_pattern_analysis import sensitivity
from src.tools.shot_pattern_analysis.delivery_geometry import (
    reachable_face_range_deg,
)
from src.tools.shot_pattern_analysis.matched_config import matched_preset
from src.tools.shot_pattern_analysis.matched_runner import (
    ExperimentConfig,
    run_matched_analysis,
)
from src.tools.shot_pattern_analysis.noise import NoiseSpec
from tests.tools.shot_pattern_analysis.test_matched_runner import AnalyticPhysics

pytestmark = pytest.mark.unit


# --- Provenance keys are portable and hashing is shared -------------------


def test_relative_file_hashes_use_posix_keys(tmp_path):
    nested = tmp_path / "a" / "b.txt"
    nested.parent.mkdir()
    nested.write_bytes(b"x")
    hashes = provenance.relative_file_hashes([nested], tmp_path)
    assert hashes == {"a/b.txt": provenance.sha256_file(nested)}


def test_relative_file_hashes_reject_files_outside_root(tmp_path):
    outside = tmp_path.parent / "outside-review-hardening.txt"
    with pytest.raises(ValueError):
        provenance.relative_file_hashes([outside], tmp_path / "root")


def test_sensitivity_hashes_survive_clean_checkout_without_cargo_lock(tmp_path):
    hashes = sensitivity.scientific_source_hashes()
    assert "Cargo.toml" in hashes
    assert all("\\" not in key for key in hashes)
    root = Path(sensitivity.__file__).resolve().parents[3]
    assert ("Cargo.lock" in hashes) == (root / "Cargo.lock").is_file()


def test_matched_snapshot_keys_are_posix():
    try:
        snapshot = matched_export.matched_source_snapshot()
    except RuntimeError as exc:  # native kernel absent on this host
        pytest.skip(str(exc))
    assert all("\\" not in key for key in snapshot["source_sha256"])


# --- Design by contract on the Monte Carlo inputs -------------------------


@pytest.mark.parametrize("field", ["n_shots", "seed"])
def test_boolean_integers_are_rejected(field):
    with pytest.raises(ValueError):
        core.AnalysisConfig(**{field: True})


def test_boolean_floats_are_rejected():
    with pytest.raises(ValueError):
        core.AnalysisConfig(face_sd_deg=True)


def test_reachable_face_range_brackets_zero():
    low, high = reachable_face_range_deg(
        base_loft_deg=10.9, lie_deg=58.0, shaft_lean_deg=0.0
    )
    assert -15.0 <= low < 0.0 < high <= 15.0


def test_unreachable_shaft_rotation_draws_fail_before_any_simulation():
    calls = []

    class CountingPhysics:
        def simulate(self, **kwargs):
            calls.append(kwargs)
            raise AssertionError("physics must not run for an unreachable draw")

    config = core.AnalysisConfig(
        n_shots=10_000, face_sd_deg=5.0, delivery_mode="shaft_rotation"
    )
    with pytest.raises(ValueError, match="shaft-twist branch"):
        core.run_analysis(config, physics=CountingPhysics())
    assert calls == []


# --- Summary statistics are exposed and independently checked -------------


def test_wilson_interval_matches_reference_values():
    low, high = core.wilson_interval(0.5, 100)
    assert low == pytest.approx(0.40383, abs=1e-5)
    assert high == pytest.approx(0.59617, abs=1e-5)
    low, high = core.wilson_interval(0.0, 50)
    assert low == pytest.approx(0.0, abs=1e-12)
    assert 0.0 < high < 0.08


@pytest.mark.parametrize("fraction,n", [(-0.1, 10), (1.1, 10), (0.5, 0), (math.nan, 5)])
def test_wilson_interval_contract(fraction, n):
    with pytest.raises(ValueError):
        core.wilson_interval(fraction, n)


def _record(pattern, face, path):
    return core.ShotRecord(pattern, 0, face, path, 0, 0, 0, 1, 0, 1, 1, 0)


def test_opposite_curve_fraction_counts_shots_that_curve_the_wrong_way():
    draws = [_record("Draw", 1.5, 3.0), _record("Draw", 3.5, 3.0)]
    fades = [_record("Fade", -1.5, -3.0), _record("Fade", -3.5, -3.0)]
    assert core.opposite_curve_fraction(draws) == 0.5
    assert core.opposite_curve_fraction(fades) == 0.5
    assert core.opposite_curve_fraction([_record("Straight", 4.0, 0.0)]) == 0.0


# --- Saved-comparison contracts --------------------------------------------


def _summary(**config):
    base = {
        "seed": 1,
        "n_shots": 2_000,
        "club_id": "driver",
        "face_sd_deg": 1.0,
        "curve_scale": 1.0,
        "delivery_mode": "fixed_loft",
        "target_radius_m": 12.5,
    }
    base.update(config)
    return {
        "config": base,
        "target_x_m": 200.0,
        "patterns": {"Straight": {}, "Draw": {}, "Fade": {}},
    }


def test_comparison_rejects_mixed_delivery_modes():
    with pytest.raises(ValueError, match="delivery_mode"):
        comparison.validate_comparison(
            [_summary(), _summary(delivery_mode="shaft_rotation")]
        )


def test_comparison_headings_follow_the_saved_configuration():
    hit_title, subtitle = comparison.comparison_headings(_summary())
    assert hit_title == "Landings Within 12.5 m"
    assert subtitle.startswith("2,000 Shots per Pattern")


# --- Putting model is only used where it was fitted ------------------------


def test_putting_model_rejects_extrapolation_beyond_the_fitted_table():
    scoring.putt_probabilities(scoring.MAX_PUTT_DISTANCE_YARDS)
    with pytest.raises(ValueError, match="fitted"):
        scoring.putt_probabilities(scoring.MAX_PUTT_DISTANCE_YARDS + 0.01)


def test_unknown_pattern_in_saved_shots_is_a_value_error(tmp_path):
    shots = tmp_path / "shots.csv"
    shots.write_text("pattern,shot_index,aimed_x_m,aimed_y_m\nHook,0,1.0,0.0\n")
    with pytest.raises(ValueError, match="Hook"):
        scoring.load_pattern_endpoints(shots)


# --- Acceptance evidence cannot escape its receipt directory ---------------


@pytest.mark.parametrize("name", ["../secret.json", "/etc/passwd", "a/../../b"])
def test_acceptance_evidence_names_must_stay_inside_receipt_dir(tmp_path, name):
    with pytest.raises(ValueError, match="outside"):
        matched_matrix.resolve_evidence_file(tmp_path / "receipt.json", name)


def test_acceptance_evidence_inside_receipt_dir_resolves(tmp_path):
    resolved = matched_matrix.resolve_evidence_file(
        tmp_path / "receipt.json", "review/notes.md"
    )
    assert resolved == (tmp_path / "review" / "notes.md").resolve()


# --- Matched runner reports completion even when the last draw is rejected -


def test_progress_reaches_total_when_final_draws_are_out_of_range():
    n_shots = 8
    request = ExperimentConfig(
        matched_preset("seven_iron"),
        NoiseSpec((0, 0, 1.0e4, 0, 0, 0, 0)),
        n_shots=n_shots,
    )
    calls = []
    result = run_matched_analysis(
        request,
        physics=AnalyticPhysics(),
        include_scoring=False,
        progress=lambda done, total: calls.append((done, total)),
    )
    statuses = [shot.status for shot in result.shots]
    assert "delivery_out_of_range" in statuses
    assert calls[-1] == (3 * n_shots, 3 * n_shots)


# --- Atomic export into a pre-created empty directory ----------------------


def test_export_into_existing_empty_directory_publishes(tmp_path, monkeypatch):
    value = {"source_sha256": {"fake.py": "abc"}, "native_binary_sha256": "def"}
    monkeypatch.setattr(matched_export, "matched_source_snapshot", lambda: dict(value))
    destination = tmp_path / "result"
    destination.mkdir()
    request = ExperimentConfig(
        matched_preset("seven_iron"), NoiseSpec((1, 0, 0, 0.00635, 0, 0, 0)), n_shots=6
    )
    monkeypatch.setattr(
        matched_export,
        "run_matched_analysis",
        lambda req, **kw: run_matched_analysis(
            req, physics=AnalyticPhysics(), include_scoring=False
        ),
    )
    matched_export.execute_and_export(request, destination, figures=False)
    assert provenance.manifest_files_match(destination / "manifest.json")
