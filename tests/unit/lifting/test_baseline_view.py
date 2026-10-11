"""Tests for the LIFT-1 baseline receipt loader/view (LIFT-8, #11748)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from src.shared.python.lifting.baseline_view import (
    DEFAULT_BASELINE_PATH,
    available_lifts,
    lift_view,
    load_baseline,
)
from src.shared.python.lifting.pack_audit.baseline import SCHEMA
from src.shared.python.lifting.pack_audit.names import ENGINES, LIFTS

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Synthetic receipt fixture (NaN/missing-value and gap-scoping coverage that
# the committed 430 KB receipt does not conveniently exercise).
# ---------------------------------------------------------------------------


def _engine_result(**overrides: Any) -> dict[str, Any]:
    base = {
        "structure": {"n_bodies": 10, "nq": 20, "nv": 18},
        "segment_masses_kg": {},
        "start": {
            "total_mass_kg": 100.0,
            "bar_above_sole_m": 1.0,
            "hand_mid_above_sole_m": 0.9,
        },
        "start_contact": {
            "value_n": 0.0,
            "non_ground_normal_force_n": 5.0,
            "n_ground_contacts": 1,
            "n_non_ground_contacts": 0,
            "reason": None,
        },
        "smoke": {"loaded": True, "stepped": True, "max_abs_qvel": 0.1},
        "same_q": {},
        "reference_fk": [],
        "phases": [
            {
                "name": "start",
                "fraction": 0.0,
                "n_targets": 2,
                "summary": {
                    "hand_bar": {
                        "l": {"axis_distance_m": 0.3},
                        "r": {"axis_distance_m": 0.3},
                    }
                },
            }
        ],
    }
    base.update(overrides)
    return base


@pytest.fixture
def synthetic_receipt() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "generated_utc": "2026-01-01T00:00:00+00:00",
        "anthropometry": {"body_mass_kg": 80.0},
        "tolerances": {"position_m": 0.02, "mass_rel": 1e-6},
        "packs": {
            "mujoco": {"repo": "MuJoCo_Models", "commit": "abc123", "licence": "MIT"},
            "opensim": {"repo": "OpenSim_Models", "commit": "def456", "licence": "MIT"},
        },
        "results": {
            "squat": {
                "mujoco": _engine_result(),
                "opensim": _engine_result(
                    start={
                        "total_mass_kg": None,
                        "bar_above_sole_m": math.nan,
                        "hand_mid_above_sole_m": 0.91,
                    }
                ),
            },
            "deadlift": {
                "mujoco": _engine_result(),
            },
        },
        "comparisons": {
            "squat": {
                "poses": {
                    "zero": {
                        "mujoco|opensim": {
                            "segments_max_m": 0.01,
                            "hands_max_m": 0.03,
                            "feet_max_m": None,
                            "bar_centre_max_m": math.nan,
                            "com_max_m": 0.0,
                            "lifter_com_max_m": 0.019999,
                        }
                    }
                },
                "mass": {},
            }
        },
        "gaps": [
            {
                "key": "grip_attachment",
                "title": "Right hand is not attached to the bar",
                "engines": ["mujoco"],
                "evidence": ["mujoco: right-hand attachment is not a bar weld"],
                "issues": ["MuJoCo_Models#1"],
                "lift_story": "LIFT-3",
                "new_issue": False,
            },
            {
                "key": "bench_mass",
                "title": "Bench mass convention differs across packs",
                "engines": ["mujoco", "opensim"],
                "evidence": ["mujoco: 1.0 kg"],
                "issues": ["MuJoCo_Models#2"],
                "lift_story": "LIFT-2",
                "new_issue": False,
            },
            {
                "key": "inertia",
                "title": "Segment inertia differs from the other packs",
                "engines": ["mujoco"],
                "evidence": ["mujoco/torso: median principal moment 1 vs 2"],
                "issues": ["MuJoCo_Models#3"],
                "lift_story": "LIFT-2",
                "new_issue": False,
            },
            {
                "key": "no_grf",
                "title": "No ground-reaction output available",
                "engines": ["mujoco", "opensim"],
                "evidence": ["mujoco: no contact model"],
                "issues": ["MuJoCo_Models#4"],
                "lift_story": "LIFT-9",
                "new_issue": False,
            },
            {
                "key": "unscoped_opensim_only",
                "title": "An unrecognised gap key naming only opensim",
                "engines": ["opensim"],
                "evidence": [],
                "issues": [],
                "lift_story": "LIFT-9",
                "new_issue": True,
            },
        ],
        "deferred": [],
    }


# ---------------------------------------------------------------------------
# load_baseline
# ---------------------------------------------------------------------------


def test_load_baseline_rejects_wrong_schema(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({"schema": "not-the-right-schema/v1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_baseline(path)


def test_load_baseline_rejects_missing_schema_key(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({"results": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_baseline(path)


def test_load_baseline_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_baseline(tmp_path / "does_not_exist.json")


def test_load_baseline_invalid_json_raises_value_error(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_baseline(path)


def test_load_baseline_default_path_is_the_committed_receipt() -> None:
    assert DEFAULT_BASELINE_PATH.name == "pack_parity_baseline.json"
    receipt = load_baseline()
    assert receipt["schema"] == SCHEMA


def test_load_baseline_accepts_explicit_path_to_committed_receipt() -> None:
    receipt = load_baseline(DEFAULT_BASELINE_PATH)
    assert receipt["schema"] == SCHEMA


# ---------------------------------------------------------------------------
# available_lifts
# ---------------------------------------------------------------------------


def test_available_lifts_real_receipt_has_all_five_in_canonical_order() -> None:
    receipt = load_baseline()
    assert available_lifts(receipt) == list(LIFTS)


def test_available_lifts_real_receipt_has_four_engines_each() -> None:
    receipt = load_baseline()
    for lift in available_lifts(receipt):
        engines = {e["engine"] for e in lift_view(receipt, lift)["engines"]}
        assert engines == set(ENGINES)


def test_available_lifts_filters_to_whats_present(synthetic_receipt) -> None:
    assert available_lifts(synthetic_receipt) == ["squat", "deadlift"]


def test_available_lifts_rejects_non_receipt() -> None:
    with pytest.raises(TypeError):
        available_lifts({"not": "a receipt"})
    with pytest.raises(TypeError):
        available_lifts("not even a dict")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# lift_view: basic shape and error handling
# ---------------------------------------------------------------------------


def test_lift_view_unknown_lift_raises_value_error_listing_valid_lifts(
    synthetic_receipt,
) -> None:
    with pytest.raises(ValueError, match=r"squat.*deadlift|deadlift.*squat"):
        lift_view(synthetic_receipt, "clean_and_jerk")


def test_lift_view_real_receipt_every_lift_round_trips() -> None:
    receipt = load_baseline()
    for lift in available_lifts(receipt):
        view = lift_view(receipt, lift)
        assert view["lift"] == lift
        assert len(view["engines"]) == 4


# ---------------------------------------------------------------------------
# "Unavailable is never zero"
# ---------------------------------------------------------------------------


def test_missing_total_mass_becomes_none_with_reason(synthetic_receipt) -> None:
    view = lift_view(synthetic_receipt, "squat")
    opensim = next(e for e in view["engines"] if e["engine"] == "opensim")
    field = opensim["total_mass_kg"]
    assert field["value"] is None
    assert field["reason"]


def test_nan_bar_above_sole_becomes_none_with_reason(synthetic_receipt) -> None:
    view = lift_view(synthetic_receipt, "squat")
    opensim = next(e for e in view["engines"] if e["engine"] == "opensim")
    field = opensim["bar_above_sole_m"]
    assert field["value"] is None
    assert field["reason"]


def test_present_finite_value_is_not_treated_as_unavailable(synthetic_receipt) -> None:
    view = lift_view(synthetic_receipt, "squat")
    mujoco = next(e for e in view["engines"] if e["engine"] == "mujoco")
    field = mujoco["total_mass_kg"]
    assert field["value"] == 100.0
    assert field["reason"] is None


def test_real_zero_contact_value_is_not_confused_with_unavailable(
    synthetic_receipt,
) -> None:
    view = lift_view(synthetic_receipt, "squat")
    mujoco = next(e for e in view["engines"] if e["engine"] == "mujoco")
    value_field = mujoco["start_contact"]["value_n"]
    assert value_field["value"] == 0.0
    assert value_field["reason"] is None


# ---------------------------------------------------------------------------
# Cross-engine comparison tolerance flagging
# ---------------------------------------------------------------------------


def test_comparison_pass_fail_unavailable_flags(synthetic_receipt) -> None:
    view = lift_view(synthetic_receipt, "squat")
    pair = view["comparisons"]["poses"]["zero"]["mujoco|opensim"]
    assert pair["segments_max_m"]["status"] == "pass"  # 0.01 <= 0.02
    assert pair["hands_max_m"]["status"] == "fail"  # 0.03 > 0.02
    assert pair["feet_max_m"]["status"] == "unavailable"  # None
    assert pair["bar_centre_max_m"]["status"] == "unavailable"  # NaN
    assert pair["com_max_m"]["status"] == "pass"  # 0.0 <= 0.02, not unavailable
    assert pair["lifter_com_max_m"]["status"] == "pass"  # exactly at tolerance


def test_comparisons_unavailable_reason_when_fewer_than_two_engines(
    synthetic_receipt,
) -> None:
    view = lift_view(synthetic_receipt, "deadlift")
    assert view["comparisons"]["poses"] == {}
    assert view["comparisons"]["reason"]


# ---------------------------------------------------------------------------
# Gap scoping per lift
# ---------------------------------------------------------------------------


def test_gaps_scoped_to_bench_press_only_for_bench_mass(synthetic_receipt) -> None:
    squat_keys = {g["key"] for g in lift_view(synthetic_receipt, "squat")["gaps"]}
    assert "bench_mass" not in squat_keys


def test_gaps_scoped_to_deadlift_only_for_inertia(synthetic_receipt) -> None:
    squat_keys = {g["key"] for g in lift_view(synthetic_receipt, "squat")["gaps"]}
    deadlift_keys = {g["key"] for g in lift_view(synthetic_receipt, "deadlift")["gaps"]}
    assert "inertia" not in squat_keys
    assert "inertia" in deadlift_keys


def test_gap_hidden_when_its_engines_are_unavailable_for_the_lift(
    synthetic_receipt,
) -> None:
    # deadlift's only result is "mujoco"; a gap naming only "opensim" has no
    # overlap with deadlift's available engines and must not surface there,
    # even though its (unrecognised) key defaults to applying to every lift.
    squat_keys = {g["key"] for g in lift_view(synthetic_receipt, "squat")["gaps"]}
    deadlift_keys = {g["key"] for g in lift_view(synthetic_receipt, "deadlift")["gaps"]}
    assert "unscoped_opensim_only" in squat_keys  # squat has opensim
    assert "unscoped_opensim_only" not in deadlift_keys  # deadlift does not


def test_cross_cutting_gap_applies_to_every_lift_with_a_matching_engine(
    synthetic_receipt,
) -> None:
    squat_keys = {g["key"] for g in lift_view(synthetic_receipt, "squat")["gaps"]}
    deadlift_keys = {g["key"] for g in lift_view(synthetic_receipt, "deadlift")["gaps"]}
    assert "no_grf" in squat_keys
    assert "no_grf" in deadlift_keys
