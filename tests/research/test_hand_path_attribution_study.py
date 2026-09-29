"""End-to-end contracts for deterministic hand-path attribution evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.run_hand_path_attribution_study import (
    FIGURE_STEMS,
    build_study,
    write_study,
)

pytestmark = pytest.mark.scientific


@pytest.fixture(scope="module")
def study() -> tuple[dict[str, object], dict[str, np.ndarray]]:
    return build_study()


def test_study_is_deterministic_and_declares_prescribed_kinematics(
    study: tuple[dict[str, object], dict[str, np.ndarray]],
) -> None:
    record, arrays = study
    second_record, second_arrays = build_study()
    assert json.dumps(record, sort_keys=True) == json.dumps(
        second_record, sort_keys=True
    )
    assert arrays.keys() == second_arrays.keys()
    for name in arrays:
        np.testing.assert_array_equal(arrays[name], second_arrays[name])
    assert record["schema_version"] == "hand-path-attribution-evidence-v1"
    models = record["models"]
    assert set(models) == {"double_pendulum", "one_arm", "two_arm"}
    assert models["two_arm"]["trajectory_kind"] == (
        "prescribed_constraint_consistent_kinematics"
    )
    assert models["one_arm"]["trajectory_kind"] == (
        "forward_simulation_on_declared_fixed_time_window"
    )
    assert models["double_pendulum"]["trajectory_kind"] == (
        "forward_simulation_truncated_at_first_valid_impact"
    )
    assert models["double_pendulum"]["terminal_event"]["name"] == (
        "first_valid_club_vertical_impact"
    )
    assert models["double_pendulum"]["terminal_event"]["time_s"] < 0.9
    assert models["double_pendulum"]["primary_estimand"]["path_length_m"] < 5.0
    for model in models.values():
        assert [phase["phase_name"] for phase in model["phase_summaries"]] == [
            "Normalized Time Quartile 1",
            "Normalized Time Quartile 2",
            "Normalized Time Quartile 3",
            "Normalized Time Quartile 4",
        ]


def test_control_preserved_evaluation_is_explicit_and_separate_from_control(
    study: tuple[dict[str, object], dict[str, np.ndarray]],
) -> None:
    record, arrays = study
    for model_name, model in record["models"].items():
        diagnostic = model["zero_velocity_control_preserved"]
        assert diagnostic["status"] == "available"
        assert "same configuration" in diagnostic["protocol"].lower()
        assert "zero generalized velocity" in diagnostic["protocol"].lower()
        assert "not canonical zvcf" in diagnostic["interpretation"].lower()
        assert f"{model_name}__force_zvcf" in arrays
        assert not np.array_equal(
            arrays[f"{model_name}__force_zvcf"],
            arrays[f"{model_name}__force_control"],
        )


def test_closure_phase_coverage_and_provenance_are_fail_closed(
    study: tuple[dict[str, object], dict[str, np.ndarray]],
) -> None:
    record, _ = study
    provenance = record["provenance"]
    assert provenance["generator"].endswith("run_hand_path_attribution_study.py")
    assert provenance["source_files"]
    for model in record["models"].values():
        assert model["phase_coverage"]["exhaustive"] is True
        assert model["phase_coverage"]["overlap_count"] == 0
        closure = model["closure"]
        assert closure["force_max_abs"] < 1e-8
        assert closure["couple_max_abs"] < 1e-8
        assert closure["power_max_abs"] < 1e-8
        assert closure["work_max_abs"] < 1e-8
        assert closure["phase_additivity_max_abs"] < 1e-8
        assert len(model["phase_summaries"]) == 4


def test_instantaneous_power_shares_mask_zero_denominators_and_flag_cancellation(
    study: tuple[dict[str, object], dict[str, np.ndarray]],
) -> None:
    record, arrays = study
    for key, model in record["models"].items():
        metadata = model["instantaneous_power_share"]
        assert metadata["undefined_samples_are_masked"] is True
        share = arrays[f"{key}__drift_power_magnitude_share"]
        valid = arrays[f"{key}__power_share_valid"]
        cancellation = arrays[f"{key}__power_cancellation_index"]
        flags = arrays[f"{key}__power_cancellation_flag"]
        assert np.all(np.isnan(share[~valid]))
        assert np.all((share[valid] >= 0.0) & (share[valid] <= 1.0))
        np.testing.assert_array_equal(flags, valid & (cancellation >= 0.25))
        for split in ("total", "drift", "control"):
            for measure in ("signed", "positive", "negative", "absolute"):
                assert f"{key}__tangent_impulse_{split}_{measure}" in arrays


def test_two_hand_common_and_differential_modes_are_exported(
    study: tuple[dict[str, object], dict[str, np.ndarray]],
) -> None:
    record, arrays = study
    assert record["models"]["two_arm"]["common_differential"]["convention"] == (
        "common=right+left; differential=(right-left)/2"
    )
    for split in ("total", "drift", "control", "zvcf"):
        assert arrays[f"two_arm__common_{split}"].shape[1] == 2
        assert arrays[f"two_arm__differential_{split}"].shape[1] == 2

    primary = record["models"]["two_arm"]["primary_estimand"]
    assert primary["name"] == "net_golfer_on_club_force_work_per_hand_path_length"
    assert "mid_grip" in primary["reference_point"]
    for split in ("total", "drift", "control", "zvcf"):
        assert np.isfinite(primary["mean_force_n"][split])
        assert arrays[f"two_arm__primary_force_{split}"].shape[1] == 2
    assert primary["mean_force_n"]["total"] == pytest.approx(
        primary["mean_force_n"]["drift"] + primary["mean_force_n"]["control"]
    )
    assert primary["final_half_drift_diagnostic"]["conclusion"] in {
        "present_in_declared_case",
        "not_demonstrated_by_declared_case",
    }


@pytest.mark.timeout(240)
def test_writer_emits_machine_readable_evidence_and_vector_figures(
    tmp_path: Path,
) -> None:
    paths = write_study(tmp_path)
    assert paths["json"].is_file()
    assert paths["npz"].is_file()
    loaded = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert loaded["schema_version"] == "hand-path-attribution-evidence-v1"
    with np.load(paths["npz"]) as arrays:
        assert "two_arm__common_zvcf" in arrays
    for stem in FIGURE_STEMS:
        svg = tmp_path / "figures" / f"{stem}.svg"
        pdf = tmp_path / "figures" / f"{stem}.pdf"
        assert svg.stat().st_size > 1_000
        assert pdf.stat().st_size > 1_000
        assert "<svg" in svg.read_text(encoding="utf-8")[:1_000]

    repeated_root = tmp_path / "repeated"
    write_study(repeated_root)
    for stem in FIGURE_STEMS:
        for suffix in ("svg", "pdf"):
            first = tmp_path / "figures" / f"{stem}.{suffix}"
            repeated = repeated_root / "figures" / f"{stem}.{suffix}"
            assert first.read_bytes() == repeated.read_bytes()
