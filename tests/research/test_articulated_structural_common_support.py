"""Contracts for cell-identity-safe structural sensitivity comparisons."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_structural_common_support import (
    build_axis_summary_record,
    build_corner_support_summary,
    build_one_sided_engineering_secants,
    classify_one_sided_engineering_secants,
    compare_common_support,
    corner_support_summary_record,
    extract_headline_cells,
    require_corner_release_evidence,
)

pytestmark = pytest.mark.scientific
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"


def _arrays(
    states: tuple[tuple[int, int], ...],
    *,
    matched_indices: tuple[int, ...],
    speed_offset_m_s: float = 0.0,
    ground_names: bool = False,
) -> dict[str, np.ndarray]:
    shape = (len(states), 2, 2, 2, 4)
    matched = np.zeros(shape, dtype=bool)
    matched.ravel()[list(matched_indices)] = True
    speed = np.arange(matched.size, dtype=float).reshape(shape) / 1000.0
    speed += speed_offset_m_s
    result = {
        "state_case_index": np.asarray([state[0] for state in states]),
        "state_sample_index": np.asarray([state[1] for state in states]),
        "velocity_factors": np.asarray([1.0, -1.0]),
        "time_steps_s": np.asarray([0.00025, 0.000125]),
        "engine_names": np.asarray(["mujoco", "pinocchio"]),
        "horizons_s": np.asarray([0.004, 0.01, 0.025, 0.05]),
        "load_match_relative_error": np.full(shape, 0.01),
        "work_match_relative_error": np.full(shape, 0.02),
    }
    if ground_names:
        result["matched"] = matched
        result["matched_speed_difference"] = speed
    else:
        result["matched_load_work"] = matched
        result["matched_final_speed_difference_m_s"] = speed
    return result


@pytest.mark.parametrize("pathway", ["shaft", "ground"])
def test_extract_headline_cells_normalizes_both_atlas_schemas(pathway: str) -> None:
    cells = extract_headline_cells(
        pathway,
        _arrays(((0, 0),), matched_indices=(0, 7), ground_names=pathway == "ground"),
    )

    assert len(cells.identities) == 32
    assert len(set(cells.identities)) == 32
    assert cells.matched.dtype == np.bool_
    assert np.flatnonzero(cells.matched).tolist() == [0, 7]
    assert cells.identities[0] == (0, 0, 1.0, 0.00025, "mujoco", 0.004)
    assert cells.pathway == pathway
    assert cells.two_engine_speed_difference_discrepancy_m_s[0] == pytest.approx(0.004)
    assert cells.time_step_speed_difference_discrepancy_m_s[0] == pytest.approx(0.008)


def test_extract_headline_cells_rejects_shape_or_identity_drift() -> None:
    arrays = _arrays(((0, 0),), matched_indices=())
    arrays["matched_load_work"] = np.zeros((1, 2, 2, 1, 4), dtype=bool)
    with pytest.raises(ValueError, match="registered headline shape"):
        extract_headline_cells("shaft", arrays)

    arrays = _arrays(((0, 0), (0, 0)), matched_indices=())
    with pytest.raises(ValueError, match="cell identities must be unique"):
        extract_headline_cells("shaft", arrays)


@pytest.mark.parametrize(
    ("pathway", "artifact", "expected_matched"),
    [
        ("shaft", "articulated_shaft_atlas.npz", 126),
        ("ground", "articulated_ground_atlas.npz", 0),
    ],
)
def test_committed_nominal_atlases_reproduce_registered_support(
    pathway: str, artifact: str, expected_matched: int
) -> None:
    with np.load(DATA / artifact) as arrays:
        cells = extract_headline_cells(pathway, arrays)

    assert len(cells.identities) == 384
    assert np.count_nonzero(cells.matched) == expected_matched
    assert np.all(np.isfinite(cells.two_engine_speed_difference_discrepancy_m_s))
    assert np.all(np.isfinite(cells.time_step_speed_difference_discrepancy_m_s))
    self_comparison = compare_common_support(cells, cells)
    assert len(self_comparison.persistent_identities) == expected_matched
    assert np.all(self_comparison.persistent_speed_change_m_s == 0.0)
    assert not np.any(self_comparison.resolved_outcome_change)
    if pathway == "ground":
        assert self_comparison.has_paired_outcome is False


def test_common_support_excludes_missing_states_and_preserves_transitions() -> None:
    nominal = extract_headline_cells(
        "shaft",
        _arrays(((0, 0), (0, 6)), matched_indices=(0, 1, 32)),
    )
    corner = extract_headline_cells(
        "shaft",
        _arrays(((0, 0),), matched_indices=(0, 2), speed_offset_m_s=0.004),
    )

    comparison = compare_common_support(nominal, corner)

    assert comparison.common_executed_cell_count == 32
    assert comparison.nominal_only_executed_cell_count == 32
    assert comparison.corner_only_executed_cell_count == 0
    assert comparison.nominal_only_identities == nominal.identities[32:]
    assert comparison.corner_only_identities == ()
    assert comparison.persistent_identities == (nominal.identities[0],)
    assert comparison.exited_identities == (nominal.identities[1],)
    assert comparison.entered_identities == (corner.identities[2],)
    assert comparison.persistent_speed_change_m_s.tolist() == pytest.approx([0.004])
    assert comparison.resolution_threshold_m_s.tolist() == pytest.approx([0.008])
    assert comparison.resolved_outcome_change.tolist() == [False]


def test_common_support_preserves_corner_only_execution_identities() -> None:
    nominal = extract_headline_cells("shaft", _arrays(((0, 0),), matched_indices=()))
    corner = extract_headline_cells(
        "shaft", _arrays(((0, 0), (0, 6)), matched_indices=())
    )

    comparison = compare_common_support(nominal, corner)

    assert comparison.nominal_only_identities == ()
    assert comparison.corner_only_identities == corner.identities[32:]
    assert comparison.corner_only_executed_cell_count == 32


def test_zero_nominal_ground_support_cannot_produce_a_paired_benefit() -> None:
    nominal = extract_headline_cells(
        "ground", _arrays(((0, 0),), matched_indices=(), ground_names=True)
    )
    corner = extract_headline_cells(
        "ground", _arrays(((0, 0),), matched_indices=(0,), ground_names=True)
    )

    comparison = compare_common_support(nominal, corner)

    assert len(comparison.entered_identities) == 1
    assert comparison.persistent_identities == ()
    assert comparison.persistent_speed_change_m_s.size == 0
    assert comparison.resolution_threshold_m_s.size == 0
    assert comparison.resolved_outcome_change.size == 0
    assert comparison.has_paired_outcome is False


def test_resolution_uses_floor_and_both_corner_numerical_discrepancies() -> None:
    nominal_arrays = _arrays(((0, 0),), matched_indices=(0,))
    corner_arrays = _arrays(((0, 0),), matched_indices=(0,))
    nominal_arrays["matched_final_speed_difference_m_s"].fill(0.0)
    corner_arrays["matched_final_speed_difference_m_s"].fill(0.004)
    resolved = compare_common_support(
        extract_headline_cells("shaft", nominal_arrays),
        extract_headline_cells("shaft", corner_arrays),
        absolute_resolution_floor_m_s=0.001,
    )
    assert resolved.resolution_threshold_m_s.tolist() == pytest.approx([0.001])
    assert resolved.resolved_outcome_change.tolist() == [True]

    corner_arrays["matched_final_speed_difference_m_s"].ravel()[4] = 0.009
    unresolved = compare_common_support(
        extract_headline_cells("shaft", nominal_arrays),
        extract_headline_cells("shaft", corner_arrays),
        absolute_resolution_floor_m_s=0.001,
    )
    assert unresolved.resolution_threshold_m_s.tolist() == pytest.approx([0.005])
    assert unresolved.resolved_outcome_change.tolist() == [False]


@pytest.mark.parametrize("floor", [-0.001, float("nan"), float("inf")])
def test_resolution_floor_must_be_finite_and_nonnegative(floor: float) -> None:
    cells = extract_headline_cells("shaft", _arrays(((0, 0),), matched_indices=()))
    with pytest.raises(ValueError, match="resolution floor"):
        compare_common_support(
            cells,
            cells,
            absolute_resolution_floor_m_s=floor,
        )


def test_one_sided_secants_preserve_direction_and_separate_support() -> None:
    nominal_arrays = _arrays(((0, 0),), matched_indices=(0, 1, 2))
    low_arrays = _arrays(((0, 0),), matched_indices=(0, 1))
    high_arrays = _arrays(((0, 0),), matched_indices=(0, 2))
    for arrays in (nominal_arrays, low_arrays, high_arrays):
        arrays["matched_final_speed_difference_m_s"].fill(0.0)
    low_arrays["matched_final_speed_difference_m_s"].ravel()[0:2] = -0.006
    high_arrays["matched_final_speed_difference_m_s"].ravel()[[0, 2]] = 0.004

    nominal = extract_headline_cells("shaft", nominal_arrays)
    low = compare_common_support(nominal, extract_headline_cells("shaft", low_arrays))
    high = compare_common_support(nominal, extract_headline_cells("shaft", high_arrays))
    secants = build_one_sided_engineering_secants(
        "height_scale",
        low,
        high,
        low_scale=0.8,
        nominal_scale=1.0,
        high_scale=1.4,
    )

    assert secants.low_to_nominal_identities == low.persistent_identities
    assert secants.nominal_to_high_identities == high.persistent_identities
    assert secants.low_to_nominal_m_s_per_unit_scale.tolist() == pytest.approx(
        [0.03, 0.03]
    )
    assert secants.nominal_to_high_m_s_per_unit_scale.tolist() == pytest.approx(
        [0.01, 0.01]
    )
    assert (
        secants.low_to_nominal_identities[1] != (secants.nominal_to_high_identities[1])
    )
    assert secants.are_averaged is False


@pytest.mark.parametrize(
    ("low", "nominal", "high"),
    [(1.0, 1.0, 1.2), (0.8, 1.1, 1.0), (float("nan"), 1.0, 1.2)],
)
def test_one_sided_secants_require_ordered_finite_scales(
    low: float, nominal: float, high: float
) -> None:
    cells = extract_headline_cells("shaft", _arrays(((0, 0),), matched_indices=()))
    comparison = compare_common_support(cells, cells)
    with pytest.raises(ValueError, match="low < nominal < high"):
        build_one_sided_engineering_secants(
            "height_scale",
            comparison,
            comparison,
            low_scale=low,
            nominal_scale=nominal,
            high_scale=high,
        )


def _classified_secants(
    low_change_m_s: float,
    high_change_m_s: float,
    *,
    low_match: int = 0,
    high_match: int = 0,
):
    nominal_arrays = _arrays(((0, 0),), matched_indices=(0, 1))
    low_arrays = _arrays(((0, 0),), matched_indices=(low_match,))
    high_arrays = _arrays(((0, 0),), matched_indices=(high_match,))
    for arrays in (nominal_arrays, low_arrays, high_arrays):
        arrays["matched_final_speed_difference_m_s"].fill(0.0)
    low_arrays["matched_final_speed_difference_m_s"].fill(low_change_m_s)
    high_arrays["matched_final_speed_difference_m_s"].fill(high_change_m_s)
    nominal = extract_headline_cells("shaft", nominal_arrays)
    low = compare_common_support(nominal, extract_headline_cells("shaft", low_arrays))
    high = compare_common_support(nominal, extract_headline_cells("shaft", high_arrays))
    return classify_one_sided_engineering_secants(
        build_one_sided_engineering_secants(
            "height_scale",
            low,
            high,
            low_scale=0.8,
            nominal_scale=1.0,
            high_scale=1.2,
        )
    )


@pytest.mark.parametrize(
    ("low_change", "high_change", "expected"),
    [
        (-0.004, -0.004, "resolved_opposing_on_shared_support"),
        (-0.004, 0.0012, "resolved_materially_unequal_on_shared_support"),
        (-0.004, 0.004, "resolved_direction_consistent_on_shared_support"),
        (-0.004, 0.0005, "resolution_limited_on_shared_support"),
    ],
)
def test_nonmonotonicity_classification_is_resolution_aware(
    low_change: float, high_change: float, expected: str
) -> None:
    classification = _classified_secants(low_change, high_change)
    assert classification.overall == expected
    assert len(classification.shared_persistent_identities) == 1
    assert len(classification.cell_classification) == 1


def test_nonmonotonicity_requires_shared_persistent_support() -> None:
    classification = _classified_secants(-0.004, 0.004, low_match=0, high_match=1)
    assert classification.overall == "insufficient_shared_persistent_support"
    assert classification.shared_persistent_identities == ()
    assert classification.cell_classification == ()


def _authority_record() -> dict[str, object]:
    return {
        "authority_sha256": "a" * 64,
        "scales": {"height": 1.0, "body_mass": 1.0, "joint_limit": 1.0},
        "model_sha256": {"0": "b" * 64},
    }


def test_corner_summary_preserves_all_support_denominators() -> None:
    cells = extract_headline_cells(
        "shaft",
        _arrays(
            tuple((0, phase) for phase in range(12)), matched_indices=tuple(range(126))
        ),
    )
    summary = build_corner_support_summary(
        "nominal",
        cells,
        requested_state_count=12,
        feasible_state_count=12,
        retained_failures=(),
        planned_headline_cell_count=384,
        all_registered_gates_passed=True,
        authority=_authority_record(),
    )

    assert summary.planned_headline_cell_count == 384
    assert summary.feasible_headline_cell_count == 384
    assert summary.executed_headline_cell_count == 384
    assert summary.matched_cell_count == 126
    assert summary.matched_fraction_of_feasible == pytest.approx(126 / 384)
    assert summary.complete_execution is True
    assert summary.qualifies_as_release_evidence is True
    require_corner_release_evidence(summary)


def test_corner_summary_retains_infeasible_state_outside_denominator() -> None:
    states = tuple((case, phase) for case in (0, 8, 9, 17) for phase in (0, 6, 12))
    states = tuple(state for state in states if state != (0, 12))
    cells = extract_headline_cells("shaft", _arrays(states, matched_indices=()))
    failure = {
        "case_index": 0,
        "phase_index": 12,
        "failure_class": "ik_nonconvergence",
    }
    summary = build_corner_support_summary(
        "height_scale-low",
        cells,
        requested_state_count=12,
        feasible_state_count=11,
        retained_failures=(failure,),
        planned_headline_cell_count=384,
        all_registered_gates_passed=True,
        authority=_authority_record(),
    )

    assert summary.retained_failures == (failure,)
    assert summary.feasible_headline_cell_count == 352
    assert summary.executed_headline_cell_count == 352
    assert summary.matched_fraction_of_feasible == 0.0
    assert summary.qualifies_as_release_evidence is True


def test_partial_corner_cannot_qualify_as_release_evidence() -> None:
    cells = extract_headline_cells(
        "shaft", _arrays(tuple((0, phase) for phase in range(11)), matched_indices=())
    )
    summary = build_corner_support_summary(
        "nominal",
        cells,
        requested_state_count=12,
        feasible_state_count=12,
        retained_failures=(),
        planned_headline_cell_count=384,
        all_registered_gates_passed=True,
        authority=_authority_record(),
    )

    assert summary.complete_execution is False
    assert summary.qualifies_as_release_evidence is False
    with pytest.raises(RuntimeError, match="does not qualify"):
        require_corner_release_evidence(summary)


def test_corner_summary_rejects_erased_failure_or_bad_plan_denominator() -> None:
    cells = extract_headline_cells(
        "ground", _arrays(((0, 0),), matched_indices=(), ground_names=True)
    )
    with pytest.raises(ValueError, match="retained failure states"):
        build_corner_support_summary(
            "height_scale-low",
            cells,
            requested_state_count=2,
            feasible_state_count=1,
            retained_failures=(),
            planned_headline_cell_count=64,
            all_registered_gates_passed=True,
            authority=_authority_record(),
        )
    with pytest.raises(ValueError, match="planned headline denominator"):
        build_corner_support_summary(
            "nominal",
            cells,
            requested_state_count=1,
            feasible_state_count=1,
            retained_failures=(),
            planned_headline_cell_count=31,
            all_registered_gates_passed=True,
            authority=_authority_record(),
        )


def test_corner_release_record_has_registered_machine_readable_fields() -> None:
    cells = extract_headline_cells(
        "ground", _arrays(((0, 0),), matched_indices=(), ground_names=True)
    )
    summary = build_corner_support_summary(
        "nominal",
        cells,
        requested_state_count=1,
        feasible_state_count=1,
        retained_failures=(),
        planned_headline_cell_count=32,
        all_registered_gates_passed=True,
        authority=_authority_record(),
    )

    record = corner_support_summary_record(summary)

    assert record["corner_id"] == "nominal"
    assert record["pathway"] == "ground"
    assert record["planned_headline_cell_count"] == 32
    assert record["feasible_headline_cell_count"] == 32
    assert record["executed_headline_cell_count"] == 32
    assert record["matched_cell_count"] == 0
    assert record["matched_fraction_of_feasible"] == 0.0
    assert record["all_registered_gates_passed"] is True
    assert record["authority"]["authority_sha256"] == "a" * 64


def test_corner_summary_rejects_executed_retained_failure_state() -> None:
    cells = extract_headline_cells(
        "shaft", _arrays(((0, 0),), matched_indices=tuple(range(32)))
    )
    with pytest.raises(ValueError, match="cannot also be executed"):
        build_corner_support_summary(
            "overlap",
            cells,
            requested_state_count=2,
            feasible_state_count=1,
            retained_failures=(
                {"case_index": 0, "phase_index": 0, "failure_class": "failed"},
            ),
            planned_headline_cell_count=64,
            all_registered_gates_passed=True,
            authority=_authority_record(),
        )


def _axis_inputs(*, shared_support: bool = True):
    nominal_arrays = _arrays(((0, 0),), matched_indices=(0, 1))
    low_arrays = _arrays(((0, 0),), matched_indices=(0, 1))
    high_indices = (0, 1) if shared_support else (2, 3)
    high_arrays = _arrays(((0, 0),), matched_indices=high_indices)
    nominal_arrays["matched_final_speed_difference_m_s"].fill(0.0)
    low_arrays["matched_final_speed_difference_m_s"].fill(-0.004)
    high_arrays["matched_final_speed_difference_m_s"].fill(0.004)
    nominal = extract_headline_cells("shaft", nominal_arrays)
    low = compare_common_support(nominal, extract_headline_cells("shaft", low_arrays))
    high = compare_common_support(nominal, extract_headline_cells("shaft", high_arrays))
    secants = build_one_sided_engineering_secants(
        "height_scale",
        low,
        high,
        low_scale=0.8,
        nominal_scale=1.0,
        high_scale=1.2,
    )
    return secants, classify_one_sided_engineering_secants(secants)


def test_axis_summary_uses_declared_unweighted_shared_support_median() -> None:
    secants, classification = _axis_inputs()

    record = build_axis_summary_record(secants, classification)

    assert record["axis_name"] == "height_scale"
    assert record["low_scale"] == 0.8
    assert record["nominal_scale"] == 1.0
    assert record["high_scale"] == 1.2
    assert record["shared_persistent_cell_count"] == 2
    assert record["summary_statistic"] == (
        "unweighted median on identities persistent in both one-sided comparisons"
    )
    assert record["low_to_nominal_secant_m_s_per_unit_scale"] == pytest.approx(0.02)
    assert record["nominal_to_high_secant_m_s_per_unit_scale"] == pytest.approx(0.02)
    assert record["low_to_nominal_secant_range_m_s_per_unit_scale"] == pytest.approx(
        [0.02, 0.02]
    )
    assert record["nonmonotonic_classification"] == (
        "resolved_direction_consistent_on_shared_support"
    )


def test_axis_summary_with_no_shared_support_emits_null_not_a_pooled_value() -> None:
    secants, classification = _axis_inputs(shared_support=False)

    record = build_axis_summary_record(secants, classification)

    assert record["shared_persistent_cell_count"] == 0
    assert record["low_to_nominal_secant_m_s_per_unit_scale"] is None
    assert record["nominal_to_high_secant_m_s_per_unit_scale"] is None
    assert record["low_to_nominal_secant_range_m_s_per_unit_scale"] is None
    assert record["nonmonotonic_classification"] == (
        "insufficient_shared_persistent_support"
    )
