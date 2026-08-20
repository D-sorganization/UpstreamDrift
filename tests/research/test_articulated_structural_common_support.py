"""Contracts for cell-identity-safe structural sensitivity comparisons."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_structural_common_support import (
    build_one_sided_engineering_secants,
    compare_common_support,
    extract_headline_cells,
)

pytestmark = pytest.mark.scientific


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
    assert comparison.persistent_identities == (nominal.identities[0],)
    assert comparison.exited_identities == (nominal.identities[1],)
    assert comparison.entered_identities == (corner.identities[2],)
    assert comparison.persistent_speed_change_m_s.tolist() == pytest.approx([0.004])
    assert comparison.resolution_threshold_m_s.tolist() == pytest.approx([0.008])
    assert comparison.resolved_outcome_change.tolist() == [False]


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
