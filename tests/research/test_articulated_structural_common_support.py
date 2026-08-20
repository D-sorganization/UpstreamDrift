"""Contracts for cell-identity-safe structural sensitivity comparisons."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_structural_common_support import (
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
    assert comparison.has_paired_outcome is False
