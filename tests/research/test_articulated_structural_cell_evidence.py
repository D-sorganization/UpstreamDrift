"""Contracts for atomic, digest-bound structural per-cell evidence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_structural_cell_evidence import (
    build_structural_cell_evidence,
    build_structural_cell_evidence_from_atlas,
    load_structural_cell_evidence,
    validate_structural_cell_evidence,
    write_structural_cell_evidence,
)
from scripts.research.proximal_distal_energy.articulated_structural_common_support import (
    HeadlineCells,
    compare_common_support,
)

pytestmark = pytest.mark.scientific
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"


def _cells(matched: tuple[bool, ...], speed: tuple[float, ...]) -> HeadlineCells:
    size = len(matched)
    identities = tuple(
        (0, 0, 1.0, 0.00025, "mujoco", 0.004 + index * 0.001) for index in range(size)
    )
    return HeadlineCells(
        pathway="shaft",
        identities=identities,
        matched=np.asarray(matched, dtype=bool),
        final_speed_difference_m_s=np.asarray(speed, dtype=float),
        load_match_relative_error=np.full(size, 0.01),
        work_match_relative_error=np.full(size, 0.02),
        two_engine_speed_difference_discrepancy_m_s=np.zeros(size),
        time_step_speed_difference_discrepancy_m_s=np.zeros(size),
    )


def test_cell_evidence_preserves_required_fields_and_support_status() -> None:
    nominal = _cells((True, True, False), (0.0, 0.0, 0.0))
    corner = _cells((True, False, True), (0.004, 0.0, 0.003))
    comparison = compare_common_support(nominal, corner)

    evidence = build_structural_cell_evidence(
        corner,
        gate_status=np.asarray([True, False, True]),
        failure_class=np.asarray(["none", "numerical_gate_failure", "none"]),
        comparison=comparison,
    )

    assert {
        "cell_identity",
        "matched_load_work",
        "matched_final_speed_difference_m_s",
        "load_match_relative_error",
        "work_match_relative_error",
        "gate_status",
        "failure_class",
        "two_engine_speed_difference_discrepancy_m_s",
        "time_step_speed_difference_discrepancy_m_s",
        "resolution_threshold_m_s",
        "resolved_outcome_change",
        "comparison_status",
        "evidence_sha256",
    } <= set(evidence)
    assert evidence["comparison_status"].tolist() == [
        "persistent_resolved",
        "exited_support",
        "entered_support",
    ]
    assert evidence["resolution_threshold_m_s"][0] == pytest.approx(0.001)
    assert np.isnan(evidence["resolution_threshold_m_s"][1:]).all()
    assert evidence["resolved_outcome_change"].tolist() == [True, False, False]
    assert len(str(evidence["evidence_sha256"].item())) == 64
    validate_structural_cell_evidence(evidence)


def test_cell_evidence_rejects_gate_failure_without_classification() -> None:
    cells = _cells((False,), (0.0,))
    with pytest.raises(ValueError, match="failed gates require a failure class"):
        build_structural_cell_evidence(
            cells,
            gate_status=np.asarray([False]),
            failure_class=np.asarray(["none"]),
        )


def test_cell_evidence_distinguishes_corner_only_execution_from_unmatched() -> None:
    nominal = _cells((False, False, False), (0.0, 0.0, 0.0))
    corner = _cells((False, False, False, False), (0.0, 0.0, 0.0, 0.0))
    comparison = compare_common_support(nominal, corner)

    evidence = build_structural_cell_evidence(
        corner,
        gate_status=np.ones(4, dtype=bool),
        failure_class=np.asarray(["none"] * 4),
        comparison=comparison,
    )

    assert evidence["comparison_status"].tolist() == [
        "common_unmatched",
        "common_unmatched",
        "common_unmatched",
        "corner_only_executed",
    ]


@pytest.mark.parametrize(
    ("pathway", "filename"),
    [
        ("shaft", "articulated_shaft_atlas.npz"),
        ("ground", "articulated_ground_atlas.npz"),
    ],
)
def test_cell_evidence_assembles_identity_and_gates_from_one_atlas(
    pathway, filename
) -> None:
    with np.load(DATA / filename, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}

    evidence = build_structural_cell_evidence_from_atlas(pathway, arrays)

    assert evidence["cell_identity"].shape == (384,)
    assert np.all(evidence["gate_status"])
    assert set(evidence["failure_class"].tolist()) == {"none"}
    assert evidence["pathway"].item() == pathway


def test_cell_evidence_round_trip_is_atomic_and_tamper_evident(tmp_path) -> None:
    cells = _cells((True,), (0.0,))
    evidence = build_structural_cell_evidence(
        cells,
        gate_status=np.asarray([True]),
        failure_class=np.asarray(["none"]),
    )
    output = tmp_path / "cells.npz"

    write_structural_cell_evidence(evidence, output)

    assert not output.with_suffix(".npz.tmp").exists()
    loaded = load_structural_cell_evidence(output)
    assert loaded["evidence_sha256"].item() == evidence["evidence_sha256"].item()
    loaded["matched_load_work"][0] = False
    with output.open("wb") as stream:
        np.savez_compressed(stream, **loaded)
    with pytest.raises(RuntimeError, match="digest does not reproduce"):
        load_structural_cell_evidence(output)
