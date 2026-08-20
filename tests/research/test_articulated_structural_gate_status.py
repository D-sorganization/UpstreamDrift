"""Contracts for fail-closed articulated structural gate evidence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_structural_cell_evidence import (
    build_structural_cell_evidence,
)
from scripts.research.proximal_distal_energy.articulated_structural_common_support import (
    extract_headline_cells,
)
from scripts.research.proximal_distal_energy.articulated_structural_gate_status import (
    derive_structural_cell_gate_status,
)

pytestmark = pytest.mark.scientific
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"


def _coordinates() -> dict[str, np.ndarray]:
    return {
        "state_case_index": np.asarray([0]),
        "state_sample_index": np.asarray([0]),
        "velocity_factors": np.asarray([1.0, -1.0]),
        "time_steps_s": np.asarray([0.00025, 0.000125]),
        "engine_names": np.asarray(["mujoco", "pinocchio"]),
        "horizons_s": np.asarray([0.004, 0.01, 0.025, 0.05]),
    }


def _shaft_fixture() -> dict[str, np.ndarray]:
    arrays = _coordinates()
    arrays["activation_names"] = np.asarray(["rigid", "bending", "torsion", "coupled"])
    full = (1, 4, 2, 2, 2, 4)
    parity = (1, 4, 2, 2, 4)
    arrays.update(
        {
            "numerical_gates_passed": np.ones(full, dtype=bool),
            "parity_gates_passed": np.ones(parity, dtype=bool),
            "small_deflection_gate_passed": np.ones(full, dtype=bool),
            "twist_gate_passed": np.ones(full, dtype=bool),
        }
    )
    return arrays


def _ground_fixture() -> dict[str, np.ndarray]:
    arrays = _coordinates()
    arrays["ground_activation_names"] = np.asarray(
        ["fixed", "translation", "free_moment", "coupled"]
    )
    arrays["primary_numerical"] = np.ones((1, 4, 2, 2, 2, 4), dtype=bool)
    arrays["primary_parity"] = np.ones((1, 4, 2, 2, 4), dtype=bool)
    return arrays


def test_shaft_gate_status_retains_all_simultaneous_failure_classes() -> None:
    arrays = _shaft_fixture()
    arrays["numerical_gates_passed"][0, 3, 0, 0, 0, 0] = False
    arrays["parity_gates_passed"][0, 0, 0, 0, 0] = False
    arrays["small_deflection_gate_passed"][0, 3, 0, 0, 0, 0] = False
    arrays["twist_gate_passed"][0, 0, 0, 0, 0, 0] = False

    status = derive_structural_cell_gate_status("shaft", arrays)

    assert status.gate_status.shape == (32,)
    assert not status.gate_status[0]
    assert not status.gate_status[4]
    assert status.failure_class[0] == (
        "numerical_gate_failure+parity_gate_failure+"
        "small_deflection_gate_failure+twist_gate_failure"
    )
    assert status.failure_class[4] == "parity_gate_failure"
    assert status.failure_class[1] == "none"


def test_ground_gate_status_broadcasts_parity_across_both_engines() -> None:
    arrays = _ground_fixture()
    arrays["primary_parity"][0, 3, 1, 1, 2] = False
    arrays["primary_numerical"][0, 0, 1, 1, 1, 2] = False

    status = derive_structural_cell_gate_status("ground", arrays)

    reshaped = status.gate_status.reshape(1, 2, 2, 2, 4)
    failures = status.failure_class.reshape(1, 2, 2, 2, 4)
    assert not reshaped[0, 1, 1, 0, 2]
    assert not reshaped[0, 1, 1, 1, 2]
    assert failures[0, 1, 1, 0, 2] == "parity_gate_failure"
    assert failures[0, 1, 1, 1, 2] == ("numerical_gate_failure+parity_gate_failure")


@pytest.mark.parametrize(
    ("pathway", "filename"),
    [
        ("shaft", "articulated_shaft_atlas.npz"),
        ("ground", "articulated_ground_atlas.npz"),
    ],
)
def test_committed_gate_status_aligns_with_headline_cells(pathway, filename) -> None:
    with np.load(DATA / filename, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}

    cells = extract_headline_cells(pathway, arrays)
    status = derive_structural_cell_gate_status(pathway, arrays)
    evidence = build_structural_cell_evidence(
        cells,
        gate_status=status.gate_status,
        failure_class=status.failure_class,
    )

    assert status.gate_status.shape == (384,)
    assert np.all(status.gate_status)
    assert set(status.failure_class.tolist()) == {"none"}
    assert evidence["cell_identity"].shape == status.gate_status.shape
    assert np.array_equal(evidence["gate_status"], status.gate_status)


def test_gate_status_rejects_missing_branch_and_shape_drift() -> None:
    missing = _shaft_fixture()
    missing["activation_names"] = np.asarray(
        ["rigid", "bending", "torsion", "torsion_2"]
    )
    with pytest.raises(ValueError, match="require exactly one"):
        derive_structural_cell_gate_status("shaft", missing)

    malformed = _ground_fixture()
    malformed["primary_numerical"] = np.ones((1, 4, 2, 2, 1, 4), dtype=bool)
    with pytest.raises(ValueError, match="registered design"):
        derive_structural_cell_gate_status("ground", malformed)
