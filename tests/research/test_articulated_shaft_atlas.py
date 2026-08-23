from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_shaft_atlas import (
    ArticulatedShaftAtlasConfig,
    _resolve_states,
)
from scripts.research.proximal_distal_energy.articulated_atlas_authority import (
    ArticulatedAtlasAuthority,
)
from scripts.research.proximal_distal_energy.articulated_scaled_authority import (
    load_scaled_authority,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"


def test_shaft_atlas_configuration_fails_closed() -> None:
    with pytest.raises(ValueError, match="activations"):
        ArticulatedShaftAtlasConfig(activations=("rigid", "coupled", "rigid"))
    with pytest.raises(ValueError, match="match_relative_tolerance"):
        ArticulatedShaftAtlasConfig(match_relative_tolerance=0.0)
    with pytest.raises(ValueError, match="horizons_s"):
        ArticulatedShaftAtlasConfig(horizons_s=(0.004, 0.049))
    with pytest.raises(ValueError, match="worker_count"):
        ArticulatedShaftAtlasConfig(worker_count=0)
    with pytest.raises(ValueError, match="shaft_damping_ratio"):
        ArticulatedShaftAtlasConfig(shaft_damping_ratio=1.0)
    with pytest.raises(ValueError, match="bending_frequency_scale"):
        ArticulatedShaftAtlasConfig(bending_frequency_scale=0.0)


def test_committed_shaft_atlas_is_complete_and_current() -> None:
    record = json.loads(
        (DATA / "articulated_shaft_atlas.json").read_text(encoding="utf-8")
    )
    with np.load(DATA / "articulated_shaft_atlas.npz") as source:
        assert source["peak_station_force_n"].shape == (12, 4, 2, 2, 2, 4)
        assert source["trajectory_relative_error"].shape == (12, 4, 2, 2, 4)
        assert np.all(source["numerical_gates_passed"])
        assert np.all(source["parity_gates_passed"])
        assert np.all(source["small_deflection_gate_passed"])
        assert np.all(source["twist_gate_passed"])
        assert source["matched_load_work"].shape == (12, 2, 2, 2, 4)
    assert record["schema_version"] == "articulated-shaft-atlas/v1"
    assert record["design"]["trajectory_count"] == 384
    assert record["results"]["all_registered_gates_passed"] is True
    assert record["limitations"]["calibration_status"] == (
        "synthetic_reference_not_equipment_calibrated"
    )
    diagnostic = json.loads(
        (DATA / "articulated_shaft_time_step_diagnostic.json").read_text(
            encoding="utf-8"
        )
    )
    assert diagnostic["schema_version"] == ("articulated-shaft-time-step-diagnostic/v1")
    assert diagnostic["monotone_refinement_passed"] is True
    assert [item["step_s"] for item in diagnostic["results"]] == [
        0.00025,
        0.000125,
        0.0000625,
    ]


def test_shaft_atlas_retains_infeasible_state_and_executes_feasible_subset() -> None:
    scaled = load_scaled_authority(
        DATA / "articulated_structural_authority_height_scale_low.json",
        DATA / "articulated_structural_authority_height_scale_low.npz",
    )
    authority = ArticulatedAtlasAuthority.from_scaled(scaled)

    selection = _resolve_states(authority, ArticulatedShaftAtlasConfig())

    assert len(selection.planned_states) == 12
    assert len(selection.feasible_states) == 11
    assert selection.retained_failures == (
        {"case_index": 0, "phase_index": 12, "failure_class": "ik_nonconvergence"},
    )
