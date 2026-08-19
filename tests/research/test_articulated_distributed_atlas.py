from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_distributed_atlas import (
    DistributedAtlasConfig,
)
from scripts.research.proximal_distal_energy.articulated_distributed_forward import (
    DistributedForwardConfig,
)

pytestmark = pytest.mark.scientific

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"


def test_distributed_atlas_configuration_fails_closed() -> None:
    with pytest.raises(ValueError, match="positive odd"):
        DistributedAtlasConfig(station_counts=(1, 2, 3))
    with pytest.raises(ValueError, match="horizons_s"):
        DistributedAtlasConfig(horizons_s=(0.004, 0.025))
    with pytest.raises(ValueError, match="divisible"):
        DistributedAtlasConfig(
            forward=DistributedForwardConfig(
                duration_s=0.05,
                time_steps_s=(0.001, 0.0005),
            ),
            horizons_s=(0.0045, 0.01, 0.025, 0.05),
        )


def test_default_atlas_declares_nested_horizon_and_equal_total_stiffness() -> None:
    config = DistributedAtlasConfig()

    assert config.station_counts == (1, 3, 5)
    assert config.horizons_s == (0.004, 0.01, 0.025, 0.05)
    assert config.horizons_s[-1] == config.forward.duration_s
    assert config.total_stiffness_n_m == 1800.0


def test_committed_distributed_atlas_is_complete_finite_and_qualified() -> None:
    summary = json.loads(
        (DATA / "articulated_distributed_grip_atlas.json").read_text(encoding="utf-8")
    )
    assert summary["schema_version"] == "articulated-distributed-grip-atlas/v1"
    assert summary["design"]["trajectory_count"] == 288
    assert summary["design"]["station_counts_per_hand"] == [1, 3, 5]
    assert summary["design"]["horizons_s"] == [0.004, 0.01, 0.025, 0.05]
    assert summary["results"]["maximum_transition_count"] == 0
    assert summary["results"]["failed_numerical_cell_count"] == 0
    assert summary["results"]["failed_parity_cell_count"] == 0
    assert summary["results"]["active_set_parity_failures"] == 0
    assert summary["results"]["time_refinement_passed"]
    assert summary["results"]["station_refinement_passed"]
    assert summary["results"]["all_registered_gates_passed"]

    with np.load(DATA / "articulated_distributed_grip_atlas.npz") as arrays:
        assert arrays["peak_station_force_n"].shape == (12, 3, 2, 2, 2, 4)
        assert arrays["trajectory_relative_error"].shape == (12, 3, 2, 2, 4)
        assert np.all(np.isfinite(arrays["peak_station_force_n"]))
        assert np.all(np.isfinite(arrays["final_q"]))
        assert np.all(arrays["numerical_gates_passed"])
        assert np.all(arrays["parity_gates_passed"])
