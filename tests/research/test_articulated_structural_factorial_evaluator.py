"""Observable and authority tests for the structural-factorial evaluator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_structural_factorial_evaluator import (
    _horizon_rows,
    require_native_engine,
    validate_authorities,
)
from scripts.research.proximal_distal_energy.spatial_full_body import (
    build_spatial_model,
)

ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = ROOT / (
    "docs/research/proximal_distal_energy_transfer/data/"
    "articulated_structural_factorial_plan.json"
)
pytestmark = pytest.mark.scientific


def test_committed_plan_binds_current_authority_bytes() -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))

    validate_authorities(plan)


def test_horizon_observables_use_declared_terminal_axis_and_signed_integrals() -> None:
    model = build_spatial_model()
    q = np.zeros((2, model.nq))
    qd = np.zeros((2, model.nq))
    qd[:, 14] = 1.0
    trace = {
        "time_s": np.array([0.0, 0.05]),
        "q": q,
        "qd": qd,
        "grip_dissipation_power_w": np.array([-1.0, -1.0]),
        "shaft_damping_power_w": np.array([-2.0, -2.0]),
        "ground_damping_power_w": np.array([-3.0, -3.0]),
        "maximum_station_force_n": np.array([2.0, 4.0]),
    }

    rows = _horizon_rows(
        model=model,
        trace=trace,
        contact_force=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        contact_power=np.array([2.0, 2.0]),
        horizons_s=(0.05,),
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["final_club_translation_speed_m_s"] == pytest.approx(1.0)
    assert row["club_linear_momentum_change_kg_m_s"] == pytest.approx(0.0)
    assert row["signed_contact_impulse_n_s"] == pytest.approx(0.05)
    assert row["signed_contact_work_j"] == pytest.approx(0.1)
    assert row["terminal_contact_dissipation_j"] == pytest.approx(0.05)
    assert row["terminal_shaft_dissipation_j"] == pytest.approx(0.1)
    assert row["terminal_ground_dissipation_j"] == pytest.approx(0.15)
    assert row["peak_grip_force_n"] == pytest.approx(4.0)


def test_unknown_engine_fails_before_import() -> None:
    with pytest.raises(ValueError, match="engine must be"):
        require_native_engine("invented")
