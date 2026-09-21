"""Native smoke test of the Crocoddyl full-body marker fit (MS-31, #10338).

Runs the whole driver on a 0.01 s window (four nodes, two FDDP iterations)
against the committed anthropometric driver document and the tour capture.
It needs the qualified Pinocchio + Crocoddyl stack, so it skips elsewhere.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_pinocchio,
    pytest.mark.requires_crocoddyl,
]

pytest.importorskip("pinocchio")
pytest.importorskip("crocoddyl")

REPO = Path(__file__).resolve().parents[3]
DOCUMENT = (
    REPO
    / "docs/development/full_body_models/evidence/ground_support/anthro_driver"
    / "full_body_spec_hipcal_scaled.json"
)
RECEIPT = (
    REPO
    / "docs/development/full_body_models/evidence/ground_support"
    / "anthro_driver_shoot_g025/receipt.json"
)
CAPTURE = REPO / "data/C3D_TA_Driver.c3d"


@pytest.fixture(scope="module")
def receipt(tmp_path_factory: pytest.TempPathFactory) -> dict:
    from src.engines.physics_engines.pinocchio.python.crocoddyl_problem import (
        FitHorizon,
        FitWeights,
    )
    from src.engines.physics_engines.pinocchio.python.full_body_fit import (
        SolverSettings,
        load_inputs,
        run_fit,
    )
    from src.engines.physics_engines.pinocchio.python.marker_kinematics import (
        MarkerIkOptions,
    )

    inputs = load_inputs(
        DOCUMENT,
        CAPTURE,
        RECEIPT,
        FitHorizon(0.0, 3.0 / 360.0, 1.0 / 360.0),
        FitWeights(),
        None,
    )
    settings = SolverSettings(max_iterations=2, verbose=False)
    out = tmp_path_factory.mktemp("fit")
    return run_fit(inputs, settings, out, ik_options=MarkerIkOptions(iterations=10))


def test_warm_start_ik_matches_canonical_band(receipt: dict) -> None:
    # The canonical MuJoCo IK on this document is 27 to 29 mm.
    assert receipt["warm_start_ik"]["marker_rms_m"] < 0.04
    assert receipt["warm_start_ik"]["closure_position_error_max_m"] < 1e-4


def test_solver_ran_and_replay_is_finite(receipt: dict) -> None:
    assert receipt["solver"]["solver"] == "crocoddyl.SolverBoxFDDP"
    assert receipt["solver"]["iterations"] >= 1
    assert receipt["replay_note"] is None
    rollout = receipt["metrics"]["fddp_rollout"]["shared"]["whole_marker_rmse_m"]
    replay = receipt["metrics"]["replay"]["shared"]["whole_marker_rmse_m"]
    assert np.isfinite(rollout) and np.isfinite(replay)
    assert rollout < 0.06 and replay < 0.06


def test_receipt_records_plant_identity(receipt: dict) -> None:
    for key in ("document_sha256", "capture_sha256", "candidate_sha256"):
        assert len(receipt[key]) == 64
    assert receipt["armature_kg_m2"] == pytest.approx(5e-3)
    assert receipt["solver"]["integrator"] == "linear_implicit_euler"
    assert set(receipt["cost_breakdown"]) == {"warm_start", "fddp"}
