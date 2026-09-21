"""Fail-closed source identity, integration, and G1 regression (#10336)."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.candidate_replay import (
    ReplayPlant,
    replay_controls,
)
from src.engines.physics_engines.mujoco.python.replay_contract import (
    ReplaySettings,
    configuration_failures,
    g1_acceptance,
    load_candidate,
    metric_coverage,
    window_indices,
)

ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.unit
SOURCE = ROOT / "evidence/matched/driver_full_pinocchio"
SPEC = (
    ROOT
    / "docs/development/full_body_models/evidence/ground_support/anthro_driver/full_body_spec_hipcal_scaled.json"
)


def test_g1_window_is_exact_and_rejects_truncation():
    times = np.arange(654) / 360
    np.testing.assert_array_equal(window_indices(times, 0.85), np.arange(307))
    with pytest.raises(ValueError, match="frame boundary"):
        window_indices(times, 0.851)
    with pytest.raises(ValueError, match="frame boundary"):
        window_indices(times[:100], 0.85)


def test_legacy_source_cannot_certify_same_configuration():
    receipt = json.loads((SOURCE / "receipt.json").read_text())
    document = json.loads(SPEC.read_text())
    failures = configuration_failures(receipt, document, ReplaySettings())
    assert any("armature" in reason for reason in failures)
    assert any("contact" in reason for reason in failures)
    assert any("candidate_sha256" in reason for reason in failures)


def test_explicit_configuration_matches_and_drift_fails_closed():
    document = json.loads(SPEC.read_text())
    receipt = {
        "armature_kg_m2": 0.005,
        "contact": deepcopy(document["contact"]),
        "candidate_sha256": "a" * 64,
        "control_interpolation": "zero_order_hold",
    }
    assert configuration_failures(receipt, document, ReplaySettings()) == []
    receipt["armature_kg_m2"] = 0.0
    assert configuration_failures(receipt, document, ReplaySettings())
    receipt["armature_kg_m2"] = float("nan")
    assert configuration_failures(receipt, document, ReplaySettings())
    receipt["armature_kg_m2"] = 0.005
    receipt["contact"]["parameters"]["friction_coefficient"] = 123
    assert configuration_failures(receipt, document, ReplaySettings())


def test_candidate_hash_and_coordinate_validation(tmp_path):
    path = SOURCE / "candidate.npz"
    arrays = load_candidate(path, hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="SHA"):
        load_candidate(path, "0" * 64)
    arrays["coordinate_order"][1] = arrays["coordinate_order"][0]
    invalid = tmp_path / "invalid.npz"
    np.savez(invalid, **arrays)
    with pytest.raises(ValueError, match="unique"):
        load_candidate(invalid, hashlib.sha256(invalid.read_bytes()).hexdigest())


def test_candidate_loading_supports_n_minus_1_controls(tmp_path):
    path = SOURCE / "candidate.npz"
    arrays = load_candidate(path, hashlib.sha256(path.read_bytes()).hexdigest())
    arrays["u"] = arrays["u"][:-1]
    shorter = tmp_path / "shorter_u.npz"
    np.savez(shorter, **arrays)
    loaded = load_candidate(shorter, hashlib.sha256(shorter.read_bytes()).hexdigest())
    assert loaded["u"].shape == (len(arrays["time_s"]) - 1, arrays["u"].shape[1])


def test_missing_club_population_cannot_be_zero_error_success():
    path = SOURCE / "candidate.npz"
    arrays = load_candidate(path, hashlib.sha256(path.read_bytes()).hexdigest())
    assert metric_coverage(arrays, 307)
    for i, name in enumerate(arrays["labels"]):
        if name.startswith("Marker_"):
            arrays["valid"][:, i] = False
    assert not metric_coverage(arrays, 307)


@pytest.mark.parametrize(
    "missing", ["parity", "complete", "converged", "root_history", "coverage"]
)
def test_g1_never_accepts_missing_evidence(missing):
    evidence = dict.fromkeys(
        ["parity", "complete", "converged", "root_history", "coverage"], True
    )
    evidence[missing] = False
    verdict = g1_acceptance({}, evidence)
    assert not verdict["is_physically_accepted"]
    assert any(
        g["name"] == missing and g["status"] != "passed" for g in verdict["gates"]
    )


def test_g1_missing_physical_metrics_cannot_pass():
    evidence = dict.fromkeys(
        ["parity", "complete", "converged", "root_history", "coverage"], True
    )
    assert not g1_acceptance({}, evidence)["is_physically_accepted"]


def test_complete_g1_measurements_pass_only_with_source_dynamics():
    metrics = {
        "whole_marker_rmse_m": 0.01,
        "early_marker_rmse_m": 0.005,
        "terminal_marker_rmse_m": 0.01,
        "club_marker_rmse_m": 0.01,
        "pelvis_yaw_rmse_rad": 0.01,
        "max_normal_force_n": 800.0,
        "max_penetration_m": 0.001,
        "max_closure_residual_m": 0.001,
        "max_closure_residual_rad": 0.001,
    }
    evidence = dict.fromkeys(
        [
            "parity",
            "complete",
            "converged",
            "root_history",
            "coverage",
            "source_dynamics",
        ],
        True,
    )
    assert g1_acceptance(metrics, evidence)["is_physically_accepted"]
    evidence.pop("source_dynamics")
    assert not g1_acceptance(metrics, evidence)["is_physically_accepted"]


def test_canonical_source_preparation_ignores_only_invalid_marker_values(tmp_path):
    from src.engines.physics_engines.mujoco.python.replay_evidence import (
        ReplayFiles,
        _prepare,
    )

    files = ReplayFiles(
        SOURCE / "candidate.npz",
        SOURCE / "receipt.json",
        SPEC,
        ROOT / "data/C3D_TA_Driver.c3d",
        ROOT
        / "docs/development/full_body_models/evidence/ground_support/anthro_driver_shoot_g025/receipt.json",
        tmp_path,
        hashlib.sha256((SOURCE / "candidate.npz").read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="Unverified source configuration"):
        _prepare(files, ReplaySettings(), False)
    assert _prepare(files, ReplaySettings(), True)[-1]


def test_replay_uses_original_state_without_resets_and_zoh_controls():
    class ConstantPlant:
        def acceleration(self, q, v, effort):
            return effort

    time = np.array([0.0, 0.1, 0.2])
    q, v, reason = replay_controls(
        ConstantPlant(),
        time,
        np.array([0.0]),
        np.array([0.0]),
        np.array([[2.0], [0.0], [999.0]]),
        ReplaySettings(),
    )
    assert reason is None
    np.testing.assert_allclose(q[:, 0], [0, 0.01, 0.03], atol=1e-9)
    np.testing.assert_allclose(v[:, 0], [0, 0.2, 0.2], atol=1e-9)


def test_budget_exhaustion_returns_only_completed_frames():
    class ConstantPlant:
        def acceleration(self, q, v, effort):
            return effort

    q, v, failure = replay_controls(
        ConstantPlant(),
        np.array([0.0, 0.1]),
        np.zeros(1),
        np.zeros(1),
        np.ones((2, 1)),
        ReplaySettings(max_evaluations=1),
    )
    assert q.shape == v.shape == (1, 1)
    assert failure is not None and "budget" in failure


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, 0.026])
def test_g1_rejects_invalid_or_excessive_whole_error(value):
    evidence = dict.fromkeys(
        ["parity", "complete", "converged", "root_history", "coverage"], True
    )
    verdict = g1_acceptance({"whole_marker_rmse_m": value}, evidence)
    assert not verdict["is_physically_accepted"]


def test_generated_receipt_rejects_legacy_source_and_preserves_artifact_hashes():
    from src.shared.python.motion_matching.pipeline.receipt_schema import (
        CandidateReplayReceipt,
    )

    directory = ROOT / "evidence/matched/driver_full_mujoco_replay"
    receipt = CandidateReplayReceipt.model_validate_json(
        (directory / "receipt.json").read_text()
    )
    assert not receipt.acceptance.is_physically_accepted
    assert receipt.parity["status"] == "UNVERIFIED"
    assert receipt.integration["pose_resets"] == 0
    assert (
        receipt.candidate_sha256
        == hashlib.sha256((SOURCE / "candidate.npz").read_bytes()).hexdigest()
    )
    for artifact in receipt.artifacts.values():
        assert (
            hashlib.sha256((directory / artifact["path"]).read_bytes()).hexdigest()
            == artifact["sha256"]
        )


@pytest.mark.requires_mujoco
def test_native_armature_is_mapped_by_name_and_changes_mass_matrix():
    pytest.importorskip("mujoco")
    document = json.loads(SPEC.read_text())
    plant = ReplayPlant(document, 0.0002577801048755657, ReplaySettings())
    q = np.zeros(len(plant.names))
    mass = plant.mass_matrix(q)
    unconditioned = ReplayPlant(
        document, 0.0002577801048755657, ReplaySettings(armature_kg_m2=0.0)
    )
    delta = mass - unconditioned.mass_matrix(q)
    np.testing.assert_allclose(delta, np.diag(plant.actuated * 0.005), atol=1e-12)
    assert not np.any(plant.actuated[:6])
    assert (
        plant.adapter.contact_parameters.as_document()
        == document["contact"]["parameters"]
    )
    assert not np.any(plant.adapter.model.geom_contype)
    assert not np.any(plant.adapter.model.geom_conaffinity)
