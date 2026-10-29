from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.humanoid_robotics.option_a.fixture import (
    run_swing_recovery_fixture,
)
from src.shared.python.humanoid_robotics.option_a.plant import (
    build_mujoco_model,
    distal_tip_position_m,
    gravity_bias_at_pose,
    transition_fd_agrees_with_rollout,
)
from src.shared.python.humanoid_robotics.option_a.trace_contract import (
    FAILURE_DEFINITION,
    payload_sha256,
    validate_trace_bundle,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_mujoco]

_REF = (
    Path(__file__).resolve().parents[3] / "fixtures" / "humanoid_robotics" / "option_a"
)
_SEED = 13837


def test_deterministic_trace_reference_bundle_and_size(tmp_path: Path) -> None:
    first = run_swing_recovery_fixture(_SEED)
    second = run_swing_recovery_fixture(_SEED)
    assert first["payload_sha256"] == second["payload_sha256"] == payload_sha256(first)
    manifest = json.loads(
        (_REF / "reference_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["payload_sha256"] == first["payload_sha256"]
    assert manifest["schema_version"] == "humanoid_option_a.trace/v2"
    stored = np.load(_REF / "reference_trace_v2.npz", allow_pickle=False)
    np.testing.assert_array_equal(stored["x_k"], first["x_k"])
    np.testing.assert_array_equal(stored["q_ref_k"], first["q_ref_k"])
    np.testing.assert_array_equal(stored["v_ref_k"], first["v_ref_k"])
    probe = tmp_path / "probe.npz"
    np.savez_compressed(probe, **{k: first[k] for k in first if k != "payload_sha256"})
    assert probe.stat().st_size < 1_000_000


def test_trace_contract_fails_closed() -> None:
    bundle = run_swing_recovery_fixture(_SEED)
    validate_trace_bundle(bundle)
    assert bundle["failure_definition"] == FAILURE_DEFINITION
    with pytest.raises(ValueError, match="unsupported"):
        validate_trace_bundle(
            {**bundle, "schema_version": "humanoid_option_a.trace/v0"}
        )
    missing = dict(bundle)
    missing.pop("A_k")
    with pytest.raises(ValueError, match="missing trace field"):
        validate_trace_bundle(missing)
    nan_u = np.array(bundle["u_k"], copy=True)
    nan_u[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_trace_bundle({**bundle, "u_k": nan_u})


def test_transition_fd_matches_rollout_at_three_states() -> None:
    model = build_mujoco_model()
    states = [
        (np.array([0.25, -0.15]), np.zeros(2)),
        (np.array([0.55, 0.35]), np.array([1.2, -0.7])),
        (np.array([0.12, -0.22]), np.array([0.35, -0.18])),
    ]
    transition_fd_agrees_with_rollout(model, states, atol=5e-3)


def test_gravity_torque_and_outcome_labels() -> None:
    model = build_mujoco_model()
    assert np.linalg.norm(gravity_bias_at_pose(model, np.array([0.42, -0.31]))) > 1e-3
    assert np.linalg.norm(gravity_bias_at_pose(model, np.zeros(2))) < 1e-6
    labels = set(run_swing_recovery_fixture(_SEED)["outcome_label"].tolist())
    assert labels == {"recovered", "unrecoverable"}


def test_failure_definition_task_space_contract() -> None:
    bundle = run_swing_recovery_fixture(_SEED)
    assert bundle["failure_definition"]["unit"] == "m"
    assert bundle["failure_definition"]["tolerance"] == pytest.approx(0.05)
    q_ref = np.asarray(bundle["q_ref_k"])
    tip_ref = distal_tip_position_m(q_ref[-1])
    assert tip_ref.shape == (2,)


def test_reference_trajectory_matches_nominal_horizon() -> None:
    bundle = run_swing_recovery_fixture(_SEED)
    q_ref = np.asarray(bundle["q_ref_k"])
    v_ref = np.asarray(bundle["v_ref_k"])
    steps = q_ref.shape[0]
    assert q_ref.shape == (steps, 2)
    assert v_ref.shape == (steps, 2)
    trial_count = int(np.asarray(bundle["trial_index"]).max()) + 1
    assert int(np.asarray(bundle["x_k"]).shape[0]) == steps * trial_count
