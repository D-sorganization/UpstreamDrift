"""Option A-2 swing-and-recovery MuJoCo fixture (#13837)."""

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
    gravity_bias_at_pose,
    transition_fd_agrees_with_rollout,
)
from src.shared.python.humanoid_robotics.option_a.trace_contract import (
    TRACE_SCHEMA_VERSION,
    export_reference_npz,
    payload_sha256,
    validate_trace_bundle,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_mujoco]

_REF = (
    Path(__file__).resolve().parents[3] / "fixtures" / "humanoid_robotics" / "option_a"
)
_SEED = 13837


def _bundle_with_nan_u0(bundle: dict) -> dict:
    bad = dict(bundle)
    u = np.array(bad["u_k"], copy=True)
    u[0, 0] = np.nan
    bad["u_k"] = u
    return bad


def test_same_seed_yields_identical_payload_sha256() -> None:
    a = run_swing_recovery_fixture(_SEED)
    b = run_swing_recovery_fixture(_SEED)
    assert a["payload_sha256"] == b["payload_sha256"] == payload_sha256(a)


@pytest.mark.parametrize(
    "mutator,match",
    [
        (
            lambda b: {**b, "schema_version": "humanoid_option_a.trace/v0"},
            "unsupported",
        ),
        (lambda b: (lambda x: (x.pop("A_k"), x)[1])(dict(b)), "missing trace field"),
        (_bundle_with_nan_u0, "non-finite"),
    ],
)
def test_trace_contract_fails_closed(mutator, match: str) -> None:
    bundle = run_swing_recovery_fixture(_SEED)
    with pytest.raises(ValueError, match=match):
        validate_trace_bundle(mutator(bundle))


def test_transition_fd_matches_rollout_at_three_states() -> None:
    model = build_mujoco_model()
    states = [
        (np.array([0.25, -0.15]), np.zeros(2)),
        (np.array([0.55, 0.35]), np.array([1.2, -0.7])),
        (np.array([0.12, -0.22]), np.array([0.35, -0.18])),
    ]
    transition_fd_agrees_with_rollout(model, states, atol=5e-3)


def test_gravity_torque_at_non_vertical_pose() -> None:
    model = build_mujoco_model()
    assert np.linalg.norm(gravity_bias_at_pose(model, np.array([0.42, -0.31]))) > 1e-3
    assert np.linalg.norm(gravity_bias_at_pose(model, np.zeros(2))) < 1e-6


def test_grid_yields_recovered_and_unrecoverable() -> None:
    labels = set(run_swing_recovery_fixture(_SEED)["outcome_label"].tolist())
    assert labels == {"recovered", "unrecoverable"}


def test_reference_bundle_matches_fixture_and_stays_small() -> None:
    bundle = run_swing_recovery_fixture(_SEED)
    manifest = json.loads(
        (_REF / "reference_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["payload_sha256"] == bundle["payload_sha256"]
    stored = np.load(_REF / "reference_trace_v1.npz", allow_pickle=False)
    np.testing.assert_array_equal(stored["x_k"], bundle["x_k"])
    tmp = _REF / "_size_probe.npz"
    export_reference_npz(bundle, str(tmp))
    assert tmp.stat().st_size < 1_000_000
    tmp.unlink()
