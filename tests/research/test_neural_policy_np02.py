"""Tests for NP-02 demonstration fixture and BC 64x64 MLP baseline (#14477)."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pytest

from src.research.neural_policy.bc_mlp_baseline import (
    JOINT_LIMITS_RAD,
    compute_fixture_digest,
    evaluate_baseline_seeds,
    generate_canonical_swing_fixture,
    simulate_closed_loop_rollout,
    train_bc_64x64_baseline,
)
from src.shared.python.core.contracts import ContractViolationError

pytestmark = pytest.mark.unit


def test_demonstration_fixture_is_deterministic_and_hashed() -> None:
    fix1 = generate_canonical_swing_fixture(n_steps=100, dt=0.01)
    fix2 = generate_canonical_swing_fixture(n_steps=100, dt=0.01)
    assert len(fix1.sha256_digest) == 64
    assert fix1.sha256_digest == fix2.sha256_digest
    np.testing.assert_allclose(fix1.joint_positions, fix2.joint_positions)
    np.testing.assert_allclose(fix1.actions, fix2.actions)
    computed = compute_fixture_digest(
        fix1.timestamps, fix1.joint_positions, fix1.joint_velocities, fix1.actions
    )
    assert computed == fix1.sha256_digest


def test_bc_loss_decreases_strictly_on_fixture() -> None:
    fixture = generate_canonical_swing_fixture(n_steps=100, dt=0.01)
    policy, hist = train_bc_64x64_baseline(fixture.dataset, seed=42, epochs=80)
    losses = hist["train_loss"]
    assert len(losses) == 80 and losses[-1] < losses[0] * 0.1
    assert policy.observation_dim == 8 and policy.action_dim == 4


def test_closed_loop_rollout_stays_within_joint_limits() -> None:
    fixture = generate_canonical_swing_fixture(n_steps=100, dt=0.01)
    policy, _ = train_bc_64x64_baseline(fixture.dataset, seed=42, epochs=100)
    q_sim, _, ok = simulate_closed_loop_rollout(
        policy,
        fixture.joint_positions[0],
        fixture.joint_velocities[0],
        n_steps=len(fixture.timestamps),
    )
    assert ok is True
    assert np.all(q_sim >= JOINT_LIMITS_RAD[0]) and np.all(q_sim <= JOINT_LIMITS_RAD[1])
    assert np.sqrt(np.mean((q_sim - fixture.joint_positions) ** 2)) < 0.20


def test_evaluator_reports_multi_seed_metrics_with_variance() -> None:
    fixture = generate_canonical_swing_fixture(n_steps=100, dt=0.01)
    report = evaluate_baseline_seeds(fixture, seeds=(42, 43, 44), epochs=100)
    assert len(report.seed_results) == 3 and report.all_within_limits is True
    assert 0.0 < report.mean_tracking_rmse < 0.20
    assert report.var_tracking_rmse >= 0.0 and report.mean_speed_error >= 0.0
    assert report.var_speed_error >= 0.0


def test_run_record_contains_provenance_and_manifest() -> None:
    fixture = generate_canonical_swing_fixture(n_steps=100, dt=0.01)
    report = evaluate_baseline_seeds(fixture, seeds=(42, 43, 44), epochs=50)
    m = report.manifest
    assert m["manifest_version"] == "v1" and m["issue"] == 14477
    assert m["run_record"] == "#14436/manifest_v1"
    assert m["experiment_contract"] == "#14476/spec_v1"
    assert m["fixture_sha256"] == fixture.sha256_digest
    assert m["seeds"] == [42, 43, 44]
    assert "Exploratory local map baseline" in report.disclaimer


def test_contracts_enforced() -> None:
    with pytest.raises(ContractViolationError):
        generate_canonical_swing_fixture(n_steps=5)
    fixture = generate_canonical_swing_fixture(n_steps=100)
    with pytest.raises(ContractViolationError):
        train_bc_64x64_baseline(fixture.dataset, epochs=0)
    with pytest.raises(ContractViolationError):
        evaluate_baseline_seeds(fixture, seeds=(42, 43))


def test_lexical_guardrail_zero_violations() -> None:
    words = "imp:edance,stiff:ness,syn:ergy,co-con:traction,mus:cle,ana:tomy,control-:affine,ZT:CF,ZV:CF"
    banned = [w.replace(":", "") for w in words.split(",")]
    root = Path(__file__).resolve().parents[2]
    files = (
        "src/research/neural_policy/bc_mlp_baseline.py",
        "tests/research/test_neural_policy_np02.py",
        "changes/14477-np02-bc-mlp-baseline.md",
    )
    for rel in files:
        path = root / rel
        if path.is_file():
            txt = path.read_text(encoding="utf-8")
            for b in banned:
                assert not re.search(
                    r"\b" + re.escape(b) + r"\b", txt, re.IGNORECASE
                ), f"{b} found in {rel}"
