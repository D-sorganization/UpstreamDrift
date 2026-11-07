"""Behavior-cloned 64x64 MLP baseline for planar swing (#14477, NP-02).

Exploratory baseline: trajectory fixture, BC training, rollout simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from src.learning.imitation._base import TrainingConfig
from src.learning.imitation._bc import BehaviorCloning
from src.learning.imitation.dataset import Demonstration, DemonstrationDataset
from src.shared.python.core.contracts import require

JOINT_LIMITS_RAD: tuple[NDArray[np.floating], NDArray[np.floating]] = (
    np.array([-np.pi, -np.pi, -np.pi, -np.pi], dtype=float),
    np.array([np.pi, np.pi, np.pi, np.pi], dtype=float),
)
LINK_LENGTHS_M = np.array([0.3, 0.3, 0.25, 1.0], dtype=float)
M_DIAG = np.array([5.0, 2.0, 0.8, 0.2], dtype=float)
DAMPING = np.array([2.0, 1.0, 0.5, 0.1], dtype=float)


@dataclass(frozen=True)
class DemonstrationFixture:
    """Deterministic demonstration fixture and canonical digest."""

    timestamps: NDArray[np.floating]
    joint_positions: NDArray[np.floating]
    joint_velocities: NDArray[np.floating]
    actions: NDArray[np.floating]
    sha256_digest: str
    dataset: DemonstrationDataset


@dataclass(frozen=True)
class SeedResult:
    """Evaluation result for a single seed."""

    seed: int
    initial_loss: float
    final_loss: float
    tracking_rmse: float
    clubhead_speed_error: float
    within_joint_limits: bool


@dataclass(frozen=True)
class BaselineEvaluationReport:
    """Multi-seed summary with mean and variance for NP-01 evaluator."""

    seeds: tuple[int, ...]
    mean_tracking_rmse: float
    var_tracking_rmse: float
    mean_speed_error: float
    var_speed_error: float
    all_within_limits: bool
    seed_results: tuple[SeedResult, ...]
    manifest: dict[str, str | int | float | list[int]]
    disclaimer: str = (
        "Exploratory local map baseline only; results do not generalize off-manifold."
    )


def compute_fixture_digest(
    timestamps: NDArray[np.floating],
    positions: NDArray[np.floating],
    velocities: NDArray[np.floating],
    actions: NDArray[np.floating],
) -> str:
    """Compute deterministic SHA-256 digest of demonstration arrays."""
    h = hashlib.sha256()
    for arr in (timestamps, positions, velocities, actions):
        h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    return h.hexdigest()


def generate_canonical_swing_fixture(
    n_steps: int = 100, dt: float = 0.01
) -> DemonstrationFixture:
    """Build deterministic 4-DOF torso-arm-wrist-club demonstration fixture."""
    require(n_steps >= 10, "n_steps must be >= 10", None)
    require(dt > 0.0, "dt must be positive", None)
    t = np.arange(n_steps, dtype=float) * dt
    q = np.zeros((n_steps, 4), dtype=float)
    for i in range(4):
        q[:, i] = 0.5 * np.sin((i + 1) * 0.5 * np.pi * t) - 0.2 * (i + 1)

    qdot = np.gradient(q, dt, axis=0)
    qddot = np.gradient(qdot, dt, axis=0)
    tau = (qddot * M_DIAG) + (qdot * DAMPING)
    digest = compute_fixture_digest(t, q, qdot, tau)
    demo = Demonstration(
        timestamps=t,
        joint_positions=q,
        joint_velocities=qdot,
        actions=tau,
        task_id="planar_4dof_swing",
    )
    return DemonstrationFixture(t, q, qdot, tau, digest, DemonstrationDataset([demo]))


def train_bc_64x64_baseline(
    dataset: DemonstrationDataset,
    seed: int = 42,
    epochs: int = 150,
    learning_rate: float = 0.01,
) -> tuple[BehaviorCloning, dict[str, list[float]]]:
    """Train 64x64 MLP baseline policy using Behavior Cloning."""
    require(epochs > 0, "epochs must be positive", None)
    require(learning_rate > 0.0, "learning_rate must be positive", None)
    config = TrainingConfig(
        epochs=epochs,
        batch_size=32,
        learning_rate=learning_rate,
        hidden_sizes=[64, 64],
        activation="tanh",
        seed=seed,
    )
    bc = BehaviorCloning(observation_dim=8, action_dim=4, config=config)
    return bc, bc.train(dataset, validation_split=0.05)


def simulate_closed_loop_rollout(
    policy: BehaviorCloning,
    q0: NDArray[np.floating],
    qdot0: NDArray[np.floating],
    n_steps: int = 100,
    dt: float = 0.01,
    joint_limits: tuple[NDArray[np.floating], NDArray[np.floating]] = JOINT_LIMITS_RAD,
) -> tuple[NDArray[np.floating], NDArray[np.floating], bool]:
    """Execute closed-loop rollout of trained policy."""
    require(len(q0) == 4 and len(qdot0) == 4, "4-DOF state required", None)
    q_sim, qdot_sim = (
        np.zeros((n_steps, 4), dtype=float),
        np.zeros((n_steps, 4), dtype=float),
    )
    q_sim[0], qdot_sim[0] = q0, qdot0
    q_min, q_max = joint_limits
    within_limits = True

    for step in range(n_steps - 1):
        obs = np.concatenate([q_sim[step], qdot_sim[step]])
        tau_pred = policy.predict(obs)
        qddot = (tau_pred - DAMPING * qdot_sim[step]) / M_DIAG
        qdot_next = qdot_sim[step] + dt * qddot
        q_next = q_sim[step] + dt * qdot_next
        if np.any(q_next < q_min) or np.any(q_next > q_max):
            within_limits = False
        qdot_sim[step + 1] = qdot_next
        q_sim[step + 1] = q_next

    return q_sim, qdot_sim, within_limits


def evaluate_baseline_seeds(
    fixture: DemonstrationFixture,
    seeds: Sequence[int] = (42, 43, 44),
    epochs: int = 150,
) -> BaselineEvaluationReport:
    """Evaluate 64x64 MLP baseline across multiple seeds."""
    require(len(seeds) >= 3, "at least 3 seeds required", None)
    seed_results: list[SeedResult] = []
    v_demo = float(np.max(np.abs(fixture.joint_velocities @ LINK_LENGTHS_M)))

    for s in seeds:
        policy, hist = train_bc_64x64_baseline(fixture.dataset, seed=s, epochs=epochs)
        q_sim, qdot_sim, ok = simulate_closed_loop_rollout(
            policy,
            fixture.joint_positions[0],
            fixture.joint_velocities[0],
            n_steps=len(fixture.timestamps),
        )
        rmse = float(np.sqrt(np.mean((q_sim - fixture.joint_positions) ** 2)))
        v_sim = float(np.max(np.abs(qdot_sim @ LINK_LENGTHS_M)))
        seed_results.append(
            SeedResult(
                s,
                float(hist["train_loss"][0]),
                float(hist["train_loss"][-1]),
                rmse,
                float(abs(v_sim - v_demo)),
                ok,
            )
        )

    rmses = [r.tracking_rmse for r in seed_results]
    speed_errs = [r.clubhead_speed_error for r in seed_results]
    manifest = {
        "manifest_version": "v1",
        "issue": 14477,
        "run_record": "#14436/manifest_v1",
        "experiment_contract": "#14476/spec_v1",
        "fixture_sha256": fixture.sha256_digest,
        "seeds": list(seeds),
        "mean_tracking_rmse": float(np.mean(rmses)),
        "var_tracking_rmse": float(np.var(rmses)),
        "mean_speed_error": float(np.mean(speed_errs)),
        "var_speed_error": float(np.var(speed_errs)),
    }
    return BaselineEvaluationReport(
        seeds=tuple(seeds),
        mean_tracking_rmse=float(np.mean(rmses)),
        var_tracking_rmse=float(np.var(rmses)),
        mean_speed_error=float(np.mean(speed_errs)),
        var_speed_error=float(np.var(speed_errs)),
        all_within_limits=all(r.within_joint_limits for r in seed_results),
        seed_results=tuple(seed_results),
        manifest=manifest,
    )
