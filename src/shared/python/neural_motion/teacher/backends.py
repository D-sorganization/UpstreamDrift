"""Teacher rollout backends — mock for software contracts; native adapters elsewhere."""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import numpy as np

from src.shared.python.dataset_tools.canonical import CANONICAL_JOINTS, N_JOINTS

from .types import TeacherRolloutRequest, TeacherRolloutResult

__all__ = ["MockTeacherRolloutBackend", "TeacherRolloutBackend"]


@runtime_checkable
class TeacherRolloutBackend(Protocol):
    """Roll out a teacher trajectory and independent replay digest."""

    def rollout(self, request: TeacherRolloutRequest) -> TeacherRolloutResult:
        """Return feasible or rejected rollout with accounting metadata."""


class MockTeacherRolloutBackend:
    """Software-contract backend — not native physical evidence."""

    def __init__(
        self,
        *,
        infeasible_scale: float = 0.35,
        zero_channel: str | None = None,
    ) -> None:
        if not np.isfinite(infeasible_scale) or infeasible_scale <= 0.0:
            raise ValueError("infeasible_scale must be a positive finite value")
        self._infeasible_scale = float(infeasible_scale)
        self._zero_channel = zero_channel

    def rollout(self, request: TeacherRolloutRequest) -> TeacherRolloutResult:
        norm = float(np.linalg.norm(request.perturbation))
        cost = 1.0 + 0.1 * norm
        if norm > self._infeasible_scale:
            digest = hashlib.sha256(
                f"reject:{request.attempt_key()}".encode()
            ).hexdigest()
            return TeacherRolloutResult(
                feasible=False,
                teacher_objective=norm,
                convergence_iterations=0,
                independent_replay_digest=digest,
                simulation_cost_units=cost,
                rejection_reason="perturbation_exceeds_feasible_envelope",
            )

        rng = np.random.default_rng(abs(hash(request.attempt_key())) % (2**32))
        n_t = request.n_samples
        times = np.linspace(0.0, request.duration_s, n_t)
        q = np.tile(request.anchor.q0[:N_JOINTS], (n_t, 1))
        if q.shape[1] < N_JOINTS:
            pad = np.zeros((n_t, N_JOINTS - q.shape[1]), dtype=np.float64)
            q = np.concatenate([q, pad], axis=1)
        q = q + 0.01 * request.perturbation[0] + 0.001 * rng.normal(size=q.shape)
        v = rng.normal(scale=0.05, size=q.shape)
        u = rng.normal(scale=0.02, size=q.shape)
        a_native = rng.normal(scale=0.01, size=q.shape)
        q_next = np.roll(q, -1, axis=0)
        q_next[-1] = q[-1]

        availability = {
            "q": "available",
            "v": "available",
            "u": "available",
            "a_native": "available",
            "q_next": "available",
        }
        if self._zero_channel is not None:
            availability[self._zero_channel] = "available"
            if self._zero_channel == "a_native":
                a_native = np.zeros_like(a_native)

        replay_digest = hashlib.sha256(q.tobytes()).hexdigest()
        objective = float(norm + 0.01 * np.mean(np.abs(u)))

        return TeacherRolloutResult(
            feasible=True,
            teacher_objective=objective,
            convergence_iterations=max(1, int(norm * 10)),
            independent_replay_digest=replay_digest,
            simulation_cost_units=cost,
            channel_availability=availability,
            sample_times_s=times,
            q=q,
            v=v,
            u=u,
            a_native=a_native,
            q_next=q_next,
        )
