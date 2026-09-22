"""Teacher episode generation from baselines and structured perturbations."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable

import numpy as np

from src.shared.python.dataset_tools.canonical import N_JOINTS
from src.shared.python.neural_motion.episodes.record import (
    EPISODE_STORE_SCHEMA,
    EpisodeRecord,
)

from .types import PerturbationKind, TeacherOutcome, TeacherSpec

__all__ = ["TeacherEpisodeGenerator", "TeacherRolloutFn"]

TeacherRolloutFn = Callable[[TeacherSpec, EpisodeRecord], EpisodeRecord]

# Channels the synthetic / baseline-derived generator can actually fill.
_SUPPLIABLE = frozenset({"q", "v", "u", "a_native", "q_next"})

_BASE_COST = 1.0


def _finite_norm(values: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(values, dtype=np.float64)))


def _perturb_scale(kind: PerturbationKind, seed: int) -> float:
    """Deterministic amplitude keyed by kind (random_torque is largest)."""
    table = {
        PerturbationKind.NEAR_BASELINE: 0.02,
        PerturbationKind.STRATIFIED: 0.08,
        PerturbationKind.LOW_DISCREPANCY: 0.05,
        PerturbationKind.RANDOM_TORQUE: 0.35,
    }
    # Tiny seed-dependent jitter so kinds stay distinct but reproducible.
    return table[kind] * (1.0 + 1e-6 * (seed % 97))


def _low_discrepancy_offsets(n_t: int, n_j: int, seed: int) -> np.ndarray:
    """Van der Corput-style 2-D offsets (software stand-in for Sobol)."""
    out = np.empty((n_t, n_j), dtype=np.float64)
    for t in range(n_t):
        for j in range(n_j):
            # Radical-inverse base-2 on a mixed index.
            idx = ((t + 1) * (j + 3) + seed) & 0xFFFFFFFF
            value = 0.0
            f = 0.5
            n = idx
            while n > 0:
                value += f * (n & 1)
                n >>= 1
                f *= 0.5
            out[t, j] = 2.0 * value - 1.0
    return out


def _default_rollout(spec: TeacherSpec, baseline: EpisodeRecord) -> EpisodeRecord:
    """Structured perturbation of a baseline episode (software contract path)."""
    n_t = int(baseline.sample_times_s.shape[0])
    scale = _perturb_scale(spec.perturbation, spec.seed)
    rng = np.random.default_rng(spec.seed)

    if spec.perturbation is PerturbationKind.LOW_DISCREPANCY:
        delta_u = scale * _low_discrepancy_offsets(n_t, N_JOINTS, spec.seed)
        delta_q = 0.25 * scale * _low_discrepancy_offsets(n_t, N_JOINTS, spec.seed + 1)
    elif spec.perturbation is PerturbationKind.STRATIFIED:
        # Stratify by time tertiles with distinct offsets.
        delta_u = np.zeros((n_t, N_JOINTS), dtype=np.float64)
        delta_q = np.zeros((n_t, N_JOINTS), dtype=np.float64)
        cuts = np.array_split(np.arange(n_t), 3)
        for band, idxs in enumerate(cuts):
            if len(idxs) == 0:
                continue
            offset = scale * (band + 1) * rng.normal(size=(len(idxs), N_JOINTS))
            delta_u[idxs] = offset
            delta_q[idxs] = 0.2 * offset
    elif spec.perturbation is PerturbationKind.RANDOM_TORQUE:
        delta_u = scale * rng.normal(size=(n_t, N_JOINTS))
        delta_q = 0.05 * scale * rng.normal(size=(n_t, N_JOINTS))
    else:  # NEAR_BASELINE
        delta_u = scale * rng.normal(size=(n_t, N_JOINTS))
        delta_q = 0.1 * scale * rng.normal(size=(n_t, N_JOINTS))

    assert baseline.u is not None and baseline.v is not None
    assert baseline.a_native is not None and baseline.q_next is not None
    q = np.asarray(baseline.q, dtype=np.float64) + delta_q
    v = np.asarray(baseline.v, dtype=np.float64) + 0.5 * delta_q
    u = np.asarray(baseline.u, dtype=np.float64) + delta_u
    a_native = np.asarray(baseline.a_native, dtype=np.float64) + 0.25 * delta_u
    q_next = np.asarray(baseline.q_next, dtype=np.float64) + delta_q
    times = np.linspace(0.0, float(spec.duration_s), n_t)

    ancestry = tuple(
        dict.fromkeys(
            (
                *baseline.ancestry,
                *spec.ancestry,
                f"perturbation:{spec.perturbation.value}",
            )
        )
    )
    return EpisodeRecord(
        trial_id=f"teacher_{spec.family_id}_{spec.seed}_{spec.perturbation.value}",
        family_id=spec.family_id,
        model_id=spec.model_id,
        control_basis=baseline.control_basis,
        units=baseline.units,
        joint_names=baseline.joint_names,
        coefficient_letters=baseline.coefficient_letters,
        schema_version=EPISODE_STORE_SCHEMA,
        sample_times_s=times,
        q=q,
        v=v,
        u=u,
        a_native=a_native,
        q_next=q_next,
        channel_availability=dict(baseline.channel_availability),
        ancestry=ancestry,
        geometry_stratum=spec.geometry_stratum,
        contact_stratum=spec.contact_stratum,
        club_stratum=spec.club_stratum,
        coefficients=baseline.coefficients,
        source_schema=baseline.source_schema,
    )


def _replay_digest(episode: EpisodeRecord) -> str:
    """Independent native-replay stand-in: hash of (q0, u, times)."""
    payload = {
        "q0": hashlib.sha256(np.ascontiguousarray(episode.q[0]).tobytes()).hexdigest(),
        "u": None
        if episode.u is None
        else hashlib.sha256(np.ascontiguousarray(episode.u).tobytes()).hexdigest(),
        "times": hashlib.sha256(
            np.ascontiguousarray(episode.sample_times_s).tobytes()
        ).hexdigest(),
        "model_id": episode.model_id,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class TeacherEpisodeGenerator:
    """Generate one teacher episode from a baseline and a typed spec.

    Missing requested channels fail closed — never invent zero measurements.
    ``RANDOM_TORQUE`` is retained as a comparison corpus role, not the primary
    sampling path.
    """

    def __init__(
        self,
        *,
        baseline: EpisodeRecord,
        rollout_fn: TeacherRolloutFn | None = None,
        outlier_torque_norm: float | None = None,
        unit_cost: float = _BASE_COST,
    ) -> None:
        if not isinstance(baseline, EpisodeRecord):
            raise TypeError("baseline must be an EpisodeRecord")
        if unit_cost <= 0.0 or not math.isfinite(unit_cost):
            raise ValueError("unit_cost must be a positive finite value")
        if outlier_torque_norm is not None and (
            not math.isfinite(outlier_torque_norm) or outlier_torque_norm <= 0.0
        ):
            raise ValueError("outlier_torque_norm must be positive finite when set")
        self._baseline = baseline
        self._rollout_fn = rollout_fn or _default_rollout
        self._outlier_torque_norm = outlier_torque_norm
        self._unit_cost = float(unit_cost)

    def generate(self, spec: TeacherSpec) -> TeacherOutcome:
        """Run one teacher attempt; return feasible outcome or rejection."""
        if not isinstance(spec, TeacherSpec):
            raise TypeError("spec must be a TeacherSpec")
        missing = [name for name in spec.requested_channels if name not in _SUPPLIABLE]
        if missing:
            return TeacherOutcome(
                feasible=False,
                reason=(
                    f"requested channel unavailable: {missing[0]}; "
                    "refusing to invent zeros"
                ),
                seed=spec.seed,
                rejection_cost=self._unit_cost,
            )
        try:
            episode = self._rollout_fn(spec, self._baseline)
        except (ValueError, TypeError, RuntimeError) as exc:
            return TeacherOutcome(
                feasible=False,
                reason=f"infeasible_dynamics: {exc}",
                seed=spec.seed,
                rejection_cost=self._unit_cost,
            )
        if not isinstance(episode, EpisodeRecord):
            return TeacherOutcome(
                feasible=False,
                reason="rollout did not return EpisodeRecord",
                seed=spec.seed,
                rejection_cost=self._unit_cost,
            )
        for name in spec.requested_channels:
            status = episode.channel_availability.get(name, "unavailable")
            value = getattr(episode, name, None)
            if status != "available" or value is None:
                return TeacherOutcome(
                    feasible=False,
                    reason=(
                        f"requested channel {name!r} missing after rollout; "
                        "refusing to invent zeros"
                    ),
                    seed=spec.seed,
                    rejection_cost=self._unit_cost,
                )

        if episode.u is None or not bool(np.all(np.isfinite(episode.u))):
            return TeacherOutcome(
                feasible=False,
                reason="nonfinite controls",
                seed=spec.seed,
                rejection_cost=self._unit_cost,
            )

        torque_norm = _finite_norm(episode.u)
        if (
            self._outlier_torque_norm is not None
            and torque_norm > self._outlier_torque_norm
        ):
            # Quarantine path: still physically meaningful — do not clip.
            return TeacherOutcome(
                feasible=False,
                reason=f"outlier_quarantined:torque_norm={torque_norm:.6g}",
                seed=spec.seed,
                rejection_cost=self._unit_cost,
                episode=episode,
                corpus_role="quarantine",
                teacher_objective=torque_norm,
                converged=False,
                replay_digest=_replay_digest(episode),
            )

        objective = float(np.mean(np.square(episode.u)))
        role = (
            "comparison"
            if spec.perturbation is PerturbationKind.RANDOM_TORQUE
            else "primary"
        )
        return TeacherOutcome(
            feasible=True,
            reason="accepted",
            seed=spec.seed,
            rejection_cost=0.0,
            episode=episode,
            teacher_objective=objective,
            converged=True,
            replay_digest=_replay_digest(episode),
            corpus_role=role,
        )
