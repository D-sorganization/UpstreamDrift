"""Preregistered serial study contract for forward attribution (#9153)."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np

from scripts.research.proximal_distal_energy.articulated_forward_contract import (
    registered_variants,
)

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ForwardAttributionStudyPlan:
    """Immutable identity, estimands, and promotion rules for a serial run."""

    source_revision: str
    source_data_sha256: str
    duration_s: float = 0.005
    time_steps_s: tuple[float, ...] = (0.001, 0.0005, 0.00025)
    engines: tuple[str, ...] = ("mujoco", "pinocchio")
    worker_count: int = 1
    random_seed: int = 9153
    momentum_relative_tolerance: float = 0.02
    work_relative_tolerance: float = 0.01
    refinement_ratio_limit: float = 0.8
    gap_tolerance_m: float = 1.0e-10
    event_time_tolerance_s: float = 1.0e-12
    smoke_states: tuple[tuple[int, int, float], ...] = ((4, 6, 0.12),)
    screening_case_indices: tuple[int, ...] = (0, 4, 8, 9, 13, 17)
    screening_sample_indices: tuple[int, ...] = (0, 6, 12)

    def __post_init__(self) -> None:
        if not _SHA40.fullmatch(self.source_revision):
            raise ValueError("source_revision must be one lowercase 40-character SHA")
        if not _SHA256.fullmatch(self.source_data_sha256):
            raise ValueError("source_data_sha256 must be one lowercase SHA-256")
        if self.worker_count != 1:
            raise ValueError("worker_count must remain one for the local serial smoke")
        if not isinstance(self.random_seed, int) or self.random_seed < 0:
            raise ValueError("random_seed must be a nonnegative integer")
        if self.engines != ("mujoco", "pinocchio"):
            raise ValueError(
                "engines must preserve the registered native-operator order"
            )
        steps = np.asarray(self.time_steps_s, dtype=np.float64)
        if (
            steps.ndim != 1
            or steps.size < 3
            or np.any(~np.isfinite(steps))
            or np.any(steps <= 0.0)
            or np.any(np.diff(steps) >= 0.0)
            or not np.allclose(
                self.duration_s / steps, np.rint(self.duration_s / steps)
            )
        ):
            raise ValueError(
                "time_steps_s must contain at least three decreasing divisors of duration_s"
            )
        for name in (
            "duration_s",
            "momentum_relative_tolerance",
            "work_relative_tolerance",
            "refinement_ratio_limit",
            "gap_tolerance_m",
            "event_time_tolerance_s",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 < self.refinement_ratio_limit < 1.0:
            raise ValueError("refinement_ratio_limit must lie in (0, 1)")
        if self.smoke_states != ((4, 6, 0.12),):
            raise ValueError("smoke_states must retain the preregistered source state")
        if self.screening_case_indices != (0, 4, 8, 9, 13, 17):
            raise ValueError("screening_case_indices must retain the prior authority")
        if self.screening_sample_indices != (0, 6, 12):
            raise ValueError("screening_sample_indices must retain the prior authority")

    def to_manifest(self) -> dict[str, Any]:
        """Return the JSON-serializable preregistration contract."""

        return {
            "schema_version": "1.2.0",
            "issue": 9153,
            "parent_epic": 8557,
            "preregistration": {
                "revision": 3,
                "amendment_timing": "before_registered_outcome_generation",
                "amendment": (
                    "freeze the smoke source state and declare the later "
                    "screening population"
                ),
            },
            "identity": {
                "source_revision": self.source_revision,
                "source_data_sha256": self.source_data_sha256,
                "random_seed": self.random_seed,
            },
            "execution": {
                "worker_count": self.worker_count,
                "case_checkpointing": "atomic_per_case",
                "resume_identity": (
                    "source_revision+source_data_sha256+schema_version+"
                    "execution_revision+case_key"
                ),
                "eligible_machine": "serial_cpu_runtime_with_qualified_native_engine",
            },
            "design": {
                "duration_s": self.duration_s,
                "time_steps_s": list(self.time_steps_s),
                "engines": list(self.engines),
                "variants": [variant.name for variant in registered_variants()],
                "smoke_states": [
                    {
                        "source_case_index": case_index,
                        "source_sample_index": sample_index,
                        "source_time_s": source_time_s,
                        "role": "runtime_and_pipeline_qualification_only",
                    }
                    for case_index, sample_index, source_time_s in self.smoke_states
                ],
                "screening_case_indices": list(self.screening_case_indices),
                "screening_sample_indices": list(self.screening_sample_indices),
                "event_path_model": "linear_state_interpolant",
                "event_kinds": ["opening", "reattachment", "stick", "slip"],
            },
            "estimands": {
                "same_trajectory_attribution": (
                    "integrals evaluated on one retained realized state history"
                ),
                "forward_counterfactual": (
                    "outcome difference between separately integrated matched interventions"
                ),
                "momentum_balance": "delta(Mv)=integral(sum(Q)+Mdot*v)dt+sum(J)",
                "generalized_work": "integral(v_transpose*Q)dt plus event work",
                "outcomes": [
                    "clubhead_speed",
                    "clubhead_direction",
                    "face_path_proxy",
                    "contact_load",
                    "event_timing",
                ],
            },
            "tolerances": {
                "momentum_relative": self.momentum_relative_tolerance,
                "work_relative": self.work_relative_tolerance,
                "refinement_ratio_limit": self.refinement_ratio_limit,
                "event_gap_m": self.gap_tolerance_m,
                "event_time_s": self.event_time_tolerance_s,
            },
            "promotion": {
                "all_retained_cases_close": True,
                "all_native_operators_available": True,
                "typed_failures_retained": True,
                "same_trajectory_and_counterfactual_separate": True,
                "human_or_coaching_claims": False,
                "incomplete_controltower_checkpoints_allowed": False,
            },
        }


@dataclass(frozen=True, slots=True)
class ClosureRefinementAssessment:
    """Three-or-more-resolution closure and contraction result."""

    momentum_refinement_ratios: tuple[float, ...]
    work_refinement_ratios: tuple[float, ...]
    failure_codes: tuple[str, ...]

    @property
    def passes(self) -> bool:
        """Return whether every preregistered closure/refinement gate passed."""

        return not self.failure_codes


def assess_closure_refinement(
    *,
    time_steps_s: tuple[float, ...],
    momentum_relative_residuals: tuple[float, ...],
    work_relative_residuals: tuple[float, ...],
    momentum_tolerance: float,
    work_tolerance: float,
    refinement_ratio_limit: float,
) -> ClosureRefinementAssessment:
    """Apply frozen closure and residual-contraction gates."""

    steps = np.asarray(time_steps_s, dtype=np.float64)
    momentum = np.asarray(momentum_relative_residuals, dtype=np.float64)
    work = np.asarray(work_relative_residuals, dtype=np.float64)
    if (
        steps.ndim != 1
        or steps.size < 3
        or momentum.shape != steps.shape
        or work.shape != steps.shape
    ):
        raise ValueError("steps and residuals must share a one-dimensional length >= 3")
    if (
        np.any(~np.isfinite(steps))
        or np.any(~np.isfinite(momentum))
        or np.any(~np.isfinite(work))
        or np.any(steps <= 0.0)
        or np.any(momentum < 0.0)
        or np.any(work < 0.0)
        or np.any(np.diff(steps) >= 0.0)
    ):
        raise ValueError(
            "steps must decrease and all residuals must be finite/nonnegative"
        )
    for name, value in (
        ("momentum_tolerance", momentum_tolerance),
        ("work_tolerance", work_tolerance),
        ("refinement_ratio_limit", refinement_ratio_limit),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not 0.0 < refinement_ratio_limit < 1.0:
        raise ValueError("refinement_ratio_limit must lie in (0, 1)")
    if np.any(momentum[:-1] == 0.0) or np.any(work[:-1] == 0.0):
        raise ValueError("zero coarse residual prevents a refinement ratio")
    momentum_ratios = tuple(float(value) for value in momentum[1:] / momentum[:-1])
    work_ratios = tuple(float(value) for value in work[1:] / work[:-1])
    failures: list[str] = []
    if np.any(momentum > momentum_tolerance):
        failures.append("momentum_closure")
    if np.any(work > work_tolerance):
        failures.append("work_closure")
    if any(value > refinement_ratio_limit for value in momentum_ratios):
        failures.append("momentum_refinement")
    if any(value > refinement_ratio_limit for value in work_ratios):
        failures.append("work_refinement")
    return ClosureRefinementAssessment(
        momentum_refinement_ratios=momentum_ratios,
        work_refinement_ratios=work_ratios,
        failure_codes=tuple(failures),
    )


__all__ = [
    "ClosureRefinementAssessment",
    "ForwardAttributionStudyPlan",
    "assess_closure_refinement",
]
