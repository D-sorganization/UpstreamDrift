"""Verified inference orchestration with safe classical fallback (NM-08)."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.neural_motion.inference.distribution import (
    DistributionBounds,
    check_target_distribution,
)
from src.shared.python.neural_motion.inference.types import (
    INFERENCE_SCHEMA,
    AttemptRecord,
    DomainCheckResult,
    InferenceBudget,
    InferenceStatus,
    VerifiedInferenceReport,
)

if TYPE_CHECKING:
    from src.shared.python.motion_matching.hybrid import ProposalCheckpointContract

__all__ = [
    "VerifiedInferenceOrchestrator",
]

logger = get_logger(__name__)

ProposalFn = Callable[[Any], np.ndarray]
PolishFn = Callable[[Any, np.ndarray], dict[str, Any]]
ClassicalFn = Callable[[Any, float], dict[str, Any]]


class VerifiedInferenceOrchestrator:
    """Orchestrates neural proposal -> refinement -> acceptance with classical fallback."""

    def __init__(
        self,
        *,
        proposal_fn: ProposalFn | None = None,
        polish_fn: PolishFn | None = None,
        classical_fn: ClassicalFn | None = None,
        expected_contract: ProposalCheckpointContract | None = None,
        loaded_contract: ProposalCheckpointContract | None = None,
        distribution_bounds: DistributionBounds | None = None,
    ) -> None:
        self._proposal_fn = proposal_fn
        self._polish_fn = polish_fn
        self._classical_fn = classical_fn
        self._expected_contract = expected_contract
        self._loaded_contract = loaded_contract
        self._bounds = distribution_bounds or DistributionBounds()

    def orchestrate(
        self,
        target: Any,
        *,
        total_budget_s: float = 10.0,
        require_independent_replay: bool = True,
        is_preview: bool = False,
    ) -> VerifiedInferenceReport:
        budget = InferenceBudget(total_budget_s=total_budget_s)
        domain_check = check_target_distribution(target, self._bounds)
        attempts: list[AttemptRecord] = []

        if not domain_check.is_in_distribution:
            reason = f"Out-of-distribution: {'; '.join(domain_check.diagnostics)}"
            attempts.append(
                self._make_rejected_attempt("distribution_gate", reason, budget)
            )
            return self._fallback_or_reject(
                target, budget, attempts, domain_check, is_preview
            )

        neural_att, neural_ok = self._try_neural_pipeline(
            target, budget, require_independent_replay
        )
        attempts.append(neural_att)

        if neural_ok:
            status = InferenceStatus.NEURAL_ACCEPTED
            return VerifiedInferenceReport(
                schema=INFERENCE_SCHEMA,
                status=status,
                selected_controls=neural_att.controls,
                attempts=tuple(attempts),
                domain_check=domain_check,
                acceptance_verdict={"is_physically_accepted": True, "status": "PASSED"},
                is_preview_only=is_preview,
                duration_s=budget.elapsed_s(),
            )

        return self._fallback_or_reject(
            target, budget, attempts, domain_check, is_preview
        )

    def _make_rejected_attempt(
        self, phase: str, reason: str, budget: InferenceBudget
    ) -> AttemptRecord:
        return AttemptRecord(
            phase=phase,
            controls=None,
            cost=None,
            independent_replay=False,
            acceptance_status="rejected",
            duration_s=budget.elapsed_s(),
            rejection_reason=reason,
        )

    def _try_neural_pipeline(
        self,
        target: Any,
        budget: InferenceBudget,
        require_replay: bool,
    ) -> tuple[AttemptRecord, bool]:
        t0 = time.perf_counter()
        if self._proposal_fn is None:
            return AttemptRecord(
                phase="neural_proposal",
                controls=None,
                cost=None,
                independent_replay=False,
                acceptance_status="rejected",
                duration_s=time.perf_counter() - t0,
                rejection_reason="Missing checkpoint / proposal function",
            ), False

        if self._expected_contract is not None and self._loaded_contract is not None:
            from src.shared.python.motion_matching.hybrid import (
                assert_proposal_checkpoint_compatible,
            )

            try:
                assert_proposal_checkpoint_compatible(
                    expected=self._expected_contract,
                    loaded=self._loaded_contract,
                )
            except ValueError as err:
                return AttemptRecord(
                    phase="neural_proposal",
                    controls=None,
                    cost=None,
                    independent_replay=False,
                    acceptance_status="rejected",
                    duration_s=time.perf_counter() - t0,
                    rejection_reason=f"Incompatible checkpoint contract: {err}",
                ), False

        try:
            raw_controls = self._proposal_fn(target)
            controls = np.asarray(raw_controls, dtype=np.float64).reshape(-1)
        except Exception as err:
            return AttemptRecord(
                phase="neural_proposal",
                controls=None,
                cost=None,
                independent_replay=False,
                acceptance_status="rejected",
                duration_s=time.perf_counter() - t0,
                rejection_reason=f"Proposal generation error: {err}",
            ), False

        if controls.size == 0 or not np.all(np.isfinite(controls)):
            return AttemptRecord(
                phase="neural_proposal",
                controls=controls,
                cost=None,
                independent_replay=False,
                acceptance_status="rejected",
                duration_s=time.perf_counter() - t0,
                rejection_reason="Proposal returned non-finite / NaN controls",
            ), False

        if (
            self._expected_contract
            and controls.size != self._expected_contract.control_dim
        ):
            return AttemptRecord(
                phase="neural_proposal",
                controls=controls,
                cost=None,
                independent_replay=False,
                acceptance_status="rejected",
                duration_s=time.perf_counter() - t0,
                rejection_reason=(
                    f"Dimension mismatch: expected {self._expected_contract.control_dim}, "
                    f"got {controls.size}"
                ),
            ), False

        return self._evaluate_neural_polish(target, controls, t0, require_replay)

    def _evaluate_neural_polish(
        self,
        target: Any,
        controls: np.ndarray,
        t0: float,
        require_replay: bool,
    ) -> tuple[AttemptRecord, bool]:
        if self._polish_fn is None:
            return AttemptRecord(
                phase="neural_proposal",
                controls=controls,
                cost=0.0,
                independent_replay=False,
                acceptance_status="passed",
                duration_s=time.perf_counter() - t0,
            ), True

        polish_res = self._polish_fn(target, controls)
        pol_controls = polish_res.get("controls", controls)
        independent = bool(polish_res.get("independent_replay", False))
        cost = float(polish_res.get("final_loss", 0.0))
        acc = polish_res.get("acceptance", {})
        acc_passed = bool(acc.get("is_physically_accepted", True))
        acc_reason = str(acc.get("reason", ""))

        if require_replay and not independent:
            return AttemptRecord(
                phase="neural_refined",
                controls=pol_controls,
                cost=cost,
                independent_replay=False,
                acceptance_status="failed",
                duration_s=time.perf_counter() - t0,
                rejection_reason="Missing independent forward replay",
            ), False

        if not acc_passed:
            return AttemptRecord(
                phase="neural_refined",
                controls=pol_controls,
                cost=cost,
                independent_replay=independent,
                acceptance_status="failed",
                duration_s=time.perf_counter() - t0,
                rejection_reason=f"Physical acceptance failed: {acc_reason}",
            ), False

        return AttemptRecord(
            phase="neural_refined",
            controls=pol_controls,
            cost=cost,
            independent_replay=independent,
            acceptance_status="passed",
            duration_s=time.perf_counter() - t0,
        ), True

    def _fallback_or_reject(
        self,
        target: Any,
        budget: InferenceBudget,
        attempts: list[AttemptRecord],
        domain_check: DomainCheckResult,
        is_preview: bool,
    ) -> VerifiedInferenceReport:
        rem_budget = budget.remaining_s()
        if rem_budget <= 0.0 or self._classical_fn is None:
            reason = (
                "Compute budget exhausted"
                if rem_budget <= 0.0
                else "No classical solver"
            )
            attempts.append(
                self._make_rejected_attempt("classical_fallback", reason, budget)
            )
            return VerifiedInferenceReport(
                schema=INFERENCE_SCHEMA,
                status=InferenceStatus.REJECTED,
                selected_controls=None,
                attempts=tuple(attempts),
                domain_check=domain_check,
                acceptance_verdict={
                    "is_physically_accepted": False,
                    "status": "REJECTED",
                },
                is_preview_only=is_preview,
                duration_s=budget.elapsed_s(),
            )

        t_classic = time.perf_counter()
        c_res = self._classical_fn(target, rem_budget)
        c_controls = c_res.get("controls")
        c_ind = bool(c_res.get("independent_replay", False))
        c_cost = float(c_res.get("final_loss", 0.0))
        c_acc = c_res.get("acceptance", {})
        c_passed = bool(c_acc.get("is_physically_accepted", True))

        attempts.append(
            AttemptRecord(
                phase="classical_fallback",
                controls=c_controls,
                cost=c_cost,
                independent_replay=c_ind,
                acceptance_status="passed" if c_passed else "failed",
                duration_s=time.perf_counter() - t_classic,
                rejection_reason="" if c_passed else str(c_acc.get("reason", "failed")),
            )
        )

        status = (
            InferenceStatus.CLASSICAL_FALLBACK if c_passed else InferenceStatus.REJECTED
        )
        return VerifiedInferenceReport(
            schema=INFERENCE_SCHEMA,
            status=status,
            selected_controls=c_controls if c_passed else None,
            attempts=tuple(attempts),
            domain_check=domain_check,
            acceptance_verdict=c_acc if c_passed else {"is_physically_accepted": False},
            is_preview_only=is_preview,
            duration_s=budget.elapsed_s(),
        )
