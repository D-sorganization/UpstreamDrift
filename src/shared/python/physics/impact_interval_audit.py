"""Fail-closed gate over the pinned Tools impact-interval energy audit (#9548).

This module owns no contact physics. The pinned Tools solver
(``shared.python.swing_sim.impact_interval``, Tools #4130 / #5079) integrates
every ledger term - recoverable normal spring energy, unilateral release at an
identified tensile-clip event, dashpot/friction and torsional-grip damping
losses, boundary storage - independently of the signed residual. UpstreamDrift
re-derives that residual from the reported terms, compares every diagnostic
against declared tolerances, and refuses to release a qualified post-impact
state when contact never separated or the numerical audit fails.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

IMPACT_INTERVAL_CONTRACT_MODULE = "shared.python.swing_sim.impact_interval"

#: Limitations of the audit evidence, surfaced verbatim in every report.
IMPACT_AUDIT_LIMITATIONS: tuple[str, ...] = (
    "Residual is O(dt) semi-implicit Euler truncation; the declared tolerance "
    "is justified only by the halving-dt convergence check in "
    "tests/shared_contracts/test_impact_interval_provider.py.",
    "Recoverable contact energy is the normal Kelvin-Voigt spring term only; "
    "the tangential branch is purely dissipative.",
    "The fixed attachment of supported boundaries performs no work but carries "
    "reaction impulse, so linear momentum is audited only for FREE and the "
    "angular balance about the attachment only for supported boundaries.",
    "Closure is software correctness only; it is not physical qualification.",
)

_LEDGER_TERMS = (
    "initial_kinetic_energy_j",
    "final_kinetic_energy_j",
    "dissipated_energy_j",
    "dashpot_and_friction_dissipation_j",
    "unilateral_release_energy_j",
    "boundary_stored_energy_j",
    "energy_residual_j",
    "linear_momentum_residual_n_s",
    "stored_contact_energy_initial_j",
    "stored_contact_energy_final_j",
    "torsional_damping_dissipation_j",
)


@dataclass(frozen=True)
class ImpactAuditTolerances:
    """Declared closure tolerances for one interval solve (SI units)."""

    energy_residual_j: float = 0.15
    linear_momentum_residual_n_s: float = 1.0e-8
    supported_momentum_residual_n_m_s: float = 1.0e-3
    ledger_closure_j: float = 1.0e-9

    def __post_init__(self) -> None:
        for name in (
            "energy_residual_j",
            "linear_momentum_residual_n_s",
            "supported_momentum_residual_n_m_s",
            "ledger_closure_j",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class ImpactAuditVerdict:
    """Outcome of auditing one solve against declared tolerances.

    ``qualified`` is True only when contact separated, the reported residual
    equals the residual recomputed from the independent ledger terms, and
    every residual is within tolerance. ``failures`` lists each violated
    check; ``limitations`` are the standing caveats of the evidence.
    """

    qualified: bool
    contact_completed: bool
    termination: str
    energy_residual_j: float
    recomputed_energy_residual_j: float
    linear_momentum_residual_n_s: float
    supported_momentum_residual_n_m_s: float
    stored_contact_energy_final_j: float
    unilateral_release_energy_j: float
    dissipated_energy_j: float
    failures: tuple[str, ...]
    limitations: tuple[str, ...] = IMPACT_AUDIT_LIMITATIONS

    def __post_init__(self) -> None:
        if self.qualified and self.failures:
            raise ValueError("a qualified verdict cannot carry failures")
        if not self.qualified and not self.failures:
            raise ValueError("an unqualified verdict must name a failure")

    def to_report(self) -> dict[str, Any]:
        """Return a JSON-serialisable record for UI/report surfaces."""
        return {
            "qualified": self.qualified,
            "contact_completed": self.contact_completed,
            "termination": self.termination,
            "energy_residual_j": self.energy_residual_j,
            "recomputed_energy_residual_j": self.recomputed_energy_residual_j,
            "linear_momentum_residual_n_s": self.linear_momentum_residual_n_s,
            "supported_momentum_residual_n_m_s": (
                None
                if math.isnan(self.supported_momentum_residual_n_m_s)
                else self.supported_momentum_residual_n_m_s
            ),
            "stored_contact_energy_final_j": self.stored_contact_energy_final_j,
            "unilateral_release_energy_j": self.unilateral_release_energy_j,
            "dissipated_energy_j": self.dissipated_energy_j,
            "failures": list(self.failures),
            "limitations": list(self.limitations),
        }


class ImpactAuditError(ValueError):
    """Raised when qualified output is requested from a failed audit."""

    def __init__(self, verdict: ImpactAuditVerdict) -> None:
        self.verdict = verdict
        super().__init__(
            "impact-interval numerical audit failed; refusing qualified output: "
            + "; ".join(verdict.failures)
        )


def _require_audit(result: object) -> Any:
    audit = getattr(result, "audit", None)
    if audit is None:
        raise TypeError("result must carry an impact-interval audit")
    for name in (*_LEDGER_TERMS, "supported_momentum_residual_n_m_s"):
        value = getattr(audit, name, None)
        if value is None:
            raise TypeError(f"audit is missing ledger term {name}")
        if name != "supported_momentum_residual_n_m_s" and not math.isfinite(value):
            raise ValueError(f"audit ledger term {name} must be finite")
    return audit


def recompute_energy_residual(audit: Any) -> float:
    """Re-derive the signed residual from the independently reported terms.

    The identity is initial kinetic + initial stored contact energy against
    final kinetic + final stored contact energy + unilateral release +
    dissipation + boundary storage. No balancing term is introduced here.
    """
    return float(
        audit.initial_kinetic_energy_j
        - audit.final_kinetic_energy_j
        + (audit.stored_contact_energy_initial_j - audit.stored_contact_energy_final_j)
        - audit.unilateral_release_energy_j
        - audit.dissipated_energy_j
        - audit.boundary_stored_energy_j
    )


def audit_impact_interval(
    result: Any,
    tolerances: ImpactAuditTolerances | None = None,
) -> ImpactAuditVerdict:
    """Audit one Tools ``ImpactIntervalResult`` against declared tolerances.

    Preconditions:
        ``result`` exposes ``audit``, ``termination`` and ``contact_completed``
        as the pinned Tools façade does; every ledger term is finite.
    Postconditions:
        The verdict reports the provider residual unchanged alongside the
        residual recomputed from the ledger terms, and is ``qualified`` only
        when ``failures`` is empty.
    """
    tolerances = ImpactAuditTolerances() if tolerances is None else tolerances
    if not isinstance(tolerances, ImpactAuditTolerances):
        raise TypeError("tolerances must be an ImpactAuditTolerances")
    audit = _require_audit(result)
    termination = getattr(result, "termination", None)
    if termination is None or not hasattr(result, "contact_completed"):
        raise TypeError("result must expose termination and contact_completed")
    contact_completed = bool(result.contact_completed)
    termination_name = str(getattr(termination, "name", termination))
    recomputed = recompute_energy_residual(audit)
    failures: list[str] = []
    if not contact_completed:
        failures.append(
            f"contact did not separate (termination={termination_name}); "
            f"{audit.stored_contact_energy_final_j:.6g} J remain stored in the "
            "contact spring"
        )
    if abs(audit.energy_residual_j - recomputed) > tolerances.ledger_closure_j:
        failures.append(
            "reported energy residual "
            f"{audit.energy_residual_j:.6g} J does not equal the residual "
            f"recomputed from the ledger terms {recomputed:.6g} J"
        )
    if (
        abs(
            audit.dissipated_energy_j
            - (
                audit.dashpot_and_friction_dissipation_j
                + audit.torsional_damping_dissipation_j
            )
        )
        > tolerances.ledger_closure_j
    ):
        failures.append(
            "dissipated energy is not the sum of dashpot/friction and torsional "
            "damping losses"
        )
    if abs(audit.energy_residual_j) > tolerances.energy_residual_j:
        failures.append(
            f"|energy residual| {abs(audit.energy_residual_j):.6g} J exceeds "
            f"{tolerances.energy_residual_j:.6g} J"
        )
    # Momentum diagnostics are boundary-specific: a fixed attachment carries
    # reaction impulse, so only the FREE boundary (supported residual NaN)
    # owes a linear-momentum balance; supported boundaries owe the angular
    # balance about the attachment instead.
    supported = float(audit.supported_momentum_residual_n_m_s)
    if (
        math.isnan(supported)
        and audit.linear_momentum_residual_n_s > tolerances.linear_momentum_residual_n_s
    ):
        failures.append(
            "free linear-momentum residual "
            f"{audit.linear_momentum_residual_n_s:.6g} N s exceeds "
            f"{tolerances.linear_momentum_residual_n_s:.6g} N s"
        )
    elif (
        not math.isnan(supported)
        and supported > tolerances.supported_momentum_residual_n_m_s
    ):
        failures.append(
            f"supported angular-momentum residual {supported:.6g} N m s exceeds "
            f"{tolerances.supported_momentum_residual_n_m_s:.6g} N m s"
        )
    return ImpactAuditVerdict(
        qualified=not failures,
        contact_completed=contact_completed,
        termination=termination_name,
        energy_residual_j=float(audit.energy_residual_j),
        recomputed_energy_residual_j=recomputed,
        linear_momentum_residual_n_s=float(audit.linear_momentum_residual_n_s),
        supported_momentum_residual_n_m_s=supported,
        stored_contact_energy_final_j=float(audit.stored_contact_energy_final_j),
        unilateral_release_energy_j=float(audit.unilateral_release_energy_j),
        dissipated_energy_j=float(audit.dissipated_energy_j),
        failures=tuple(failures),
    )


def qualified_post_impact_state(
    result: Any,
    tolerances: ImpactAuditTolerances | None = None,
) -> Any:
    """Return ``result.to_post_impact_state()`` only when the audit qualifies.

    Raises:
        ImpactAuditError: when contact did not separate or any residual is
            outside tolerance. The verdict is attached for reporting.
    """
    verdict = audit_impact_interval(result, tolerances)
    if not verdict.qualified:
        raise ImpactAuditError(verdict)
    return result.to_post_impact_state()


__all__ = [
    "IMPACT_AUDIT_LIMITATIONS",
    "IMPACT_INTERVAL_CONTRACT_MODULE",
    "ImpactAuditError",
    "ImpactAuditTolerances",
    "ImpactAuditVerdict",
    "audit_impact_interval",
    "qualified_post_impact_state",
    "recompute_energy_residual",
]
