"""Per-segment & system momentum/energy balance evaluator with tolerance policies (#15690)."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from src.shared.python.physics.energy_ledger_contract import (
    UNAVAILABLE,
    ConservationAuditResult,
    EnergyLedgerSeries,
    EnergyLedgerStep,
    LedgerAvailability,
    LedgerScalar,
    UnavailableType,
    audit_energy_ledger,
)

POLICY_SCHEMA_VERSION: str = "1.0.0"
VALID_STATUSES: frozenset[str] = frozenset({"pass", "fail", "unavailable"})
_ENERGY_FLAGS = (
    "kinetic",
    "potential",
    "actuator_work",
    "dissipation",
    "external_work",
)
_ALL_FLAGS = (*_ENERGY_FLAGS, "linear_momentum", "angular_momentum")
_TIME_TOL_S: float = 1e-12
_PRESETS: dict[str, tuple[float, float, float, float, bool, bool, str]] = {
    "strict_analytic": (1e-9, 1e-4, 1e-4, 1e-12, True, True, "Strict analytic gate."),
    "forward_replay": (1e-3, 1e-2, 1e-2, 1e-6, True, True, "Forward replay policy."),
    "inverse_dynamics": (1e-1, 1e-1, 1e-1, 1e-4, False, True, "Inverse dynamics."),
}


def _is_pos_finite(v: object) -> bool:
    """Return True if v is a strictly positive finite real number."""
    return (
        isinstance(v, int | float)
        and not isinstance(v, bool)
        and math.isfinite(v)
        and float(v) > 0.0
    )


def _balance_unavail(av: LedgerAvailability, policy: TolerancePolicy) -> bool:
    """Return True if required energy or momentum components are unavailable."""
    if not all(getattr(av, n) for n in _ENERGY_FLAGS):
        return True
    return policy.require_momentum and (
        not av.linear_momentum or not av.angular_momentum
    )


@dataclass(frozen=True)
class TolerancePolicy:
    """Versioned conservation & segment-sum tolerance policy."""

    name: str
    version: str
    energy_tolerance_j: float
    momentum_tolerance_ns: float
    angular_momentum_tolerance_nms: float
    segment_sum_tolerance_j: float
    require_momentum: bool = False
    require_segment_sum: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        """Validate policy identity, tolerance positivity, and boolean flags."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("policy name must be a non-empty string")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("policy version must be a non-empty string")
        tols = (
            self.energy_tolerance_j,
            self.momentum_tolerance_ns,
            self.angular_momentum_tolerance_nms,
            self.segment_sum_tolerance_j,
        )
        if any(not _is_pos_finite(t) for t in tols):
            raise ValueError("all policy tolerances must be positive finite floats")
        if not isinstance(self.require_momentum, bool) or not isinstance(
            self.require_segment_sum, bool
        ):
            raise TypeError("require_momentum and require_segment_sum must be booleans")

    @property
    def policy_sha256(self) -> str:
        """Return deterministic SHA-256 hex digest of this tolerance policy."""
        payload = {k: v for k, v in asdict(self).items() if k != "description"}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def tolerance_policy_preset(name: str) -> TolerancePolicy:
    """Return a versioned TolerancePolicy for a named preset."""
    spec = _PRESETS.get(name)
    if spec is None:
        opts = ", ".join(sorted(_PRESETS))
        raise ValueError(f"unknown tolerance policy preset {name!r} ({opts})")
    te, tp, tl, ts, rm, rs, desc = spec
    return TolerancePolicy(name, POLICY_SCHEMA_VERSION, te, tp, tl, ts, rm, rs, desc)


@dataclass(frozen=True)
class SegmentEnergyRecord:
    """Named per-segment EnergyLedgerSeries input for balance evaluation."""

    segment_id: str
    series: EnergyLedgerSeries
    segment_name: str = ""

    def __post_init__(self) -> None:
        """Validate segment identifier and optional human-readable name."""
        if not isinstance(self.segment_id, str) or not self.segment_id.strip():
            raise ValueError("segment_id must be a non-empty string")
        if not isinstance(self.segment_name, str):
            raise TypeError("segment_name must be a string")


@dataclass(frozen=True)
class SegmentBalanceEvaluation:
    """Conservation evaluation result for a single segment."""

    segment_id: str
    status: str
    passed: bool
    audit: ConservationAuditResult
    max_abs_energy_residual_j: LedgerScalar
    max_abs_linear_momentum_residual_ns: LedgerScalar
    max_abs_angular_momentum_residual_nms: LedgerScalar
    unavailable_components: tuple[str, ...] = ()
    violations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate segment status and passed invariant."""
        if self.status not in VALID_STATUSES:
            raise ValueError(f"invalid segment status: {self.status!r}")
        if self.passed != (self.status == "pass"):
            raise ValueError("passed must match status == 'pass'")


@dataclass(frozen=True)
class SystemBalanceEvaluation:
    """Aggregated per-segment and system-level conservation evaluation."""

    passed: bool
    status: str
    policy: TolerancePolicy
    policy_sha256: str
    system_audit: ConservationAuditResult
    segment_evaluations: tuple[SegmentBalanceEvaluation, ...]
    segment_sum_residual_j: LedgerScalar
    segment_sum_violation: str | None = None
    failing_segment_ids: tuple[str, ...] = ()
    unavailable_segment_ids: tuple[str, ...] = ()
    violations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate system status, policy hash consistency, and pass invariant."""
        if self.status not in VALID_STATUSES:
            raise ValueError(f"invalid system status: {self.status!r}")
        if self.policy_sha256 != self.policy.policy_sha256:
            raise ValueError("policy_sha256 must match policy.policy_sha256")
        if self.passed != (self.status == "pass" and not self.violations):
            raise ValueError("passed must match status == 'pass' with no violations")


def _run_audit(
    series: EnergyLedgerSeries, p: TolerancePolicy
) -> ConservationAuditResult:
    """Audit a single EnergyLedgerSeries using thresholds from a TolerancePolicy."""
    return audit_energy_ledger(
        series,
        energy_tolerance_j=p.energy_tolerance_j,
        momentum_tolerance_ns=p.momentum_tolerance_ns,
        angular_momentum_tolerance_nms=p.angular_momentum_tolerance_nms,
        require_momentum=p.require_momentum,
    )


def _resolve_status(num_fail: bool, has_unavail: bool, passed: bool) -> str:
    """Resolve canonical status ('pass' | 'fail' | 'unavailable')."""
    if num_fail:
        return "fail"
    return "unavailable" if has_unavail else ("pass" if passed else "fail")


def _evaluate_segment(
    rec: SegmentEnergyRecord, policy: TolerancePolicy
) -> SegmentBalanceEvaluation:
    """Evaluate a single segment ledger against the active tolerance policy."""
    av, a = rec.series.availability, _run_audit(rec.series, policy)
    num_fail = any("UNAVAILABLE" not in v for v in a.violations)
    status = _resolve_status(num_fail, _balance_unavail(av, policy), a.passed)
    unavail = tuple(n for n in _ALL_FLAGS if not getattr(av, n))
    res = (
        a.max_abs_energy_residual_j,
        a.max_abs_linear_momentum_residual_ns,
        a.max_abs_angular_momentum_residual_nms,
    )
    seg_cls = SegmentBalanceEvaluation
    return seg_cls(
        rec.segment_id, status, status == "pass", a, *res, unavail, a.violations
    )


def _step_mech_sum(
    seg_lists: Sequence[tuple[EnergyLedgerStep, ...]], idx: int
) -> float | None:
    """Return summed segment mechanical energy at step idx, or None if unavailable."""
    total = 0.0
    for st_list in seg_lists:
        st = st_list[idx]
        if isinstance(st.kinetic_j, UnavailableType) or isinstance(
            st.potential_j, UnavailableType
        ):
            return None
        total += float(st.kinetic_j) + float(st.potential_j)
    return total


def _compute_segment_sum_residual(
    system: EnergyLedgerSeries,
    segments: Sequence[SegmentEnergyRecord],
    policy: TolerancePolicy,
) -> tuple[LedgerScalar, str | None]:
    """Compute max |sum(segment T+V) - system (T+V)| without zero-filling UNAVAILABLE."""
    all_ser = (system, *(s.series for s in segments))
    if any(
        not (ser.availability.kinetic and ser.availability.potential) for ser in all_ser
    ):
        return UNAVAILABLE, None
    sys_steps, seg_lists = system.steps, [s.series.steps for s in segments]
    if any(len(st) != len(sys_steps) for st in seg_lists):
        return UNAVAILABLE, "segment_sum_violation: segment step count mismatch"
    max_res = 0.0
    for idx, sys_st in enumerate(sys_steps):
        if any(abs(st[idx].time_s - sys_st.time_s) > _TIME_TOL_S for st in seg_lists):
            return UNAVAILABLE, f"segment_sum_violation: timestamp mismatch at {idx}"
        seg_e = _step_mech_sum(seg_lists, idx)
        if seg_e is None:
            return UNAVAILABLE, None
        sys_e = float(sys_st.kinetic_j) + float(sys_st.potential_j)  # type: ignore[arg-type]
        max_res = max(max_res, abs(seg_e - sys_e))
    if policy.require_segment_sum and max_res > policy.segment_sum_tolerance_j:
        tol = policy.segment_sum_tolerance_j
        return max_res, f"segment_sum_violation: {max_res:.3e} J > {tol:.3e} J"
    return max_res, None


def evaluate_segment_and_system_balance(
    system_series: EnergyLedgerSeries,
    segments: Sequence[SegmentEnergyRecord],
    policy: TolerancePolicy | str = "strict_analytic",
) -> SystemBalanceEvaluation:
    """Evaluate per-segment and system balance under a versioned TolerancePolicy."""
    if not segments:
        raise ValueError("segments must be non-empty")
    seg_ids = [s.segment_id for s in segments]
    if len(set(seg_ids)) != len(seg_ids):
        raise ValueError("duplicate segment_id in segments")
    pol = tolerance_policy_preset(policy) if isinstance(policy, str) else policy
    sys_audit = _run_audit(system_series, pol)
    seg_evals = tuple(_evaluate_segment(seg, pol) for seg in segments)
    sum_res, sum_viol = _compute_segment_sum_residual(system_series, segments, pol)
    fail_ids = tuple(e.segment_id for e in seg_evals if e.status == "fail")
    unavail_ids = tuple(e.segment_id for e in seg_evals if e.status == "unavailable")
    sys_unavail = _balance_unavail(system_series.availability, pol)
    sys_num_fail = any("UNAVAILABLE" not in v for v in sys_audit.violations)
    viols = list(sys_audit.violations) + ([sum_viol] if sum_viol else [])
    for ev in seg_evals:
        if ev.status != "pass":
            msg = "; ".join(ev.violations) or ev.status
            viols.append(f"segment {ev.segment_id} ({ev.status}): {msg}")
    status = _resolve_status(
        sys_num_fail or bool(fail_ids) or sum_viol is not None,
        sys_unavail or bool(unavail_ids),
        sys_audit.passed and not viols,
    )
    ok, sha, cls = (
        status == "pass" and not viols,
        pol.policy_sha256,
        SystemBalanceEvaluation,
    )
    tail = (sum_res, sum_viol, fail_ids, unavail_ids, tuple(viols))
    return cls(ok, status, pol, sha, sys_audit, seg_evals, *tail)
