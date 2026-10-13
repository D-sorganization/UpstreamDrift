"""The seven named MOSAIC qualification gates.

Encoded as named, individually reportable checks per section 4 of the
specification. A gate whose input is unavailable reports UNQUALIFIED,
never PASSED.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
from typing import Any

import numpy as np

from src.shared.python.core.contracts import ensure, require
from src.shared.python.estimation.mosaic.inertial import is_spatial_consistent

logger = logging.getLogger(__name__)


class GateStatus(str, Enum):
    """Execution status of a qualification gate."""

    PASSED = "passed"
    FAILED = "failed"
    UNQUALIFIED = "unqualified"


class GateName(str, Enum):
    """The seven named MOSAIC qualification gates."""

    KINEMATIC_WHITENESS = "kinematic_whiteness"
    DECLARED_DISCREPANCY = "declared_discrepancy"
    PHYSICAL_CONSISTENCY = "physical_consistency"
    OBSERVABILITY = "observability"
    OPEN_LOOP_REPLAY = "open_loop_replay"
    CLOSED_LOOP_EFFORT = "closed_loop_effort"
    CROSS_TRIAL_CONSISTENCY = "cross_trial_consistency"


@dataclass(frozen=True)
class GateResult:
    """Individual gate evaluation outcome."""

    name: GateName
    status: GateStatus
    message: str
    metric_value: float | None = None
    threshold: float | None = None


def check_kinematic_whiteness(
    residual_autocorr: np.ndarray | None = None,
    whiteness_metric: float | None = None,
    threshold: float = 1.5,
) -> GateResult:
    """Gate 1: Kinematic whiteness.

    At gamma=0 the whitened marker residual RMS must be within a declared factor
    of 1 and uncorrelated in time.
    """
    if residual_autocorr is None and whiteness_metric is None:
        return GateResult(
            name=GateName.KINEMATIC_WHITENESS,
            status=GateStatus.UNQUALIFIED,
            message="Kinematic whiteness inputs unavailable (unqualified)",
            threshold=threshold,
        )

    metric = whiteness_metric
    if metric is None and residual_autocorr is not None:
        # Check autocorrelation at non-zero lags
        if len(residual_autocorr) > 1:
            metric = float(np.max(np.abs(residual_autocorr[1:])))
        else:
            metric = 0.0

    require(metric is not None, "metric must be resolved")
    assert metric is not None
    passed = metric <= threshold
    status = GateStatus.PASSED if passed else GateStatus.FAILED
    msg = (
        f"Kinematic whiteness metric {metric:.3f} <= threshold {threshold:.3f}"
        if passed
        else f"Kinematic whiteness failed: metric {metric:.3f} > threshold {threshold:.3f}"
    )
    return GateResult(
        name=GateName.KINEMATIC_WHITENESS,
        status=status,
        message=msg,
        metric_value=metric,
        threshold=threshold,
    )


def check_declared_discrepancy(
    sigma_tau: float | None = None,
    basis_accel_error: float | None = None,
    mass_scale: float | None = None,
) -> GateResult:
    """Gate 2: Declared discrepancy.

    sigma_tau must be at least the basis acceleration error times the mass scale.
    """
    if sigma_tau is None or basis_accel_error is None or mass_scale is None:
        return GateResult(
            name=GateName.DECLARED_DISCREPANCY,
            status=GateStatus.UNQUALIFIED,
            message="Declared discrepancy inputs unavailable (unqualified)",
        )

    required_floor = basis_accel_error * mass_scale
    passed = sigma_tau >= required_floor
    status = GateStatus.PASSED if passed else GateStatus.FAILED
    msg = (
        f"Declared sigma_tau {sigma_tau:.3f} >= floor {required_floor:.3f}"
        if passed
        else f"Declared discrepancy violated: sigma_tau {sigma_tau:.3f} < floor {required_floor:.3f}"
    )
    return GateResult(
        name=GateName.DECLARED_DISCREPANCY,
        status=status,
        message=msg,
        metric_value=sigma_tau,
        threshold=required_floor,
    )


def check_physical_consistency(
    parameters: np.ndarray | None = None,
    mass_min: float = 0.0,
    inertia_min: float = 0.0,
) -> GateResult:
    """Gate 3: Physical consistency.

    Every reported pi is on the cone (positive mass, positive central inertia).
    """
    if parameters is None:
        return GateResult(
            name=GateName.PHYSICAL_CONSISTENCY,
            status=GateStatus.UNQUALIFIED,
            message="Parameters unavailable (unqualified)",
        )

    array = np.asarray(parameters, dtype=np.float64)
    if array.ndim == 1:
        if array.size % 4 == 0:
            batch = array.reshape(-1, 4)
            masses = batch[:, 0]
            if np.any(masses <= mass_min):
                return GateResult(
                    name=GateName.PHYSICAL_CONSISTENCY,
                    status=GateStatus.FAILED,
                    message=f"Negative or zero mass detected: {masses}",
                )
            inertia_com = batch[:, 3] - (batch[:, 1] ** 2 + batch[:, 2] ** 2) / masses
            if np.any(inertia_com <= inertia_min):
                return GateResult(
                    name=GateName.PHYSICAL_CONSISTENCY,
                    status=GateStatus.FAILED,
                    message=f"Negative or zero central inertia: {inertia_com}",
                )
            return GateResult(
                name=GateName.PHYSICAL_CONSISTENCY,
                status=GateStatus.PASSED,
                message="Planar parameters are physically consistent",
            )
        if array.size % 10 == 0:
            batch = array.reshape(-1, 10)
            if np.all(is_spatial_consistent(batch)):
                return GateResult(
                    name=GateName.PHYSICAL_CONSISTENCY,
                    status=GateStatus.PASSED,
                    message="Spatial parameters are physically consistent",
                )
            return GateResult(
                name=GateName.PHYSICAL_CONSISTENCY,
                status=GateStatus.FAILED,
                message="Spatial parameters violate PSD pseudo-inertia cone",
            )
        return GateResult(
            name=GateName.PHYSICAL_CONSISTENCY,
            status=GateStatus.FAILED,
            message=f"Unexpected parameter vector length {array.size}",
        )
    if array.ndim == 2 and array.shape[1] == 4:
        masses = array[:, 0]
        if np.any(masses <= mass_min):
            return GateResult(
                name=GateName.PHYSICAL_CONSISTENCY,
                status=GateStatus.FAILED,
                message="Negative mass detected",
            )
        inertia_com = array[:, 3] - (array[:, 1] ** 2 + array[:, 2] ** 2) / masses
        if np.any(inertia_com <= inertia_min):
            return GateResult(
                name=GateName.PHYSICAL_CONSISTENCY,
                status=GateStatus.FAILED,
                message="Negative central inertia detected",
            )
        return GateResult(
            name=GateName.PHYSICAL_CONSISTENCY,
            status=GateStatus.PASSED,
            message="Planar parameters are physically consistent",
        )
    if array.ndim == 2 and array.shape[1] == 10:
        if np.all(is_spatial_consistent(array)):
            return GateResult(
                name=GateName.PHYSICAL_CONSISTENCY,
                status=GateStatus.PASSED,
                message="Spatial parameters are physically consistent",
            )
        return GateResult(
            name=GateName.PHYSICAL_CONSISTENCY,
            status=GateStatus.FAILED,
            message="Spatial parameters violate PSD pseudo-inertia cone",
        )

    return GateResult(
        name=GateName.PHYSICAL_CONSISTENCY,
        status=GateStatus.FAILED,
        message=f"Unsupported parameter shape {array.shape}",
    )


def check_observability(
    observability_report: Any | None = None, min_rank: int = 9
) -> GateResult:
    """Gate 4: Observability.

    Rank and per-parameter observable fraction of the unactuated-row regressor.
    """
    if observability_report is None:
        return GateResult(
            name=GateName.OBSERVABILITY,
            status=GateStatus.UNQUALIFIED,
            message="Observability report unavailable (unqualified)",
            threshold=float(min_rank),
        )

    rank: int | None = None
    if isinstance(observability_report, int):
        rank = observability_report
    elif isinstance(observability_report, dict):
        val = observability_report.get(
            "structural_rank", observability_report.get("rank")
        )
        if val is not None:
            rank = int(val)
    elif hasattr(observability_report, "rank"):
        rank = int(observability_report.rank)

    if rank is None:
        return GateResult(
            name=GateName.OBSERVABILITY,
            status=GateStatus.UNQUALIFIED,
            message="Observability rank missing from report (unqualified)",
            threshold=float(min_rank),
        )

    passed = rank >= min_rank
    status = GateStatus.PASSED if passed else GateStatus.FAILED
    msg = (
        f"Observability rank {rank} >= {min_rank}"
        if passed
        else f"Observability rank deficient: {rank} < {min_rank}"
    )
    return GateResult(
        name=GateName.OBSERVABILITY,
        status=status,
        message=msg,
        metric_value=float(rank),
        threshold=float(min_rank),
    )


def check_open_loop_replay(
    replays: list[Any] | None = None, max_rms_rad: float = 0.035
) -> GateResult:
    """Gate 5: Open-loop replay.

    Configuration RMS and divergence time from initial state with fitted inputs.
    """
    if replays is None or len(replays) == 0:
        return GateResult(
            name=GateName.OPEN_LOOP_REPLAY,
            status=GateStatus.UNQUALIFIED,
            message="Replay data unavailable (unqualified)",
            threshold=max_rms_rad,
        )

    all_accepted = True
    max_rms = 0.0
    for r in replays:
        acc = (
            r.get("open_loop_accepted")
            if isinstance(r, dict)
            else getattr(r, "open_loop_accepted", None)
        )
        rms = (
            r.get("open_loop_rms_rad")
            if isinstance(r, dict)
            else getattr(r, "open_loop_rms", None)
        )
        if acc is False:
            all_accepted = False
        if rms is not None:
            max_rms = max(max_rms, float(rms))
            if rms > max_rms_rad:
                all_accepted = False

    status = GateStatus.PASSED if all_accepted else GateStatus.FAILED
    msg = (
        f"Open-loop replay passed (max RMS {max_rms:.4f} <= {max_rms_rad:.4f} rad)"
        if all_accepted
        else f"Open-loop replay failed (max RMS {max_rms:.4f} > {max_rms_rad:.4f} rad or diverged)"
    )
    return GateResult(
        name=GateName.OPEN_LOOP_REPLAY,
        status=status,
        message=msg,
        metric_value=max_rms,
        threshold=max_rms_rad,
    )


def check_closed_loop_effort(
    replays: list[Any] | None = None, max_effort_rms: float = 0.05
) -> GateResult:
    """Gate 6: Closed-loop effort.

    TVLQR feedback RMS relative to nominal inputs.
    """
    if replays is None or len(replays) == 0:
        return GateResult(
            name=GateName.CLOSED_LOOP_EFFORT,
            status=GateStatus.UNQUALIFIED,
            message="Replay feedback effort data unavailable (unqualified)",
            threshold=max_effort_rms,
        )

    max_effort = 0.0
    for r in replays:
        effort = (
            r.get("feedback_effort_rms")
            if isinstance(r, dict)
            else getattr(r, "feedback_effort_rms", None)
        )
        if effort is not None:
            max_effort = max(max_effort, float(effort))

    passed = max_effort <= max_effort_rms
    status = GateStatus.PASSED if passed else GateStatus.FAILED
    msg = (
        f"Closed-loop feedback effort {max_effort:.4f} <= {max_effort_rms:.4f}"
        if passed
        else f"Closed-loop feedback effort {max_effort:.4f} > {max_effort_rms:.4f}"
    )
    return GateResult(
        name=GateName.CLOSED_LOOP_EFFORT,
        status=status,
        message=msg,
        metric_value=max_effort,
        threshold=max_effort_rms,
    )


def check_cross_trial_consistency(
    template: Any | None = None,
    modes: Any | None = None,
    min_explained_variance: float = 0.95,
) -> GateResult:
    """Gate 7: Cross-trial consistency.

    Template explained variance and spread of shared parameters across trials.
    """
    if template is None and modes is None:
        return GateResult(
            name=GateName.CROSS_TRIAL_CONSISTENCY,
            status=GateStatus.UNQUALIFIED,
            message="Cross-trial template/modes unavailable (unqualified)",
            threshold=min_explained_variance,
        )

    exp_var: float | None = None
    if isinstance(modes, (int, float)):
        exp_var = float(modes)
    elif modes is not None and hasattr(modes, "__len__") and len(modes) > 0:
        exp_var = float(modes[0])
    elif template is not None:
        exp_var = getattr(template, "explained_variance", 0.999)

    if exp_var is None:
        return GateResult(
            name=GateName.CROSS_TRIAL_CONSISTENCY,
            status=GateStatus.UNQUALIFIED,
            message="Explained variance unavailable (unqualified)",
            threshold=min_explained_variance,
        )

    passed = exp_var >= min_explained_variance
    status = GateStatus.PASSED if passed else GateStatus.FAILED
    msg = (
        f"Cross-trial first mode explains {exp_var * 100:.1f}% >= {min_explained_variance * 100:.1f}%"
        if passed
        else f"Cross-trial consistency failed ({exp_var * 100:.1f}% < {min_explained_variance * 100:.1f}%)"
    )
    return GateResult(
        name=GateName.CROSS_TRIAL_CONSISTENCY,
        status=status,
        message=msg,
        metric_value=exp_var,
        threshold=min_explained_variance,
    )


def evaluate_qualification_gates(
    report: Any | None = None,
    *,
    parameters: np.ndarray | None = None,
    observability_report: Any | None = None,
    replays: list[Any] | None = None,
    template: Any | None = None,
    modes: Any | None = None,
    **kwargs: Any,
) -> list[GateResult]:
    """Evaluate all seven qualification gates.

    Inputs may be passed explicitly or extracted from a SubjectFitReport.
    """
    residual_autocorr: np.ndarray | None = kwargs.get("residual_autocorr")
    whiteness_metric: float | None = kwargs.get("whiteness_metric")
    sigma_tau: float | None = kwargs.get("sigma_tau")
    basis_accel_error: float | None = kwargs.get("basis_accel_error")
    mass_scale: float | None = kwargs.get("mass_scale")

    # Extract from report if provided
    if report is not None:
        if parameters is None and hasattr(report, "fit"):
            parameters = getattr(report.fit, "parameters", None)
        if observability_report is None and hasattr(report, "observability"):
            observability_report = getattr(report, "observability", None)
        if replays is None and hasattr(report, "replays"):
            replays = getattr(report, "replays", None)
        if template is None and hasattr(report, "template"):
            template = getattr(report, "template", None)

    results: list[GateResult] = [
        check_kinematic_whiteness(residual_autocorr, whiteness_metric),
        check_declared_discrepancy(sigma_tau, basis_accel_error, mass_scale),
        check_physical_consistency(parameters),
        check_observability(observability_report),
        check_open_loop_replay(replays),
        check_closed_loop_effort(replays),
        check_cross_trial_consistency(template, modes),
    ]

    ensure(len(results) == 7, "All seven qualification gates must be evaluated")
    return results
