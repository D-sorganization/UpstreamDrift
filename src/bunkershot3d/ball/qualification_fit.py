"""Fit, held-out validation and report for the transfer qualification (#9543).

The three steps, in the order they must run
-------------------------------------------

1. :func:`fit_transfer` fits the declared
   :class:`~bunkershot3d.ball.splash.MomentumTransfer` parameters on the
   designated calibration strokes, bounded to ``[0, 1]`` so the momentum
   budget of issue #8657 survives the fit, and refuses to report a value for
   a parameter the data cannot identify. A fit that fails is returned as a
   :class:`~.qualification.FitOutcome` with its reasons, never discarded.
2. :func:`validate_holdout` freezes those parameters and compares the
   prediction for every held-out stroke against its measured launch under
   ASME V&V 20 (:func:`bunkershot3d.vandv.validation.validate`), so each
   stroke says whether its discrepancy is inside the numerical, input and
   measurement uncertainty or is a detected model-form error. Per regime,
   bias, RMS error and interval coverage are then judged against the
   predeclared :data:`~.qualification.PRACTICAL_TOLERANCES`.
3. :func:`qualify` runs both on a
   :class:`~.qualification.QualificationDataset` and returns the versioned
   :class:`~.qualification.TransferQualification`. It is a pure function of
   the dataset and the tolerances: re-running it on the same evidence yields
   the same version.

What counts as identification
-----------------------------

The ball-speed model is ``v = eta * (1 - s (1 - D_r)) * K`` with ``K`` fixed
per stroke by the solver and the lie, so ``s`` is separable from ``eta``
only through variation in ``D_r`` across the calibration strokes. The fit
therefore refuses ``packing_sensitivity`` when the calibration beds span
less than :data:`MIN_RELATIVE_DENSITY_SPREAD`, and refuses any parameter
set whose scaled Jacobian is numerically rank-deficient. Spin identifies
only the product of ``sand_ball_friction`` and the lever-arm convention, so
the lever arm is never fitted and ``sand_ball_friction`` is fitted only
when every calibration stroke carries a spin record.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from ..vandv.validation import ValidationComparison, ValidationResult, validate
from .qualification import (
    PRACTICAL_TOLERANCES,
    QUALIFICATION_EVIDENCE_SCHEMA,
    FitOutcome,
    FitStatus,
    MeasuredStroke,
    PracticalTolerances,
    QualificationDataset,
    RegimeVerdict,
    TransferQualification,
    TransferQualificationError,
    UseRegime,
    rig_capability_markdown,
)
from .splash import (
    DEFAULT_MOMENTUM_TRANSFER,
    BallLaunchResult,
    MomentumTransfer,
    compute_ball_launch_from_splash,
)

__all__ = [
    "DEFAULT_FIT_PARAMETERS",
    "FITTABLE_PARAMETERS",
    "IDENTIFIABILITY_CONDITION_LIMIT",
    "MIN_CALIBRATION_STROKES",
    "MIN_RELATIVE_DENSITY_SPREAD",
    "fit_transfer",
    "predict_launch",
    "qualification_report_markdown",
    "qualify",
    "validate_holdout",
]

FITTABLE_PARAMETERS: tuple[str, ...] = (
    "efficiency",
    "packing_sensitivity",
    "sand_ball_friction",
)
"""Parameters a fit may declare. ``spin_lever_arm_fraction`` is a convention
degenerate with ``sand_ball_friction`` and is never fitted."""

DEFAULT_FIT_PARAMETERS: tuple[str, ...] = ("efficiency", "packing_sensitivity")
"""The two parameters ball speed alone can identify."""

MIN_CALIBRATION_STROKES = 10
"""Fewest calibration strokes a fit accepts: the ledger's video spec sample."""

MIN_RELATIVE_DENSITY_SPREAD = 0.15
"""Smallest ``max(D_r) - min(D_r)`` across calibration strokes that lets
``packing_sensitivity`` be separated from ``efficiency``."""

IDENTIFIABILITY_CONDITION_LIMIT = 1e6
"""Largest condition number of the column-scaled Jacobian accepted as
identifiable. Beyond it the parameters trade off along a line the data does
not constrain."""

_FINITE_DIFFERENCE_STEP = 1e-6
"""Relative step for the parameter gradients of the prediction."""

_RPM_TO_RAD_S = 2.0 * math.pi / 60.0


def predict_launch(
    stroke: MeasuredStroke, transfer: MomentumTransfer
) -> BallLaunchResult:
    """The shipped pipeline's launch for one measured stroke.

    Args:
        stroke: The stroke.
        transfer: The partition parameters to predict with.

    Returns:
        The launch, exactly as :func:`bunkershot3d.ball.compute_ball_launch_from_splash`
        would report it.
    """
    return compute_ball_launch_from_splash(
        lie=stroke.lie,
        ball=stroke.ball,
        delivery=stroke.delivery,
        club_loft_rad=stroke.club_loft_rad,
        transfer=transfer,
    )


def _transfer_with(
    base: MomentumTransfer, names: Sequence[str], values: np.ndarray
) -> MomentumTransfer:
    """Return ``base`` with the named parameters replaced by ``values``."""
    return replace(
        base, **{name: float(v) for name, v in zip(names, values, strict=True)}
    )


def _weighted_residuals(
    strokes: Sequence[MeasuredStroke],
    base: MomentumTransfer,
    names: Sequence[str],
    values: np.ndarray,
    *,
    with_spin: bool,
) -> np.ndarray:
    """``(S - D) / u_exp`` for ball speed and, when fitted, spin."""
    transfer = _transfer_with(base, names, values)
    residuals: list[float] = []
    for stroke in strokes:
        launch = predict_launch(stroke, transfer)
        residuals.append(
            (launch.ball_speed_m_s - stroke.measured_ball_speed_m_s)
            / stroke.ball_speed_standard_uncertainty_m_s
        )
        if with_spin and stroke.spin_rate is not None:
            measured = float(stroke.spin_rate.value)  # type: ignore[arg-type]
            u_spin = (
                0.5 * stroke.spin_rate.relative_expanded_uncertainty * abs(measured)
            )
            residuals.append(
                (launch.spin_rate_rpm * _RPM_TO_RAD_S - measured) / max(u_spin, 1e-12)
            )
    return np.asarray(residuals, dtype=float)


def _speed_gradient(
    stroke: MeasuredStroke, transfer: MomentumTransfer, names: Sequence[str]
) -> np.ndarray:
    """Central-difference gradient of predicted ball speed in the parameters."""
    gradient = np.zeros(len(names))
    for index, name in enumerate(names):
        centre = float(getattr(transfer, name))
        step = max(_FINITE_DIFFERENCE_STEP, _FINITE_DIFFERENCE_STEP * abs(centre))
        upper = min(1.0, centre + step)
        lower = max(0.0, centre - step)
        v_upper = predict_launch(
            stroke, replace(transfer, **{name: upper})
        ).ball_speed_m_s
        v_lower = predict_launch(
            stroke, replace(transfer, **{name: lower})
        ).ball_speed_m_s
        gradient[index] = (v_upper - v_lower) / (upper - lower)
    return gradient


def _failed(
    status: FitStatus,
    names: tuple[str, ...],
    strokes: Sequence[MeasuredStroke],
    reasons: tuple[str, ...],
) -> FitOutcome:
    """A preserved failed fit."""
    return FitOutcome(
        status=status,
        parameters=names,
        transfer=None,
        standard_errors={},
        sensitivities={},
        covariance=(),
        residual_rms=None,
        calibration_stroke_ids=tuple(s.stroke_id for s in strokes),
        reasons=reasons,
    )


def _precheck(
    strokes: Sequence[MeasuredStroke], names: tuple[str, ...]
) -> FitOutcome | None:
    """The refusals that need no optimiser, or ``None`` to proceed."""
    if len(strokes) < MIN_CALIBRATION_STROKES:
        return _failed(
            FitStatus.INSUFFICIENT_DATA,
            names,
            strokes,
            (
                f"{len(strokes)} calibration stroke(s) against the "
                f"{MIN_CALIBRATION_STROKES} this fit requires",
            ),
        )
    if "sand_ball_friction" in names and any(s.spin_rate is None for s in strokes):
        return _failed(
            FitStatus.INSUFFICIENT_DATA,
            names,
            strokes,
            (
                "sand_ball_friction is declared but not every calibration stroke "
                "carries a spin record; a parameter only spin constrains cannot "
                "be fitted to strokes with no spin",
            ),
        )
    densities = [s.delivery.bed_relative_density for s in strokes]
    spread = max(densities) - min(densities)
    if "packing_sensitivity" in names and spread < MIN_RELATIVE_DENSITY_SPREAD:
        return _failed(
            FitStatus.UNIDENTIFIABLE,
            names,
            strokes,
            (
                f"calibration beds span a relative density range of {spread:.3g}, "
                f"below the {MIN_RELATIVE_DENSITY_SPREAD:g} needed to separate "
                "packing_sensitivity from efficiency; at one packing the two are "
                "one number",
            ),
        )
    return None


def _check_identifiability(
    jacobian: np.ndarray,
    names: tuple[str, ...],
    strokes: tuple[MeasuredStroke, ...],
) -> tuple[FitOutcome | None, float]:
    scale = np.linalg.norm(jacobian, axis=0)
    if np.any(scale == 0.0):
        inert = [n for n, s in zip(names, scale, strict=True) if s == 0.0]
        return (
            _failed(
                FitStatus.UNIDENTIFIABLE,
                names,
                strokes,
                ("the calibration strokes do not respond to " + ", ".join(inert),),
            ),
            math.nan,
        )
    singular = np.linalg.svd(jacobian / scale, compute_uv=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0.0 else math.inf
    if condition > IDENTIFIABILITY_CONDITION_LIMIT:
        return (
            _failed(
                FitStatus.UNIDENTIFIABLE,
                names,
                strokes,
                (
                    f"scaled Jacobian condition number {condition:.3g} exceeds "
                    f"{IDENTIFIABILITY_CONDITION_LIMIT:g}; the parameters trade off "
                    "along a direction the data does not constrain",
                ),
            ),
            condition,
        )
    return None, condition


def _build_converged_outcome(
    result: Any,
    names: tuple[str, ...],
    strokes: tuple[MeasuredStroke, ...],
    initial: MomentumTransfer,
    jacobian: np.ndarray,
    condition: float,
) -> FitOutcome:
    residuals = np.asarray(result.fun, dtype=float)
    dof = max(1, residuals.size - len(names))
    chi_squared_reduced = float(residuals @ residuals) / dof
    covariance = np.linalg.inv(jacobian.T @ jacobian) * max(1.0, chi_squared_reduced)
    transfer = _transfer_with(initial, names, result.x)
    sensitivities = np.zeros(len(names))
    for stroke in strokes:
        launch = predict_launch(stroke, transfer)
        gradient = _speed_gradient(stroke, transfer, names)
        sensitivities += gradient * result.x / max(launch.ball_speed_m_s, 1e-12)
    sensitivities /= len(strokes)
    at_bound = [n for n, v in zip(names, result.x, strict=True) if v <= 0.0 or v >= 1.0]
    reasons = [
        f"converged on {len(strokes)} stroke(s); reduced chi-squared "
        f"{chi_squared_reduced:.3g}; condition number {condition:.3g}"
    ]
    if at_bound:
        reasons.append(
            "at a physical bound: "
            + ", ".join(at_bound)
            + "; a bound-limited value is a statement about the model, not the sand"
        )
    return FitOutcome(
        status=FitStatus.CONVERGED,
        parameters=names,
        transfer=transfer,
        standard_errors={
            n: float(math.sqrt(covariance[i, i])) for i, n in enumerate(names)
        },
        sensitivities={n: float(sensitivities[i]) for i, n in enumerate(names)},
        covariance=tuple(tuple(float(v) for v in row) for row in covariance),
        residual_rms=float(math.sqrt(residuals @ residuals / residuals.size)),
        calibration_stroke_ids=tuple(s.stroke_id for s in strokes),
        reasons=tuple(reasons),
    )


def fit_transfer(
    strokes: Sequence[MeasuredStroke],
    *,
    parameters: tuple[str, ...] = DEFAULT_FIT_PARAMETERS,
    initial: MomentumTransfer = DEFAULT_MOMENTUM_TRANSFER,
) -> FitOutcome:
    """Fit the declared parameters on the calibration strokes.

    Args:
        strokes: The designated calibration strokes.
        parameters: Names from :data:`FITTABLE_PARAMETERS` to fit; every
            other parameter keeps its value in ``initial``.
        initial: Starting point, and the source of the unfitted values.

    Returns:
        The outcome. A failed fit is returned, not raised, so it is preserved
        in the evidence.

    Raises:
        TransferQualificationError: If a declared parameter is not fittable
            or is repeated.
    """
    names = tuple(parameters)
    unknown = sorted(set(names) - set(FITTABLE_PARAMETERS))
    if unknown or len(names) != len(set(names)) or not names:
        raise TransferQualificationError(
            "fit parameters must be distinct names from "
            + ", ".join(FITTABLE_PARAMETERS)
            + f", got {parameters!r}"
        )
    strokes = tuple(strokes)
    refused = _precheck(strokes, names)
    if refused is not None:
        return refused
    # Imported here, as ``calibration.optimizer`` does: the package import must
    # not require the optimiser extras (tests/bunkershot3d/test_optimizer.py).
    from scipy.optimize import least_squares

    with_spin = "sand_ball_friction" in names
    x0 = np.array([float(getattr(initial, n)) for n in names])

    def _residual_fun(x: np.ndarray) -> np.ndarray:
        return _weighted_residuals(strokes, initial, names, x, with_spin=with_spin)

    result = least_squares(
        _residual_fun,
        x0,
        bounds=(np.zeros(len(names)), np.ones(len(names))),
        method="trf",
    )
    if not result.success:
        return _failed(
            FitStatus.NOT_CONVERGED, names, strokes, (f"optimiser: {result.message}",)
        )
    jacobian = np.asarray(result.jac, dtype=float)
    failed_ident, condition = _check_identifiability(jacobian, names, strokes)
    if failed_ident is not None:
        return failed_ident
    return _build_converged_outcome(
        result, names, strokes, initial, jacobian, condition
    )


def _compare_stroke(stroke: MeasuredStroke, fit: FitOutcome) -> ValidationResult:
    """One held-out stroke under V&V 20, with the fit's parameter uncertainty
    entering ``u_input`` in quadrature with the sand-state input uncertainty."""
    transfer = fit.transfer
    if transfer is None:  # pragma: no cover - guarded by the caller
        raise TransferQualificationError("cannot compare without frozen parameters")
    launch = predict_launch(stroke, transfer)
    u_parameters = 0.0
    if fit.covariance:
        gradient = _speed_gradient(stroke, transfer, fit.parameters)
        cov = np.asarray(fit.covariance, dtype=float)
        u_parameters = float(math.sqrt(max(0.0, gradient @ cov @ gradient)))
    return validate(
        ValidationComparison(
            quantity="ball_speed_m_s",
            unit="m/s",
            simulation_value=launch.ball_speed_m_s,
            experiment_value=stroke.measured_ball_speed_m_s,
            numerical=stroke.numerical,
            u_input=math.hypot(stroke.u_input_m_s, u_parameters),
            u_exp=stroke.ball_speed_standard_uncertainty_m_s,
            reference=stroke.ball_speed.source,
            notes=(f"stroke {stroke.stroke_id}, raw {stroke.raw_data_digest[:12]}",),
            measured_record=stroke.ball_speed,
        )
    )


def _dominant_uncertainty(results: Sequence[ValidationResult]) -> str:
    """Which of the three standard uncertainties is largest on average."""
    means = {
        "numerical": float(np.mean([r.u_num for r in results])),
        "measurement": float(np.mean([r.u_exp or 0.0 for r in results])),
        "input": float(np.mean([r.u_input for r in results])),
    }
    name, value = max(means.items(), key=lambda item: item[1])
    return name if value > 0.0 else "none"


def _angle_bias(
    strokes: Sequence[MeasuredStroke], transfer: MomentumTransfer
) -> float | None:
    """``mean(S - D)`` of launch angle over the strokes that carry one."""
    errors = [
        predict_launch(s, transfer).launch_angle_rad - float(s.launch_angle.value)  # type: ignore[arg-type]
        for s in strokes
        if s.launch_angle is not None
    ]
    return float(np.mean(errors)) if errors else None


def _judge_regime(
    regime: UseRegime,
    strokes: Sequence[MeasuredStroke],
    fit: FitOutcome,
    tolerances: PracticalTolerances,
    calibration_batches: frozenset[str],
) -> RegimeVerdict:
    """The verdict for one regime's held-out strokes."""
    ids = tuple(s.stroke_id for s in strokes)
    sessions = tuple(sorted({s.session_id for s in strokes}))
    independent = bool(strokes) and all(
        s.sand_batch_id not in calibration_batches for s in strokes
    )
    reasons: list[str] = []
    if not fit.succeeded:
        reasons.append(
            f"calibration fit ended {fit.status.value}: " + "; ".join(fit.reasons)
        )
    if not strokes:
        reasons.append("no held-out stroke in this regime")
    elif len(strokes) < tolerances.min_holdout_strokes_per_regime:
        reasons.append(
            f"{len(strokes)} held-out stroke(s) against the "
            f"{tolerances.min_holdout_strokes_per_regime} the tolerances require"
        )
    if not fit.succeeded or not strokes:
        return RegimeVerdict(
            regime=regime,
            qualified=False,
            holdout_stroke_ids=ids,
            holdout_session_ids=sessions,
            independent_sand_batches=independent,
            relative_bias=None,
            relative_rms_error=None,
            interval_coverage=None,
            launch_angle_bias_rad=None,
            noise_limited_count=0,
            model_error_detected_count=0,
            dominant_uncertainty="none",
            reasons=tuple(reasons),
        )
    results = [_compare_stroke(s, fit) for s in strokes]
    relative = np.array([r.relative_error for r in results], dtype=float)
    bias = float(np.mean(relative))
    rms = float(math.sqrt(np.mean(relative**2)))
    covered = [
        abs(r.comparison_error) <= (r.expanded_uncertainty or 0.0) for r in results
    ]
    coverage = float(np.mean(covered))
    angle_bias = _angle_bias(strokes, fit.transfer)  # type: ignore[arg-type]
    if abs(bias) > tolerances.max_abs_relative_bias:
        reasons.append(
            f"relative bias {bias:+.3g} exceeds {tolerances.max_abs_relative_bias:g}"
        )
    if rms > tolerances.max_relative_rms_error:
        reasons.append(
            f"relative RMS error {rms:.3g} exceeds {tolerances.max_relative_rms_error:g}"
        )
    if coverage < tolerances.min_interval_coverage:
        reasons.append(
            f"interval coverage {coverage:.2f} is below "
            f"{tolerances.min_interval_coverage:g}"
        )
    if (
        angle_bias is not None
        and abs(angle_bias) > tolerances.max_launch_angle_bias_rad
    ):
        reasons.append(
            f"launch-angle bias {math.degrees(angle_bias):+.2f} deg exceeds "
            f"{math.degrees(tolerances.max_launch_angle_bias_rad):g} deg"
        )
    noise_limited = sum(1 for r in results if r.noise_limited)
    qualified = not reasons
    if qualified:
        reasons.append(
            f"every predeclared tolerance met; {noise_limited} of {len(results)} "
            "stroke(s) noise-limited under V&V 20"
        )
    return RegimeVerdict(
        regime=regime,
        qualified=qualified,
        holdout_stroke_ids=ids,
        holdout_session_ids=sessions,
        independent_sand_batches=independent,
        relative_bias=bias,
        relative_rms_error=rms,
        interval_coverage=coverage,
        launch_angle_bias_rad=angle_bias,
        noise_limited_count=noise_limited,
        model_error_detected_count=len(results) - noise_limited,
        dominant_uncertainty=_dominant_uncertainty(results),
        reasons=tuple(reasons),
    )


def validate_holdout(
    dataset: QualificationDataset,
    fit: FitOutcome,
    tolerances: PracticalTolerances = PRACTICAL_TOLERANCES,
) -> tuple[RegimeVerdict, ...]:
    """Judge every regime of the matrix on the held-out strokes.

    Args:
        dataset: The dataset; only its held-out strokes are compared.
        fit: The frozen fit. A failed fit rejects every regime.
        tolerances: The predeclared tolerances.

    Returns:
        One verdict per regime, in matrix order.
    """
    calibration_batches = frozenset(
        s.sand_batch_id for s in dataset.calibration_strokes
    )
    by_regime: dict[str, list[MeasuredStroke]] = {
        r.key: [] for r in dataset.intended_use.regimes
    }
    for stroke in dataset.holdout_strokes:
        regime = dataset.intended_use.regime_for(stroke.delivery, stroke.lie)
        if regime is not None:  # guaranteed by the dataset contract
            by_regime[regime.key].append(stroke)
    return tuple(
        _judge_regime(
            regime, by_regime[regime.key], fit, tolerances, calibration_batches
        )
        for regime in dataset.intended_use.regimes
    )


def qualify(
    dataset: QualificationDataset,
    *,
    tolerances: PracticalTolerances = PRACTICAL_TOLERANCES,
    parameters: tuple[str, ...] = DEFAULT_FIT_PARAMETERS,
    initial: MomentumTransfer = DEFAULT_MOMENTUM_TRANSFER,
) -> TransferQualification:
    """Run intake, calibration and held-out validation once, re-runnably.

    Args:
        dataset: Every stroke on file with its designated split.
        tolerances: The predeclared tolerances.
        parameters: Parameters to fit.
        initial: Starting point and source of unfitted values.

    Returns:
        The versioned evidence, including every rejected regime and a failed
        fit if that is what happened.
    """
    fit = fit_transfer(
        dataset.calibration_strokes, parameters=parameters, initial=initial
    )
    verdicts = validate_holdout(dataset, fit, tolerances)
    digest = dataset.evidence_digest()
    return TransferQualification(
        version=f"{QUALIFICATION_EVIDENCE_SCHEMA}+{digest[:12]}",
        evidence_digest=digest,
        protocol=dataset.protocol,
        intended_use=dataset.intended_use,
        tolerances=tolerances,
        fit=fit,
        verdicts=verdicts,
        calibration_stroke_ids=tuple(s.stroke_id for s in dataset.calibration_strokes),
        holdout_stroke_ids=tuple(s.stroke_id for s in dataset.holdout_strokes),
    )


def _cell(value: float | None, fmt: str) -> str:
    """One numeric report cell, or ``n/a`` when the statistic does not exist."""
    return "n/a" if value is None else format(value, fmt)


def qualification_report_markdown(qualification: TransferQualification) -> str:
    """Render the evidence as the report issue #9543 asks for.

    Rejected regimes, the uncertainty attribution and the rig capability
    register are always present; a qualified regime is listed with the
    tolerances it met.
    """
    q = qualification
    fit = q.fit
    lines = [
        f"# Sand-to-Ball Transfer Qualification `{q.version}`",
        "",
        f"Evidence digest: `{q.evidence_digest}`",
        f"Protocol: {q.protocol.name} {q.protocol.version} -- {q.protocol.apparatus}",
        f"Calibration strokes: {len(q.calibration_stroke_ids)}; held-out strokes: "
        f"{len(q.holdout_stroke_ids)}",
        "",
        "## Calibration fit",
        "",
        f"Status: **{fit.status.value}**; parameters: {', '.join(fit.parameters)}",
    ]
    if fit.transfer is not None:
        for name in fit.parameters:
            lines.append(
                f"- `{name}` = {getattr(fit.transfer, name):.4g} "
                f"(standard error {fit.standard_errors[name]:.2g}, mean normalised "
                f"sensitivity {fit.sensitivities[name]:+.3g})"
            )
        lines.append(f"- weighted residual RMS: {fit.residual_rms:.3g}")
    lines.extend(f"- {reason}" for reason in fit.reasons)
    lines.extend(["", "## Held-out verdicts by regime", ""])
    lines.append(
        "| Regime | Verdict | Strokes | Sessions | Unseen batches | Rel. bias | "
        "Rel. RMS | Coverage | Noise-limited | Model error | Dominant u | Reasons |"
    )
    lines.append(
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    for v in q.verdicts:
        lines.append(
            f"| {v.regime.key} | {'qualified' if v.qualified else 'rejected'} | "
            f"{len(v.holdout_stroke_ids)} | {len(v.holdout_session_ids)} | "
            f"{'yes' if v.independent_sand_batches else 'no'} | "
            f"{_cell(v.relative_bias, '+.3g')} | {_cell(v.relative_rms_error, '.3g')} | "
            f"{_cell(v.interval_coverage, '.2f')} | {v.noise_limited_count} | "
            f"{v.model_error_detected_count} | {v.dominant_uncertainty} | "
            + "; ".join(v.reasons)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Predeclared tolerances",
            "",
            f"- |relative bias| <= {q.tolerances.max_abs_relative_bias:g}",
            f"- relative RMS <= {q.tolerances.max_relative_rms_error:g}",
            f"- interval coverage >= {q.tolerances.min_interval_coverage:g}",
            f"- |launch-angle bias| <= "
            f"{math.degrees(q.tolerances.max_launch_angle_bias_rad):g} deg",
            f"- held-out strokes per regime >= "
            f"{q.tolerances.min_holdout_strokes_per_regime}",
            "",
            q.tolerances.rationale,
            "",
            "## What the three-camera rig can and cannot measure",
            "",
            rig_capability_markdown(),
            "",
            "No camera purchase and no experiment is assumed complete by this "
            "report; a quantity the rig cannot measure stays unavailable in the "
            "stroke record.",
        ]
    )
    return "\n".join(lines)
