"""Validation: the ASME V&V 20 metric (issue #8616).

The third and last of the three V&V steps, and the only one that touches
experimental data.

The equations
-------------

::

    E     = S - D                                   simulation minus experiment
    u_num = u_h + u_it + u_ro                       SIMPLE ADDITION, not RMS
    u_val = sqrt(u_num^2 + u_input^2 + u_exp^2)     quadrature
    U     = k * u_val,  k = 2 for about 95%

The asymmetry between the two combinations is the substance, not a
typo.  ``u_h``, ``u_it`` and ``u_ro`` are **epistemic and correlated**:
they are three faces of one discretised solve, and a scheme that is
under-resolved in space is usually under-converged in iteration too.
Adding them is the conservative treatment V&V 20 prescribes.  ``u_num``,
``u_input`` and ``u_exp`` come from independent sources and combine in
quadrature.

The sentence most validation plots leave out
--------------------------------------------

**If ``|E| <= u_val`` the comparison is noise-limited and you have
learned nothing about model error.**  The model error is somewhere inside
an interval that the measurement and the numerics already fill; the
agreement you are looking at is the agreement of two error bars.
:attr:`ValidationResult.noise_limited` is that test, and
:meth:`ValidationResult.statement` says it in words rather than leaving a
reader to eyeball two overlapping bars.

When ``|E| > u_val`` the model-form error is bounded by
``E +/- u_val`` -- an interval, never a point estimate.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .exceptions import VandVError
from .reference_data import require_measurable

__all__ = [
    "COVERAGE_FACTOR",
    "NumericalUncertainty",
    "ValidationComparison",
    "ValidationReport",
    "ValidationResult",
    "validate",
    "validation_report",
]

COVERAGE_FACTOR = 2.0
"""``k`` in ``U = k u_val``. Two, for approximately 95% coverage."""


def _require_non_negative(name: str, value: float) -> float:
    """Coerce and check one uncertainty component.

    Raises:
        VandVError: If the value is negative or non-finite.
    """
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise VandVError(
            f"{name} must be a finite non-negative uncertainty, got {value!r}"
        )
    return number


@dataclass(frozen=True, slots=True)
class NumericalUncertainty:
    """``u_num = u_h + u_it + u_ro``, by simple addition.

    Attributes:
        u_h: Discretisation uncertainty, normally the GCI-derived ``u_h``
            from :mod:`bunkershot3d.vandv.gci`.
        u_it: Iterative-convergence uncertainty. Zero for the F0 tier,
            which solves no linear system and iterates nothing.
        u_ro: Round-off uncertainty.
    """

    u_h: float
    u_it: float = 0.0
    u_ro: float = 0.0

    def __post_init__(self) -> None:
        """Validate every component.

        Raises:
            VandVError: If any component is negative or non-finite.
        """
        for name in ("u_h", "u_it", "u_ro"):
            _require_non_negative(name, getattr(self, name))

    @property
    def total(self) -> float:
        """``u_h + u_it + u_ro``.

        Simple addition, deliberately: the three are correlated faces of
        one discrete solve, so V&V 20 treats them as epistemic and does
        not root-sum-square them.
        """
        return self.u_h + self.u_it + self.u_ro

    @property
    def root_sum_square(self) -> float:
        """What ``u_num`` would be if the components were combined in RMS.

        Provided only so a test can pin the difference and a reader can
        see how much the correct treatment costs.  It is never used in
        :func:`validate`.
        """
        return math.sqrt(self.u_h**2 + self.u_it**2 + self.u_ro**2)


@dataclass(frozen=True)
class ValidationComparison:
    """One simulation-versus-experiment comparison, fully specified.

    Attributes:
        quantity: SI-suffixed quantity name. Checked against the register
            of quantities nobody has measured.
        unit: Unit both values are in.
        simulation_value: ``S``.
        experiment_value: ``D``.
        numerical: ``u_num`` and its three components.
        u_input: Standard uncertainty from the simulation inputs.
        u_exp: Standard uncertainty of the experimental value, or
            ``None`` when the source does not report one.
        reference: Citation for ``D``.
        coverage_factor: ``k``.
        notes: Anything a reader of the verdict needs.
    """

    quantity: str
    unit: str
    simulation_value: float
    experiment_value: float
    numerical: NumericalUncertainty
    u_input: float
    u_exp: float | None
    reference: str
    coverage_factor: float = COVERAGE_FACTOR
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the comparison and refuse unmeasured quantities.

        Raises:
            NoReferenceDataError: If the quantity has no published
                measurement at all.
            VandVError: If a value or uncertainty is unusable.
        """
        require_measurable(self.quantity)
        for name in ("simulation_value", "experiment_value"):
            if not math.isfinite(float(getattr(self, name))):
                raise VandVError(f"{name} must be finite, got {getattr(self, name)!r}")
        _require_non_negative("u_input", self.u_input)
        if self.u_exp is not None:
            _require_non_negative("u_exp", self.u_exp)
        if not math.isfinite(self.coverage_factor) or self.coverage_factor <= 0.0:
            raise VandVError(
                f"coverage factor must be positive, got {self.coverage_factor!r}"
            )
        if not self.reference.strip():
            raise VandVError(
                f"comparison of {self.quantity!r} carries no reference for the "
                "experimental value; an unsourced D is not a measurement"
            )


@dataclass(frozen=True)
class ValidationResult:
    """The V&V 20 verdict for one comparison.

    Attributes:
        comparison: What was compared.
        comparison_error: ``E = S - D``.
        u_num: ``u_h + u_it + u_ro``.
        u_input: As supplied.
        u_exp: As supplied, or ``None``.
        u_val: ``sqrt(u_num^2 + u_input^2 + u_exp^2)``, or ``None`` when
            ``u_exp`` is unknown and the interval cannot be closed.
        expanded_uncertainty: ``U = k u_val``, or ``None``.
    """

    comparison: ValidationComparison
    comparison_error: float
    u_num: float
    u_input: float
    u_exp: float | None
    u_val: float | None
    expanded_uncertainty: float | None

    @property
    def is_indeterminate(self) -> bool:
        """True when ``u_exp`` is unknown, so no verdict is possible.

        A missing experimental uncertainty does not make a comparison
        favourable; it makes it unassessable.  Treating an unreported
        ``u_exp`` as zero would silently claim the measurement was exact.
        """
        return self.u_val is None

    @property
    def noise_limited(self) -> bool:
        """``|E| <= u_val``: the comparison carries no information.

        The single most useful line in V&V 20 and the one most validation
        plots omit.  ``False`` when the verdict is indeterminate, because
        nothing at all is known in that case.
        """
        if self.u_val is None:
            return False
        return abs(self.comparison_error) <= self.u_val

    @property
    def model_error_interval(self) -> tuple[float, float] | None:
        """``E +/- u_val``, the bound on the model-form error.

        ``None`` when the verdict is indeterminate.  The interval is
        returned even when the comparison is noise-limited, because it
        straddles zero there and that is exactly the finding.
        """
        if self.u_val is None:
            return None
        return (self.comparison_error - self.u_val, self.comparison_error + self.u_val)

    @property
    def relative_error(self) -> float | None:
        """``E / D``, or ``None`` when ``D`` is zero."""
        if self.comparison.experiment_value == 0.0:
            return None
        return self.comparison_error / self.comparison.experiment_value

    def statement(self) -> str:
        """The verdict in words, including the noise-limited sentence."""
        unit = self.comparison.unit
        head = (
            f"{self.comparison.quantity}: S = "
            f"{self.comparison.simulation_value:.6g} {unit}, D = "
            f"{self.comparison.experiment_value:.6g} {unit}, "
            f"E = {self.comparison_error:+.4g} {unit}"
        )
        if self.u_val is None:
            return (
                f"{head}\n  INDETERMINATE: the source reports no uncertainty on "
                "D, so u_val cannot be formed. No statement about model error "
                "is possible, and treating u_exp as zero would claim the "
                f"measurement was exact.\n  reference: {self.comparison.reference}"
            )
        lines = [
            head,
            f"  u_num = {self.u_num:.4g} (u_h + u_it + u_ro, simple addition), "
            f"u_input = {self.u_input:.4g}, u_exp = {self.u_exp:.4g}",
            f"  u_val = {self.u_val:.4g} {unit}, "
            f"U = {self.comparison.coverage_factor:g} u_val = "
            f"{self.expanded_uncertainty:.4g} {unit}",
        ]
        if self.noise_limited:
            lines.append(
                f"  NOISE-LIMITED: |E| = {abs(self.comparison_error):.4g} <= "
                f"u_val = {self.u_val:.4g}. Nothing has been learned about "
                "model error. This is agreement between two error bars, not "
                "evidence that the model is right."
            )
        else:
            low, high = self.model_error_interval or (0.0, 0.0)
            lines.append(
                f"  model-form error is bounded by [{low:+.4g}, {high:+.4g}] "
                f"{unit}; the interval excludes zero, so a real model error "
                "has been detected."
            )
        lines.append(f"  reference: {self.comparison.reference}")
        lines.extend(f"  note: {note}" for note in self.comparison.notes)
        return "\n".join(lines)


def validate(comparison: ValidationComparison) -> ValidationResult:
    """Apply the V&V 20 metric to one comparison.

    Args:
        comparison: The fully specified comparison.

    Returns:
        The verdict, including whether it is noise-limited.

    Raises:
        VandVError: If ``comparison`` is not a
            :class:`ValidationComparison`.
    """
    if not isinstance(comparison, ValidationComparison):
        raise VandVError(
            f"expected a ValidationComparison, got {type(comparison).__name__}"
        )
    error = comparison.simulation_value - comparison.experiment_value
    u_num = comparison.numerical.total
    u_input = float(comparison.u_input)
    if comparison.u_exp is None:
        return ValidationResult(
            comparison=comparison,
            comparison_error=error,
            u_num=u_num,
            u_input=u_input,
            u_exp=None,
            u_val=None,
            expanded_uncertainty=None,
        )
    u_exp = float(comparison.u_exp)
    u_val = math.sqrt(u_num**2 + u_input**2 + u_exp**2)
    return ValidationResult(
        comparison=comparison,
        comparison_error=error,
        u_num=u_num,
        u_input=u_input,
        u_exp=u_exp,
        u_val=u_val,
        expanded_uncertainty=comparison.coverage_factor * u_val,
    )


@dataclass(frozen=True)
class ValidationReport:
    """Several validation results, with the noise-limited share reported.

    Attributes:
        results: The verdicts, in the order they were formed.
    """

    results: tuple[ValidationResult, ...]

    def __post_init__(self) -> None:
        """Validate.

        Raises:
            VandVError: If the report is empty.
        """
        if not self.results:
            raise VandVError(
                "a validation report needs at least one comparison; an empty "
                "report is not the same as a passing one"
            )

    @property
    def noise_limited_fraction(self) -> float:
        """Share of comparisons that carry no information about model error."""
        limited = sum(1 for result in self.results if result.noise_limited)
        return limited / len(self.results)

    @property
    def indeterminate_fraction(self) -> float:
        """Share of comparisons whose ``u_exp`` the source never reported."""
        unknown = sum(1 for result in self.results if result.is_indeterminate)
        return unknown / len(self.results)

    @property
    def informative_results(self) -> tuple[ValidationResult, ...]:
        """Comparisons that actually bound a model error away from zero."""
        return tuple(
            result
            for result in self.results
            if not result.is_indeterminate and not result.noise_limited
        )

    def summary(self) -> str:
        """A multi-line statement fit for the credibility statement."""
        lines = [
            f"validation over {len(self.results)} comparison(s): "
            f"{self.noise_limited_fraction:.0%} noise-limited, "
            f"{self.indeterminate_fraction:.0%} indeterminate, "
            f"{len(self.informative_results)} informative"
        ]
        lines.extend(result.statement() for result in self.results)
        return "\n".join(lines)


def validation_report(
    comparisons: Sequence[ValidationComparison],
) -> ValidationReport:
    """Validate a sequence of comparisons and collect the verdicts.

    Args:
        comparisons: The comparisons to run.

    Returns:
        The report.

    Raises:
        VandVError: If no comparisons are supplied.
    """
    return ValidationReport(tuple(validate(item) for item in comparisons))
