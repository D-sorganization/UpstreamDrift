"""Variance attribution of putting outcomes to error sources (#14175).

Two complementary answers to "where do my putts go wrong?":

* Sobol first-order (``s1``) and total (``st``) indices of strokes, make
  probability and leave distance, via the Tools Saltelli design and estimator.
* Factor elimination: expected strokes with one error source's spread set to
  zero (and cut by ``reduction``), using common random numbers so the
  difference is the source's effect and not Monte Carlo noise.

Read the indices with care. A holed/missed outcome is binary and pace x line
interactions are strong (a putt drops only when speed *and* line are both
inside the cup window), so ``st`` can far exceed ``s1`` and the first-order
indices need not sum to one. Eliminating a source is the interventional
number to quote for coaching; the indices say how the variance is shared.

Factors are plain names from :data:`.skill.CHANNELS`. The per-green surface and
per-putt roughness factors come from the green sampler (a separate PR): add
their names and deviate mappings next to ``execution_from_deviates`` and the
evaluators here, which only ever see names, need no change.

The physics is injected as a :data:`PuttModel`; this module imports none.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri
from shared.python.swing_sim.variation.sobol_sensitivity import (
    SaltelliDesign,
    SobolFactor,
    SobolObservations,
    SobolOutput,
    SobolReport,
    analyze_sobol,
    generate_saltelli_design,
)

from src.shared.python.contracts import require
from src.shared.python.putting_analytics.sampling import (
    ExecutionSample,
    ReadSample,
    execution_from_deviates,
    read_from_deviates,
    sample_execution,
    sample_read,
)
from src.shared.python.putting_analytics.skill import CHANNELS, PuttingSkillProfile

OUTPUT_NAMES: tuple[str, ...] = ("strokes", "holed", "leave_m")
_UNIT_EPS = 1e-12  # keeps the inverse normal CDF finite at the cube edges


@dataclass(frozen=True)
class PuttOutcomes:
    """Per-putt strokes taken, holed flag and leave distance (m)."""

    strokes: np.ndarray
    holed: np.ndarray
    leave_m: np.ndarray

    def __post_init__(self) -> None:
        shapes = {self.strokes.shape, self.holed.shape, self.leave_m.shape}
        require(len(shapes) == 1, "outcome arrays must share one shape", shapes)


PuttModel = Callable[[ExecutionSample, ReadSample], PuttOutcomes]


@dataclass(frozen=True)
class AttributionConfig:
    """Study settings. ``base_samples`` should be a power of two (Sobol design)."""

    factors: tuple[str, ...] = CHANNELS
    base_samples: int = 1024
    n_samples: int = 20_000
    seed: int = 0
    n_bootstrap: int = 200
    reduction: float = 0.1

    def __post_init__(self) -> None:
        require(len(self.factors) >= 1, "at least one factor is required")
        for name in self.factors:
            require(name in CHANNELS, f"unknown factor; expected one of {CHANNELS}")
        require(len(set(self.factors)) == len(self.factors), "factors must be unique")
        require(self.base_samples >= 2, "base_samples must be at least 2")
        require(self.n_samples >= 2, "n_samples must be at least 2")
        require(self.seed >= 0, "seed must be non-negative")
        require(0.0 < self.reduction <= 1.0, "reduction must be in (0, 1]")


@dataclass(frozen=True)
class FactorElimination:
    """Strokes gained per putt if ``factor`` were removed or cut.

    ``gain`` is for removing the source entirely, ``gain_se`` its paired Monte
    Carlo standard error, ``gain_reduced`` for cutting its spread by the
    config's ``reduction`` (10 % by default).
    """

    factor: str
    gain: float
    gain_se: float
    gain_reduced: float


@dataclass(frozen=True)
class ClassAttribution:
    """Sobol report and elimination table for one putt class."""

    baseline_strokes: float
    sobol: SobolReport
    eliminations: tuple[FactorElimination, ...]

    def elimination(self, factor: str) -> FactorElimination:
        """Return the elimination row for ``factor``."""
        for row in self.eliminations:
            if row.factor == factor:
                return row
        raise ValueError(f"no elimination row for factor {factor!r}")


def _simulate(
    model: PuttModel, profile: PuttingSkillProfile, z: Mapping[str, np.ndarray]
) -> PuttOutcomes:
    return model(
        execution_from_deviates(profile.stroke, z),
        read_from_deviates(profile.read, z),
    )


def _design_deviates(
    design: SaltelliDesign, factors: Sequence[str]
) -> dict[str, np.ndarray]:
    """Map the unit-cube design to standard normals; unvaried channels get zero."""
    unit = np.clip(design.physical_points, _UNIT_EPS, 1.0 - _UNIT_EPS)
    z = {name: np.zeros(design.total_samples) for name in CHANNELS}
    for col, name in enumerate(factors):
        z[name] = ndtri(unit[:, col])
    return z


def sobol_attribution(
    model: PuttModel, profile: PuttingSkillProfile, config: AttributionConfig
) -> SobolReport:
    """Sobol indices of strokes, holed and leave over the configured factors.

    Factors outside ``config.factors`` are frozen (spread zero). The Tools
    estimator takes externally computed observations, so the model is run once
    over the whole Saltelli design, vectorised.
    """
    held = profile.without(*(c for c in CHANNELS if c not in config.factors))
    unit_cube = tuple(SobolFactor(name, name, 0.0, 1.0, "") for name in config.factors)
    design = generate_saltelli_design(unit_cube, config.base_samples, config.seed)
    out = _simulate(model, held, _design_deviates(design, config.factors))
    values = np.column_stack(
        [out.strokes, out.holed.astype(float), out.leave_m]
    ).astype(float)
    observations = SobolObservations(
        design=design,
        outputs=tuple(SobolOutput(name) for name in OUTPUT_NAMES),
        values=values,
        success=np.ones(design.total_samples, dtype=bool),
    )
    return analyze_sobol(observations, n_bootstrap=config.n_bootstrap, seed=config.seed)


def _strokes(
    model: PuttModel, profile: PuttingSkillProfile, config: AttributionConfig
) -> np.ndarray:
    """Strokes for ``config.n_samples`` putts; the seed fixes every channel's draws."""
    n, seed = config.n_samples, config.seed
    execution = sample_execution(profile.stroke, n, seed)
    return model(execution, sample_read(profile.read, n, seed)).strokes


def _mean_and_se(paired_difference: np.ndarray) -> tuple[float, float]:
    n = paired_difference.size
    return float(paired_difference.mean()), float(
        paired_difference.std(ddof=1) / np.sqrt(n)
    )


def elimination_gains(
    model: PuttModel, profile: PuttingSkillProfile, config: AttributionConfig
) -> tuple[float, tuple[FactorElimination, ...]]:
    """Baseline expected strokes and the gain from removing/cutting each source.

    Every variant reuses the baseline's draws (common random numbers), so a
    source the model ignores gains exactly zero and the standard errors are
    those of the paired difference.
    """
    base = _strokes(model, profile, config)
    keep = 1.0 - config.reduction
    rows = []
    for name in config.factors:
        removed = _strokes(model, profile.without(name), config)
        cut = _strokes(model, profile.scaled(keep, [name]), config)
        gain, gain_se = _mean_and_se(base - removed)
        rows.append(FactorElimination(name, gain, gain_se, float((base - cut).mean())))
    return float(base.mean()), tuple(rows)


def attribute_variance(
    models: Mapping[str, PuttModel],
    profile: PuttingSkillProfile,
    config: AttributionConfig | None = None,
) -> dict[str, ClassAttribution]:
    """Run the Sobol study and the elimination study for every putt class.

    ``models`` maps a putt-class label (e.g. ``"3-6 ft"``) to a model bound to
    that class's distance and green.
    """
    require(len(models) >= 1, "at least one putt class is required")
    config = config or AttributionConfig()
    result = {}
    for label, model in models.items():
        baseline, rows = elimination_gains(model, profile, config)
        result[label] = ClassAttribution(
            baseline, sobol_attribution(model, profile, config), rows
        )
    return result


def strokes_gained_per_round(
    attribution: Mapping[str, ClassAttribution],
    putts_per_round: Mapping[str, float],
) -> dict[str, tuple[float, float]]:
    """Per-round strokes gained by removing, and by cutting, each error source.

    ``putts_per_round`` weights each putt class by how often it occurs in a
    round. Returns ``{factor: (gain_if_removed, gain_if_cut)}``.
    """
    require(
        set(putts_per_round) == set(attribution),
        "putts_per_round must name exactly the attributed putt classes",
    )
    require(all(w >= 0.0 for w in putts_per_round.values()), "weights must be >= 0")
    totals: dict[str, list[float]] = {}
    for label, weight in putts_per_round.items():
        for row in attribution[label].eliminations:
            acc = totals.setdefault(row.factor, [0.0, 0.0])
            acc[0] += weight * row.gain
            acc[1] += weight * row.gain_reduced
    return {factor: (acc[0], acc[1]) for factor, acc in totals.items()}
