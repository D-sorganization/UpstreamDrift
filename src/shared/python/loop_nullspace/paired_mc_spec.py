"""Declared inputs of a paired nonlinear Monte Carlo campaign (NSX-91, #12720).

A campaign compares strategies on the *same* random draws (common random
numbers, CRN) and then confirms the comparison on independent hold-out seeds.
This module holds the value types and the honesty gates (rule R13):

* :class:`TrialDraw` derives every random input from ``(seed set, trial index,
  disturbance)`` alone, through :class:`numpy.random.SeedSequence`.  A draw
  therefore never depends on the strategy, on the order trials run in, or on how
  they are batched, which is what makes CRN, serial/batched equality and
  hold-out independence structural rather than conventional.
* :func:`require_independent_integrator` refuses a replay that shares the
  generator's integrator or step (an inverse crime).
* :func:`compare_noise` refuses a synthetic generator whose noise is *identical*
  to the assumed noise (circular) and reports how a different one differs.

Pure numpy; the disturbance model is :mod:`perturbation.disturbance` (V6).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.loop_nullspace.evidence import Array
from src.shared.python.perturbation.disturbance import DisturbanceSpec

__all__ = [
    "ANGLE_UNITS",
    "SEED_SETS",
    "CampaignSpec",
    "GeneratorSpec",
    "IntegratorSignature",
    "NoiseComparison",
    "NoiseRelation",
    "OutcomeChannel",
    "OutcomeKind",
    "TrialDraw",
    "compare_noise",
    "require_independent_integrator",
]

ANGLE_UNITS = ("rad", "deg")
SEED_SETS = ("paired", "holdout")
_STEP_RTOL = 1e-9


class OutcomeKind(str, Enum):
    """Statistics an outcome gets: wrapped (angle) or plain (scalar)."""

    ANGLE = "angle"
    SCALAR = "scalar"


@dataclass(frozen=True)
class OutcomeChannel:
    """One reported outcome (face angle first, then CHS, angle of attack, launch).

    An ``ANGLE`` outcome needs ``unit`` in ``rad``/``deg`` and a positive
    ``tolerance`` (the exceedance half-width of
    :func:`outcome_dispersion.compute_outcome_dispersion`); ``target`` is the
    angle the wrapped error is measured from.  ``miss_half_width`` is optional.
    """

    name: str
    unit: str
    kind: OutcomeKind
    target: float = 0.0
    tolerance: float | None = None
    miss_half_width: float | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.unit:
            raise ValueError("an outcome needs a non-empty name and unit")
        if not math.isfinite(self.target):
            raise ValueError(f"target must be finite, got {self.target}")
        if self.kind is OutcomeKind.ANGLE:
            if self.unit not in ANGLE_UNITS:
                raise ValueError(f"angle unit must be one of {ANGLE_UNITS}")
            if self.tolerance is None or not self.tolerance > 0.0:
                raise ValueError(f"angle outcome {self.name!r} needs a tolerance > 0")


@dataclass(frozen=True)
class IntegratorSignature:
    """Name and step of a time integrator; ``step_s=None`` means adaptive."""

    name: str
    step_s: float | None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("an integrator needs a name")
        if self.step_s is not None and not (
            math.isfinite(self.step_s) and self.step_s > 0.0
        ):
            raise ValueError(f"step_s must be positive and finite, got {self.step_s}")


def require_independent_integrator(
    replay: IntegratorSignature, generator: IntegratorSignature
) -> None:
    """Reject a replay that shares the generator's integrator *or* step (R13).

    Raises ``ValueError`` ("inverse crime") if the names match case-insensitively
    or both steps are fixed and equal to ``1e-9`` relative; an adaptive step
    never equals a fixed one.
    """
    same_name = replay.name.strip().lower() == generator.name.strip().lower()
    same_step = (
        replay.step_s is not None
        and generator.step_s is not None
        and math.isclose(replay.step_s, generator.step_s, rel_tol=_STEP_RTOL)
    )
    if same_name or same_step:
        raise ValueError(
            f"inverse crime: replay integrator {replay!r} shares the "
            f"{'name' if same_name else 'step'} of the generator {generator!r}; "
            "R13 requires a different integrator and step"
        )


class NoiseRelation(str, Enum):
    """How the generator's noise relates to the noise the campaign assumes."""

    IDENTICAL = "identical"
    MISMATCHED = "mismatched"


@dataclass(frozen=True)
class NoiseComparison:
    """``relation`` plus a human-readable list of the declared differences."""

    relation: NoiseRelation
    differences: tuple[str, ...]


def _spec_differences(assumed: DisturbanceSpec, other: DisturbanceSpec) -> list[str]:
    out: list[str] = []
    for field in (
        "stage",
        "channels",
        "units",
        "spectrum",
        "persistent",
        "bandwidth_hz",
        "band_hz",
        "signal_gain",
        "signal_bounds",
    ):
        if getattr(assumed, field) != getattr(other, field):
            out.append(f"{assumed.name}: {field} differs")
    same_shape = assumed.covariance.shape == other.covariance.shape
    if not same_shape or not np.allclose(
        assumed.covariance, other.covariance, rtol=1e-12, atol=0.0
    ):
        out.append(f"{assumed.name}: covariance differs")
    return out


def compare_noise(
    assumed: Sequence[DisturbanceSpec], generator: Sequence[DisturbanceSpec]
) -> NoiseComparison:
    """Compare the campaign's assumed disturbances with the generator's (by name).

    ``IDENTICAL`` only when every spec is present in both with the same stage,
    channels, units, spectrum, signal law and covariance.
    """
    mine = {s.name: s for s in assumed}
    theirs = {s.name: s for s in generator}
    diffs = [f"{n}: absent from the generator" for n in mine if n not in theirs]
    diffs += [f"{n}: absent from the campaign" for n in theirs if n not in mine]
    for name in mine.keys() & theirs.keys():
        diffs += _spec_differences(mine[name], theirs[name])
    relation = NoiseRelation.MISMATCHED if diffs else NoiseRelation.IDENTICAL
    return NoiseComparison(relation, tuple(sorted(diffs)))


@dataclass(frozen=True)
class GeneratorSpec:
    """How a synthetic ("truth") data set was generated, for the R13 gates."""

    integrator: IntegratorSignature
    noise: tuple[DisturbanceSpec, ...]


@dataclass(frozen=True)
class CampaignSpec:
    """Everything strategy-independent about a campaign.

    ``n_trials`` paired trials share draws across strategies; ``n_holdout``
    trials use the independent ``holdout_seed``.  ``dt`` and ``n_steps`` are the
    disturbance grid (white draws have covariance ``S / dt`` per step, V6).

    Raises:
        ValueError: Fewer than two trials, no hold-out trials, equal seeds,
            duplicate disturbance or outcome names, a non-positive ``dt``, or
            a generator whose noise is identical to the assumed noise.
    """

    disturbances: tuple[DisturbanceSpec, ...]
    n_steps: int
    dt: float
    n_trials: int
    n_holdout: int
    paired_seed: int
    holdout_seed: int
    outcomes: tuple[OutcomeChannel, ...]
    generator: GeneratorSpec | None = None

    def __post_init__(self) -> None:
        if len(self.disturbances) == 0:
            raise ValueError("at least one disturbance (DisturbanceSpec) is required")
        names = [d.name for d in self.disturbances]
        if len(set(names)) != len(names):
            raise ValueError("disturbance names must be unique")
        if len(self.outcomes) == 0:
            raise ValueError("at least one outcome is required")
        out_names = [o.name for o in self.outcomes]
        if len(set(out_names)) != len(out_names):
            raise ValueError("outcome names must be unique")
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {self.n_steps}")
        if not (math.isfinite(self.dt) and self.dt > 0.0):
            raise ValueError(f"dt must be positive and finite, got {self.dt}")
        if self.n_trials < 2:
            raise ValueError(f"n_trials must be >= 2, got {self.n_trials}")
        if self.n_holdout < 1:
            raise ValueError(f"n_holdout must be >= 1, got {self.n_holdout}")
        if min(self.paired_seed, self.holdout_seed) < 0:
            raise ValueError("seeds must be non-negative integers")
        if self.paired_seed == self.holdout_seed:
            raise ValueError("hold-out seed must differ from the paired seed")
        if self.generator is not None:
            cmp_ = compare_noise(self.disturbances, self.generator.noise)
            if cmp_.relation is NoiseRelation.IDENTICAL:
                raise ValueError(
                    "circular: the generator noise is identical to the assumed "
                    "noise (R13 needs mismatched noise for synthetic recovery)"
                )

    def disturbance(self, name: str) -> DisturbanceSpec:
        """The declared disturbance called ``name`` (``ValueError`` if unknown)."""
        for spec in self.disturbances:
            if spec.name == name:
                return spec
        raise ValueError(f"unknown disturbance {name!r}")

    def trials_in(self, seed_set: str) -> int:
        """Number of trials of ``seed_set`` (``paired`` or ``holdout``)."""
        if seed_set not in SEED_SETS:
            raise ValueError(f"seed_set must be one of {SEED_SETS}, got {seed_set!r}")
        return self.n_trials if seed_set == "paired" else self.n_holdout

    def base_seed(self, seed_set: str) -> int:
        """Base seed of ``seed_set``."""
        self.trials_in(seed_set)
        return self.paired_seed if seed_set == "paired" else self.holdout_seed


@dataclass(frozen=True)
class TrialDraw:
    """The random inputs of one trial; independent of the strategy.

    ``realise(name)`` returns ``(n_steps, n_channels)`` for a per-step
    disturbance and ``(n_channels,)`` for a persistent one, in the
    disturbance's units.  For a signal-dependent (M1) disturbance the underlying
    standard normals are the same for every ``signal``, so CRN holds across
    strategies that spend different effort.
    """

    spec: CampaignSpec
    seed_set: str
    index: int

    def __post_init__(self) -> None:
        n = self.spec.trials_in(self.seed_set)
        if not 0 <= self.index < n:
            raise ValueError(f"index must be in [0, {n}), got {self.index}")

    def child_seed(self, disturbance: str) -> int:
        """Seed of ``disturbance`` for this trial (pure in its three keys)."""
        position = [d.name for d in self.spec.disturbances].index(disturbance)
        seq = np.random.SeedSequence(
            [self.spec.base_seed(self.seed_set), self.index, position]
        )
        return int(seq.generate_state(1, dtype=np.uint64)[0])

    def realise(self, name: str, signal: ArrayLike | None = None) -> Array:
        """Draw disturbance ``name`` for this trial (see the class docstring)."""
        dist = self.spec.disturbance(name)
        out = dist.sample(
            1,
            self.spec.n_steps,
            self.spec.dt,
            seed=self.child_seed(name),
            signal=signal,
        )
        return np.asarray(out[0])
