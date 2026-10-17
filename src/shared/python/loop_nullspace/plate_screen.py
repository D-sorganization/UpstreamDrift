"""Family-wise screen that eliminates candidates inconsistent with force plates (OF-12).

A *candidate* is a motion and model, sampled as frames ``(index, snapshot,
decomposition)``.  Each frame's measured ground reaction (a vertical-cell
plate observation, OF-11, supplied by an :data:`ObservationProvider`) either
fits inside the candidate's feasible set or it does not.  The screen answers
one question over the whole trajectory: **can the data rule this candidate
out?**

Statistics (binding, epic #13061):

* **Family-wise control.**  Every tested row of every frame is one test.  With
  ``M`` rows the per-row band is the Bonferroni coverage
  ``k = Phi^-1(1 - alpha / (2 M))``, so a candidate that is right is eliminated
  with probability at most ``alpha`` (Gaussian rows at the declared sigma),
  whatever the dependence between frames and rows.
* **Common-mode clock offset.**  The offset between the plate clock and the
  model clock is one unknown shared by every frame.  It is *profiled*: the
  screen evaluates each declared shift and the candidate survives when **any**
  shift makes every frame consistent.  The false-elimination bound then holds
  for a true offset on the grid; a grid spanning ``r`` offset sigmas adds
  ``2 Phi(-r)`` for an offset outside it.
* **Joint verdict decides.**  A frame is consistent when the candidate's full
  feasible set (declared hypotheses, friction cones when declared, and the
  observation bands) is non-empty.  The observation-only verdict (weighted
  least squares over the whole null space, ignoring cones and capacity) is
  reported beside it for diagnosis.

Non-claims: elimination is conditional on the declared model, inertial
parameters, friction coefficient, uncertainties, coverage and alignment.  A
surviving candidate is **not** certified; infinitely many solutions may remain.
Results from unqualified plates are unqualified (deferred validation DV-13061).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any

from src.shared.python.loop_nullspace.capacity import CapacityVector
from src.shared.python.loop_nullspace.contracts import ClosedLoopSnapshot
from src.shared.python.loop_nullspace.decomposition import Decomposition
from src.shared.python.loop_nullspace.feasible_set import FeasibleSet
from src.shared.python.loop_nullspace.ground_contact import (
    GRF_ID,
    GroundObservation,
    foot_friction_cones,
    observation_accounting,
)
from src.shared.python.loop_nullspace.registry import HypothesisSet

__all__ = [
    "FamilyWiseRule",
    "FrameVerdict",
    "ObservationProvider",
    "OffsetVerdict",
    "PlateConsistencyReport",
    "ProviderFactory",
    "ScreenFrame",
    "ScreenSetup",
    "offset_grid",
    "plate_consistency",
]

#: Supplies a frame's ground observation, or ``None`` when no data cover it.
ObservationProvider = Callable[[int, ClosedLoopSnapshot], "GroundObservation | None"]
#: Builds the provider for one profiled clock-offset shift [s].
ProviderFactory = Callable[[float], ObservationProvider]

_CONE_ID = "H-FRICTION-CONE"
_CAPACITY_ID = "H-TORQUE-CAPACITY"
_DECIDING = "joint"


@dataclass(frozen=True)
class FamilyWiseRule:
    """Declared family-wise false-elimination level ``alpha`` (``0 < alpha < 1``)."""

    alpha: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.alpha) or not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1), got {self.alpha!r}")

    def coverage(self, n_rows: int) -> float:
        """Bonferroni per-row coverage ``Phi^-1(1 - alpha / (2 n_rows))``."""
        if n_rows < 1:
            raise ValueError(f"n_rows must be at least 1, got {n_rows}")
        return NormalDist().inv_cdf(1.0 - self.alpha / (2.0 * n_rows))


@dataclass(frozen=True)
class ScreenFrame:
    """One frame of a candidate: its index, snapshot and decomposition."""

    index: int
    snapshot: ClosedLoopSnapshot
    decomposition: Decomposition

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, ClosedLoopSnapshot):
            raise TypeError("snapshot must be a ClosedLoopSnapshot")
        if not isinstance(self.decomposition, Decomposition):
            raise TypeError("decomposition must be a Decomposition")


@dataclass(frozen=True)
class ScreenSetup:
    """The declared rule, hypotheses, profiled offset shifts and optional capacity.

    ``hypotheses`` must declare ``H-GRF-OBSERVATION``; friction cones are added
    for frames with foot contacts when ``H-FRICTION-CONE`` is declared, and
    ``capacity`` is required exactly when ``H-TORQUE-CAPACITY`` is.
    ``offsets_s`` are distinct finite shifts (see :func:`offset_grid`).
    """

    rule: FamilyWiseRule
    hypotheses: HypothesisSet
    offsets_s: tuple[float, ...]
    capacity: CapacityVector | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.rule, FamilyWiseRule):
            raise TypeError("rule must be a FamilyWiseRule")
        if not isinstance(self.hypotheses, HypothesisSet):
            raise TypeError("hypotheses must be a HypothesisSet")
        declared = self.hypotheses.hypothesis_ids()
        if GRF_ID not in declared:
            raise ValueError(f"hypotheses must declare {GRF_ID}")
        if (_CAPACITY_ID in declared) != (self.capacity is not None):
            raise ValueError(
                f"capacity is required exactly when {_CAPACITY_ID} is declared"
            )
        shifts = tuple(float(s) for s in self.offsets_s)
        if not shifts or not all(math.isfinite(s) for s in shifts):
            raise ValueError("offsets_s must be a non-empty tuple of finite shifts")
        if len(set(shifts)) != len(shifts):
            raise ValueError("offsets_s must be distinct")
        object.__setattr__(self, "offsets_s", shifts)

    def declares(self, hypothesis_id: str) -> bool:
        """Whether the declared hypotheses include ``hypothesis_id``."""
        return hypothesis_id in self.hypotheses.hypothesis_ids()


@dataclass(frozen=True)
class FrameVerdict:
    """One frame at one shift.  ``tested`` is false when no data covered it."""

    index: int
    tested: bool
    n_rows: int = 0
    rank_visible: int = 0
    chi2: float = 0.0
    max_normalised_residual: float = 0.0
    observation_consistent: bool = True
    joint_feasible: bool = True
    conflict: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible form."""
        return {**dataclasses.asdict(self), "conflict": list(self.conflict)}


@dataclass(frozen=True)
class OffsetVerdict:
    """Every frame at one profiled clock-offset shift, at the family-wise coverage."""

    shift_s: float
    coverage: float
    n_rows: int
    frames: tuple[FrameVerdict, ...]

    @property
    def consistent(self) -> bool:
        """True when every tested frame is jointly feasible."""
        return all(f.joint_feasible for f in self.frames)

    @property
    def first_eliminating_index(self) -> int | None:
        """Index of the first jointly infeasible frame, or ``None``."""
        return next((f.index for f in self.frames if not f.joint_feasible), None)

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible form."""
        return {
            "shift_s": self.shift_s,
            "coverage": self.coverage,
            "n_rows": self.n_rows,
            "consistent": self.consistent,
            "first_eliminating_index": self.first_eliminating_index,
            "frames": [f.to_dict() for f in self.frames],
        }


@dataclass(frozen=True)
class PlateConsistencyReport:
    """The screen's answer for one candidate.

    ``eliminated`` is true only when data were tested and no profiled shift
    leaves every frame jointly feasible.  ``deciding_verdict`` is ``"joint"``.
    """

    alpha: float
    offsets: tuple[OffsetVerdict, ...]
    deciding_verdict: str = field(default=_DECIDING)

    @property
    def tested(self) -> bool:
        """True when at least one row was tested at some shift."""
        return any(o.n_rows for o in self.offsets)

    @property
    def eliminated(self) -> bool:
        """True when the data rule the candidate out at every profiled shift."""
        return self.tested and not any(o.consistent for o in self.offsets)

    @property
    def surviving_shifts_s(self) -> tuple[float, ...]:
        """Shifts at which every tested frame is jointly feasible."""
        return tuple(o.shift_s for o in self.offsets if o.consistent and o.n_rows)

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible form."""
        return {
            "alpha": self.alpha,
            "deciding_verdict": self.deciding_verdict,
            "tested": self.tested,
            "eliminated": self.eliminated,
            "surviving_shifts_s": list(self.surviving_shifts_s),
            "offsets": [o.to_dict() for o in self.offsets],
        }


def offset_grid(
    *, sigma_offset_s: float, reach_sigmas: float, max_step_s: float
) -> tuple[float, ...]:
    """Symmetric shifts spanning ``+/- reach_sigmas * sigma_offset_s``, including 0.

    The step is at most ``max_step_s`` (use the plate's minimum sample
    interval).  A zero offset uncertainty gives the single shift ``0``.
    """
    for name, value in (("reach_sigmas", reach_sigmas), ("max_step_s", max_step_s)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite, got {value!r}")
    if not math.isfinite(sigma_offset_s) or sigma_offset_s < 0.0:
        raise ValueError(
            f"sigma_offset_s must be finite and >= 0, got {sigma_offset_s!r}"
        )
    reach = reach_sigmas * sigma_offset_s
    steps = math.ceil(reach / max_step_s)
    if steps == 0:
        return (0.0,)
    step = reach / steps
    return tuple(i * step for i in range(-steps, steps + 1))


def plate_consistency(
    frames: Sequence[ScreenFrame],
    providers: ProviderFactory,
    setup: ScreenSetup,
) -> PlateConsistencyReport:
    """Screen one candidate's ``frames`` against plate data at every profiled shift.

    Preconditions: ``frames`` non-empty with unique indices and of type
    :class:`ScreenFrame`; ``setup`` a :class:`ScreenSetup` (``TypeError`` /
    ``ValueError``).  Postconditions: one :class:`OffsetVerdict` per shift in
    ``setup.offsets_s`` order; a frame whose provider returns ``None`` is
    untested, never fabricated.
    """
    _check_frames(frames)
    if not isinstance(setup, ScreenSetup):
        raise TypeError("setup must be a ScreenSetup")
    verdicts = tuple(
        _offset_verdict(frames, providers(shift), setup, shift)
        for shift in setup.offsets_s
    )
    return PlateConsistencyReport(alpha=setup.rule.alpha, offsets=verdicts)


def _check_frames(frames: Sequence[ScreenFrame]) -> None:
    if not frames:
        raise ValueError("frames must not be empty")
    if not all(isinstance(f, ScreenFrame) for f in frames):
        raise TypeError("frames must be ScreenFrame instances")
    indices = [f.index for f in frames]
    if len(set(indices)) != len(indices):
        raise ValueError("frame indices must be unique")


def _offset_verdict(
    frames: Sequence[ScreenFrame],
    provider: ObservationProvider,
    setup: ScreenSetup,
    shift_s: float,
) -> OffsetVerdict:
    observations = {f.index: provider(f.index, f.snapshot) for f in frames}
    n_rows = sum(o.n_rows for o in observations.values() if o is not None)
    coverage = setup.rule.coverage(n_rows) if n_rows else math.inf
    verdicts = tuple(
        _frame_verdict(frame, observations[frame.index], setup, coverage)
        for frame in frames
    )
    return OffsetVerdict(shift_s, coverage, n_rows, verdicts)


def _frame_verdict(
    frame: ScreenFrame,
    observation: GroundObservation | None,
    setup: ScreenSetup,
    coverage: float,
) -> FrameVerdict:
    if observation is None:
        return FrameVerdict(frame.index, tested=False)
    banded = dataclasses.replace(observation, coverage=coverage)
    accounting = observation_accounting(frame.decomposition, banded)
    feasible = _joint_set(frame, banded, setup)
    empty = feasible.is_empty()
    return FrameVerdict(
        index=frame.index,
        tested=True,
        n_rows=banded.n_rows,
        rank_visible=accounting.rank_visible,
        chi2=accounting.chi2,
        max_normalised_residual=accounting.max_normalised_residual,
        observation_consistent=accounting.max_normalised_residual <= coverage,
        joint_feasible=not empty,
        conflict=feasible.conflict_set() if empty else (),
    )


def _joint_set(
    frame: ScreenFrame, observation: GroundObservation, setup: ScreenSetup
) -> FeasibleSet:
    snapshot = frame.snapshot
    cones = (
        foot_friction_cones(snapshot)
        if setup.declares(_CONE_ID) and snapshot.foot_contacts
        else ()
    )
    return FeasibleSet.from_hypotheses(
        frame.decomposition,
        setup.hypotheses,
        capacity=setup.capacity,
        cones=cones,
        observation=observation,
    )
