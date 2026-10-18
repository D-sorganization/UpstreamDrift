"""Which optimizer solve, bar model and window make each lift's bundle (#11745).

LIFT-5 packages one same-input bundle per barbell lift of the parity
standard.  Every bundle follows the same pipeline
(:mod:`.bundles`): a Tools Movement Optimizer solve, full-body marker IK,
and a MuJoCo tracking window packaged as an :class:`InputBundle`.  This
module records, per lift, the inputs of that pipeline and why they were
chosen, so the generator (``scripts/generate_lift_bundle.py``) and the tests
read one table:

* the back squat keeps its committed recipe (``full_squat``, bar on a
  torso shelf, window centred on the bottom);
* the deadlift and the clean-and-jerk are pulls: the bar hangs in the hands
  (the optimizer's ``deadlift`` bar model) and the window is centred on the
  knee pass;
* the bench press lies supine on a bench contact body (:mod:`.bench_press`):
  the optimizer's arm-chain bar model is applied to the model's shoulders and
  the window is centred on the bottom (the bar at the chest);
* the snatch is a pull with a phase-correct bar: the Tools snatch
  configuration pins a via pose at each phase start of the standard, the bar
  hangs in the hands through both pulls and is overhead from the catch on
  (``bar_convention="snatch"``, :func:`movement_optimizer.exercises.snatch.
  snatch_bar_path`), and the window is centred on the knee pass.

:data:`DEFERRED_LIFTS` lists any lift without a credible trajectory from this
pipeline, with the measured reason.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from src.shared.python.lifting.pack_audit.names import LIFTS
from src.shared.python.lifting.sagittal_retarget import SagittalSolution

Array = NDArray[np.float64]

_REPO = Path(__file__).resolve().parents[4]
EVIDENCE_DIR = _REPO / "docs/development/lifting/evidence"
#: Window events a recipe may centre its bundle on.
WINDOW_EVENTS: tuple[str, ...] = ("bottom", "knee_pass")
#: Bar conventions whose bar is held in the hands (the ``snatch`` bar hangs
#: through the pulls and is overhead from the catch on).
HAND_HELD_CONVENTIONS: tuple[str, ...] = ("deadlift", "snatch")


@dataclass(frozen=True)
class LiftRecipe:
    """Inputs of one lift's same-input bundle.

    Attributes:
        lift: Parity-standard lift key (grip width, phases).
        bundle_name: Evidence directory and bundle file stem.
        exercise: Movement Optimizer exercise that is solved.
        bar_convention: Optimizer exercise whose bar model places the bar.
        window_event: Solution event the bundle window is centred on.
        phase: Standard phase the window covers.
        duration_s: Optimizer movement duration.
        bar_mass_kg: Bar load given to the optimizer.
        rationale: Why this solve and bar model represent the lift.
    """

    lift: str
    bundle_name: str
    exercise: str
    bar_convention: str
    window_event: str
    phase: str
    duration_s: float
    bar_mass_kg: float
    rationale: str

    def __post_init__(self) -> None:
        if self.lift not in LIFTS:
            raise ValueError(f"lift must be one of {LIFTS}: {self.lift!r}")
        if self.window_event not in WINDOW_EVENTS:
            raise ValueError(f"window_event must be one of {WINDOW_EVENTS}")
        if not (self.duration_s > 0.0 and self.bar_mass_kg > 0.0):
            raise ValueError("duration_s and bar_mass_kg must be positive")

    @property
    def bundle_path(self) -> Path:
        """Committed bundle of this lift."""
        return EVIDENCE_DIR / self.bundle_name / f"{self.bundle_name}_bundle.npz"

    @property
    def parity_record_path(self) -> Path:
        """Committed cross-engine bar-path parity record of this lift."""
        return EVIDENCE_DIR / self.bundle_name / "bar_path_parity.json"

    @property
    def hand_held(self) -> bool:
        """True when the bar is held in the hands rather than on a torso shelf."""
        return self.bar_convention in HAND_HELD_CONVENTIONS

    @property
    def supine(self) -> bool:
        """True when the lifter lies on the bench contact body (#11745)."""
        return self.bar_convention == "bench_press"


_PULL_RATIONALE = (
    "The bar hangs in the hands during a pull, so its path is the optimizer's "
    "deadlift bar model (one arm length straight below the shoulder) applied "
    "to the model's shoulders."
)

RECIPES: Mapping[str, LiftRecipe] = MappingProxyType(
    {
        "squat": LiftRecipe(
            lift="squat",
            bundle_name="back_squat",
            exercise="full_squat",
            bar_convention="full_squat",
            window_event="bottom",
            phase="bottom",
            duration_s=3.0,
            bar_mass_kg=100.0,
            rationale="Bar on the upper back; committed by #12245/#12534.",
        ),
        "deadlift": LiftRecipe(
            lift="deadlift",
            bundle_name="deadlift",
            exercise="deadlift",
            bar_convention="deadlift",
            window_event="knee_pass",
            phase="knee_pass",
            duration_s=2.0,
            bar_mass_kg=100.0,
            rationale=_PULL_RATIONALE,
        ),
        "clean_and_jerk": LiftRecipe(
            lift="clean_and_jerk",
            bundle_name="clean_and_jerk",
            exercise="clean",
            bar_convention="deadlift",
            window_event="knee_pass",
            phase="first_pull",
            duration_s=1.2,
            bar_mass_kg=100.0,
            rationale=(
                _PULL_RATIONALE
                + " The optimizer's clean bar model keeps the bar on the "
                "shoulders for the whole lift (front rack), which is not where "
                "it is during the pull. The window covers the first pull only; "
                "the rack, the jerk and the overhead phases are not in it."
            ),
        ),
        "snatch": LiftRecipe(
            lift="snatch",
            bundle_name="snatch",
            exercise="snatch",
            bar_convention="snatch",
            window_event="knee_pass",
            phase="first_pull",
            duration_s=2.0,
            bar_mass_kg=100.0,
            rationale=(
                "The Tools snatch configuration pins a via pose at the start "
                "of each phase of the standard (knee pass at 0.30, full "
                "extension at 0.45, catch at 0.55 held to 0.70), so the "
                "shoulder rises through the first and second pulls before the "
                "lifter drops under the bar into a catch with the standard's "
                "trunk lean. Its phase-correct bar hangs in the hands through "
                "the pulls, turns over across the turnover and is overhead "
                "from the catch on, applied to the model's shoulders. The "
                "window spans the knee pass, from the end of the first pull "
                "into the second; the turnover, catch and recovery are not in "
                "it, and the full-body IK does not follow them (#11745)."
            ),
        ),
        "bench_press": LiftRecipe(
            lift="bench_press",
            bundle_name="bench_press",
            exercise="bench_press",
            bar_convention="bench_press",
            window_event="bottom",
            phase="chest_touch",
            duration_s=2.0,
            bar_mass_kg=100.0,
            rationale=(
                "Supine on a bench contact body: four back spheres (OSL-2 "
                "geometry) press on a raised support and the feet stand on "
                "the floor. The bar path is the optimizer's bench arm-chain "
                "bar model (arm tip relative to the shoulder) applied to the "
                "model's shoulders, scaled by its straight-up reach; only the "
                "arms move. The window is centred on the bottom (chest touch)."
            ),
        ),
    }
)

#: Lifts without a credible trajectory from this pipeline, with the reason.
#: Empty since the snatch got its pull phase (Tools snatch via schedule).
DEFERRED_LIFTS: Mapping[str, str] = MappingProxyType({})


def bundle_lifts() -> tuple[str, ...]:
    """Lifts with a recipe, in the standard's order."""
    return tuple(lift for lift in LIFTS if lift in RECIPES)


def recipe(lift: str) -> LiftRecipe:
    """The recipe of *lift*.

    Raises:
        ValueError: For a deferred or unknown lift (the reason is included).
    """
    if lift in DEFERRED_LIFTS:
        raise ValueError(f"{lift} has no bundle: {DEFERRED_LIFTS[lift]}")
    if lift not in RECIPES:
        raise ValueError(f"unknown lift {lift!r}; use {list(RECIPES)}")
    return RECIPES[lift]


def knee_pass_time(solution: SagittalSolution) -> float:
    """First sample time at which the bar is at or above knee height.

    Heights are the optimizer's own (ankle origin, ``up``): the knee is the
    shank tip, ``segment_lengths_m[0] * cos(shank angle)``.

    Raises:
        ValueError: If the source has no segment lengths or the bar never
            reaches the knee.
    """
    raw = solution.source.get("segment_lengths_m")
    if raw is None:
        raise ValueError("solution source has no segment_lengths_m")
    shank = float(np.asarray(raw, dtype=float)[0])
    knee = shank * np.cos(solution.segment_angles_rad[:, 0])
    above = np.flatnonzero(solution.bar_m[:, 1] >= knee)
    if above.size == 0:
        raise ValueError("the bar never reaches knee height")
    return float(solution.time_s[above[0]])


def window_centre_time(solution: SagittalSolution, event: str) -> float:
    """Time of *event* on the solution: the lowest bar or the knee pass."""
    if event == "bottom":
        return float(solution.time_s[int(np.argmin(solution.bar_m[:, 1]))])
    if event == "knee_pass":
        return knee_pass_time(solution)
    raise ValueError(f"event must be one of {WINDOW_EVENTS}: {event!r}")


def window_start_time(solution: SagittalSolution, event: str, window_s: float) -> float:
    """Solution sample nearest to ``centre - window_s / 2``, kept in the lift.

    Postcondition: the result is a sample time and ``result + window_s`` does
    not pass the end of the solution.
    """
    if not (np.isfinite(window_s) and window_s > 0.0):
        raise ValueError("window_s must be positive and finite")
    times = solution.time_s
    wanted = window_centre_time(solution, event) - window_s / 2
    latest = times[-1] - window_s + 1e-12
    candidates = times[times <= latest]
    if candidates.size == 0:
        raise ValueError("window_s is longer than the lift")
    return float(candidates[int(np.argmin(np.abs(candidates - wanted)))])
