"""A :data:`.variance.PuttModel` for one putt class on a uniform plane (#14177).

Each putt is played the way a golfer plays it: read the green (with error),
pick the line that holes the putt at the intended pace on the green *as read*,
then strike it (with error) on the *true* green.

* **Read.** A :class:`.sampling.ReadSample` scales the grade, rotates the fall
  line and scales the stimp. The perceived green is again a uniform plane.
* **Pace.** The start speed is the roll law's nominal speed for
  ``pace_ft_past`` on the perceived green (perceived rolling resistance plus
  perceived along-line grade), so a stimp or grade misread becomes a pace
  error. It is computed exactly for every putt.
* **Line.** The aim is the centre of the band of aims that hole the putt at
  that pace on the perceived green with a perfect stroke: the golfer's "make
  it at my pace" line, not the strokes-gained optimum of
  :class:`.perceived_green.PerceivedGreenPlanner`, which needs a full
  candidate search per read and is too slow for a variance study. Finding a
  line costs a few batch rolls, so lines are planned only at the nodes of a
  :class:`ReadLattice` and interpolated (bilinearly, extrapolating past the
  edge nodes) for each putt. The lattice needs only two axes: on a uniform
  plane with speed-independent rolling resistance, a green read ``s`` times
  slower rolls every putt along the same path as a green read ``s`` times
  steeper (time and speed rescale, the path does not), so the line depends on
  the read only through ``grade_ratio * stimp_ratio`` and the aspect error.
  A study plans at most nine lines per putt class, shared by every variant.
* **Strike.** Start line error is the face/path blend of
  :meth:`.sampling.ExecutionSample.start_line_error_deg`; speed is the
  planned speed times the sampled speed ratio. Every putt rolls on the true
  green in one :func:`.batch_roll.roll_batch` call.

Toe strike is not modelled: ``strike_toe_mm`` is ignored, so its Sobol
indices and its elimination gain are exactly zero.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from src.shared.python.contracts import require
from src.shared.python.core.physics_constants import GRAVITY_M_S2
from src.shared.python.putting_analytics.batch_roll import Capture, roll_batch
from src.shared.python.putting_analytics.expected_putts import (
    ExpectedPuttsCurve,
    PuttAction,
)
from src.shared.python.putting_analytics.roll_model import FT_M, BatchRollModel
from src.shared.python.putting_analytics.sampling import ExecutionSample, ReadSample
from src.shared.python.putting_analytics.skill import (
    DEFAULT_FACE_WEIGHT,
    ReadDispersion,
)
from src.shared.python.putting_analytics.variance import StrokeOutcomes
from src.shared.python.putting_dynamics.friction import (
    DEFAULT_STATIC_MU,
    FrictionParams,
    stimp_to_rolling_mu,
)
from src.shared.python.putting_dynamics.surfaces import HeightField, SurfaceSpec

COARSE_AIMS_DEG = tuple(float(a) for a in np.arange(-24.0, 24.01, 1.0))
FINE_HALF_WIDTH_DEG = 1.0
FINE_STEP_DEG = 0.05
WARM_SPAN = 3.0  # a warm-started fan is the fine fan stretched by this factor
MAX_GRADE_PCT = 10.0  # HeightField.planar's limit; a heavy over-read is clipped
DEFAULT_LATTICE_SPAN_SD = 2.0


@dataclass(frozen=True)
class PlaneGreen:
    """Uniform planar green: grade (%), fall direction (deg CCW of the line), stimp."""

    grade_pct: float
    aspect_deg: float
    stimp: float

    def __post_init__(self) -> None:
        require(0.0 <= self.grade_pct <= MAX_GRADE_PCT, "grade_pct in [0, 10]")
        require(self.stimp > 0.0, "stimp must be positive")

    @property
    def mu_roll(self) -> float:
        """Base rolling resistance for this stimp."""
        return stimp_to_rolling_mu(self.stimp)

    def surface(self) -> SurfaceSpec:
        """The green as a :class:`SurfaceSpec`; zero grade gives the flat surface.

        The static hold is raised to the rolling resistance on greens slower
        than about stimp 7, where the default hold sits below it and the
        friction model rejects the surface; a read can easily go that slow.
        """
        mu = self.mu_roll
        friction = FrictionParams(mu_roll0=mu, mu_static=max(DEFAULT_STATIC_MU, mu))
        flat = SurfaceSpec.flat_uniform(self.stimp, friction)
        if self.grade_pct == 0.0:
            return flat
        plane = HeightField.planar(self.grade_pct, self.aspect_deg)
        return dataclasses.replace(flat, height=plane)

    def perceived(
        self, grade_ratio: float, aspect_error: float, stimp_ratio: float
    ) -> PlaneGreen:
        """The green as read: grade and stimp scaled, fall line rotated."""
        grade = min(self.grade_pct * grade_ratio, MAX_GRADE_PCT)
        aspect = (self.aspect_deg + aspect_error + 180.0) % 360.0 - 180.0
        return PlaneGreen(grade, aspect if grade else 0.0, self.stimp * stimp_ratio)


def nominal_speeds_mps(
    green: PlaneGreen, read: ReadSample, distance_ft: float, pace_ft_past: float
) -> np.ndarray:
    """Per-putt start speed for ``pace_ft_past`` on each perceived green.

    Same law as :meth:`.roll_model.BatchRollModel.nominal_speed_mps`:
    ``v0**2 = 2 g (mu_r + dh/dx) (d + pace)``, with the perceived rolling
    resistance and along-line grade. The planar green's ``dh/dx`` is
    ``-grade cos(aspect)``.
    """
    grade = np.minimum(green.grade_pct * read.grade_ratio, MAX_GRADE_PCT) / 100.0
    aspect = np.radians(green.aspect_deg + read.aspect_error_deg)
    decel = green.mu_roll / read.stimp_ratio - grade * np.cos(aspect)
    require(bool(np.all(decel > 0.0)), "a read is too steep downhill for a pace")
    roll_m = (distance_ft + pace_ft_past) * FT_M
    return np.sqrt(2.0 * GRAVITY_M_S2 * decel * roll_m)


def _nodes(center: float, sd: float, span: float) -> tuple[float, ...]:
    return (center,) if sd == 0.0 else (center - span * sd, center, center + span * sd)


@dataclass(frozen=True)
class ReadLattice:
    """Read nodes where lines are planned.

    ``log_grade`` is ``log(grade_ratio * stimp_ratio)``, the read's effective
    steepness (see the module notes); ``aspect_deg`` is the fall-line error.
    """

    log_grade: tuple[float, ...] = (0.0,)
    aspect_deg: tuple[float, ...] = (0.0,)

    def __post_init__(self) -> None:
        for axis in self.axes:
            require(len(axis) >= 1, "every lattice axis needs a node")
            require(bool(np.all(np.diff(axis) > 0.0)), "lattice nodes must increase")

    @classmethod
    def from_dispersion(
        cls, read: ReadDispersion, span_sd: float = DEFAULT_LATTICE_SPAN_SD
    ) -> ReadLattice:
        """Nodes at the mean and +/- ``span_sd`` spreads of each lattice axis.

        An axis with no spread gets a single node at its mean.
        """
        require(span_sd > 0.0, "span_sd must be positive")
        var_grade = math.log1p(read.grade_rel_sd**2)
        var_stimp = math.log1p(read.stimp_rel_sd**2)
        center = math.log1p(read.grade_rel_bias) - 0.5 * (var_grade + var_stimp)
        return cls(
            _nodes(center, math.sqrt(var_grade + var_stimp), span_sd),
            _nodes(0.0, read.aspect_sd_deg, span_sd),
        )

    @property
    def axes(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """The node axes in (effective grade, aspect) order."""
        return (self.log_grade, self.aspect_deg)


def _axis_weights(
    nodes: tuple[float, ...], x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Lower node index and linear weight of the upper node (extrapolates)."""
    if len(nodes) == 1:
        return np.zeros(x.size, dtype=int), np.zeros(x.size)
    grid = np.asarray(nodes)
    lower = np.clip(np.searchsorted(grid, x, side="right") - 1, 0, grid.size - 2)
    weight = (x - grid[lower]) / (grid[lower + 1] - grid[lower])
    return lower, weight


def interpolate(
    lattice: ReadLattice, values: np.ndarray, read: ReadSample
) -> np.ndarray:
    """Bilinear interpolation of node ``values`` (effective grade x aspect)."""
    log_grade = np.log(read.grade_ratio) + np.log(read.stimp_ratio)
    aspect = np.asarray(read.aspect_error_deg, dtype=float)
    (i, wi), (j, wj) = (
        _axis_weights(lattice.log_grade, log_grade),
        _axis_weights(lattice.aspect_deg, aspect),
    )
    i1 = np.minimum(i + 1, len(lattice.log_grade) - 1)
    j1 = np.minimum(j + 1, len(lattice.aspect_deg) - 1)
    return (1.0 - wi) * ((1.0 - wj) * values[i, j] + wj * values[i, j1]) + wi * (
        (1.0 - wj) * values[i1, j] + wj * values[i1, j1]
    )


def _centre_aim(aims: np.ndarray, holed: np.ndarray, leave_m: np.ndarray) -> float:
    """Midpoint of the holed aims, or the aim with the shortest leave if none hole."""
    if holed.any():
        hits = aims[holed]
        return float(0.5 * (hits.min() + hits.max()))
    return float(aims[int(np.argmin(leave_m))])


@dataclass
class SkillRollModel:
    """Play putts of one class: plan on the read green, roll on the true green.

    ``baseline`` prices a miss (strokes = 1 + baseline(leave) after a miss);
    ``lattice`` sets where lines are planned (build it with
    :meth:`ReadLattice.from_dispersion` from the profile under study);
    ``pace_ft_past`` is the golfer's intended pace; ``face_weight`` blends
    face and path into the start line.
    """

    green: PlaneGreen
    distance_ft: float
    baseline: ExpectedPuttsCurve
    lattice: ReadLattice = field(default_factory=ReadLattice)
    pace_ft_past: float = 1.5
    face_weight: float = DEFAULT_FACE_WEIGHT
    capture: Capture = "effective_radius"
    _aims: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        require(self.distance_ft > 0.0, "distance_ft must be positive")
        require(self.pace_ft_past >= 0.0, "pace_ft_past must be non-negative")
        require(0.0 <= self.face_weight <= 1.0, "face_weight must be in [0, 1]")
        self._true_surface = self.green.surface()

    def _fan(self, model: BatchRollModel, aims: np.ndarray) -> tuple[float, bool]:
        """Centre aim of the fan, and whether its holed band lies inside the fan."""
        actions = [PuttAction(float(a), self.pace_ft_past) for a in aims]
        result = model.roll_many(self.distance_ft, actions, 1, 0)
        leave = np.hypot(result.rest_x_m - self.distance_ft * FT_M, result.rest_y_m)
        holed = result.holed
        inside = bool(holed.any() and not holed[0] and not holed[-1])
        return _centre_aim(aims, holed, leave), inside

    def line(self, read: PlaneGreen, guess: float | None = None) -> float:
        """Aim (deg) that holes the putt at the intended pace on green ``read``.

        With a ``guess`` (a neighbouring node's line) one fine fan around it
        usually brackets the holed band; otherwise, or if it does not, a coarse
        fan locates the band and a fine fan centres it.
        """
        model = BatchRollModel(read.surface(), capture=self.capture)
        fine = np.arange(
            -FINE_HALF_WIDTH_DEG, FINE_HALF_WIDTH_DEG + 1e-9, FINE_STEP_DEG
        )
        if guess is not None:
            aim, inside = self._fan(model, guess + WARM_SPAN * fine)
            if inside:
                return aim
        coarse, _ = self._fan(model, np.asarray(COARSE_AIMS_DEG))
        return self._fan(model, coarse + fine)[0]

    def node_aims(self) -> np.ndarray:
        """Planned aim at every lattice node (computed once, then cached).

        Each node after the first is warm-started from its planned neighbour.
        Every line on a flat green is straight at the hole, so none is rolled.
        """
        if self._aims is None:
            log_grade, aspect = self.lattice.axes
            aims = np.zeros((len(log_grade), len(aspect)))
            if self.green.grade_pct == 0.0:
                self._aims = aims
                return aims
            for i, j in itertools.product(range(len(log_grade)), range(len(aspect))):
                read = self.green.perceived(math.exp(log_grade[i]), aspect[j], 1.0)
                if i == j == 0:
                    guess = None
                else:
                    guess = float(aims[i, j - 1] if j > 0 else aims[i - 1, j])
                aims[i, j] = self.line(read, guess)
            self._aims = aims
        return self._aims

    def __call__(self, execution: ExecutionSample, read: ReadSample) -> StrokeOutcomes:
        """Strokes, holed flag and leave (m) for every sampled putt."""
        aim = interpolate(self.lattice, self.node_aims(), read)
        speed0 = nominal_speeds_mps(
            self.green, read, self.distance_ft, self.pace_ft_past
        )
        heading = np.radians(aim + execution.start_line_error_deg(self.face_weight))
        speed = speed0 * execution.speed_ratio
        zeros = np.zeros(speed.size)
        start = np.stack(
            (zeros, zeros, speed * np.cos(heading), speed * np.sin(heading), speed)
        )
        hole = (self.distance_ft * FT_M, 0.0)
        result = roll_batch(start, self._true_surface, hole, self.capture)
        miss_m = np.hypot(result.rest_x_m - hole[0], result.rest_y_m)
        leave_m = np.where(result.holed, 0.0, miss_m)
        strokes = 1.0 + np.where(result.holed, 0.0, self.baseline(leave_m / FT_M))
        return StrokeOutcomes(strokes, result.holed, leave_m)
