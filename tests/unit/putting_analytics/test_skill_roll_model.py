"""Skill-profile roll model: plan on the read green, roll on the true one (#14177)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.shared.python.putting_analytics.expected_putts import ExpectedPuttsCurve
from src.shared.python.putting_analytics.roll_model import BatchRollModel
from src.shared.python.putting_analytics.sampling import (
    ExecutionSample,
    ReadSample,
    sample_execution,
    sample_read,
)
from src.shared.python.putting_analytics.skill import (
    PuttingSkillProfile,
    ReadDispersion,
    StrokeDispersion,
)
from src.shared.python.putting_analytics.skill_roll_model import (
    FINE_STEP_DEG,
    PlaneGreen,
    ReadLattice,
    SkillRollModel,
    interpolate,
    nominal_speeds_mps,
)
from src.shared.python.putting_analytics.variance import (
    AttributionConfig,
    elimination_gains,
)

pytestmark = pytest.mark.unit

BASELINE = ExpectedPuttsCurve(
    np.array([0.5, 3.0, 10.0, 30.0, 60.0]), np.array([1.0, 1.1, 1.5, 1.9, 2.2])
)
FLAT = PlaneGreen(0.0, 0.0, 10.0)
RIGHT_TO_LEFT = PlaneGreen(2.0, 90.0, 10.0)  # falls toward the golfer's left (+y)
N = 64


def _perfect_strike(n: int = N) -> ExecutionSample:
    zeros = np.zeros(n)
    return ExecutionSample(zeros, zeros, np.ones(n), zeros)


def _perfect_read(n: int = N) -> ReadSample:
    return ReadSample(np.ones(n), np.zeros(n), np.ones(n))


def test_lattice_combines_grade_and_stimp_reads() -> None:
    lattice = ReadLattice.from_dispersion(ReadDispersion(0.2, 4.0, 0.1), span_sd=2.0)
    var = np.log1p(0.2**2) + np.log1p(0.1**2)
    sd = np.sqrt(var)
    np.testing.assert_allclose(
        lattice.log_grade, np.array([-2 * sd, 0.0, 2 * sd]) - 0.5 * var
    )
    assert lattice.aspect_deg == (-8.0, 0.0, 8.0)
    assert ReadLattice.from_dispersion(ReadDispersion()).axes == ((0.0,), (0.0,))


def test_interpolation_is_exact_at_nodes_and_linear_beyond() -> None:
    lattice = ReadLattice((-0.2, 0.0, 0.2), (-4.0, 0.0, 4.0))
    g, a = np.meshgrid(lattice.log_grade, lattice.aspect_deg, indexing="ij")
    values = 3.0 * g - 0.5 * a  # linear, so interpolation is exact
    log_g = np.array([-0.2, 0.1, 0.35, -0.5])
    log_s = np.array([0.0, 0.05, -0.1, 0.1])
    aspect = np.array([4.0, -2.0, 7.0, -9.0])
    read = ReadSample(np.exp(log_g), aspect, np.exp(log_s))
    np.testing.assert_allclose(
        interpolate(lattice, values, read), 3.0 * (log_g + log_s) - 0.5 * aspect
    )


def test_a_slow_read_breaks_like_a_shallow_read() -> None:
    """The similarity that lets one lattice axis cover both grade and stimp.

    Reading the green slow means hitting it harder, so the ball breaks less:
    a 0.75x stimp read plans the same line as a 0.75x grade read.
    """
    model = SkillRollModel(RIGHT_TO_LEFT, 12.0, BASELINE)
    slow = model.line(RIGHT_TO_LEFT.perceived(1.0, 0.0, 0.75))
    shallow = model.line(RIGHT_TO_LEFT.perceived(0.75, 0.0, 1.0))
    assert slow == pytest.approx(shallow, abs=2 * FINE_STEP_DEG)
    assert abs(slow) < abs(model.line(RIGHT_TO_LEFT)) - 0.5


@pytest.mark.parametrize("ratios", [(1.0, 0.0, 1.0), (1.4, 25.0, 0.8), (0.5, -40, 1.3)])
def test_nominal_speed_matches_the_roll_model(ratios: tuple[float, ...]) -> None:
    grade, aspect, stimp = ratios
    read = ReadSample(np.array([grade]), np.array([aspect]), np.array([stimp]))
    speed = nominal_speeds_mps(RIGHT_TO_LEFT, read, 12.0, 1.5)[0]
    model = BatchRollModel(RIGHT_TO_LEFT.perceived(grade, aspect, stimp).surface())
    assert speed == pytest.approx(model.nominal_speed_mps(12.0, 1.5), rel=1e-9)


def test_perfect_golfer_holes_a_straight_putt() -> None:
    model = SkillRollModel(FLAT, 6.0, BASELINE)
    out = model(_perfect_strike(), _perfect_read())
    assert out.holed.all() and np.all(out.strokes == 1.0) and np.all(out.leave_m == 0)
    assert model.node_aims().shape == (1, 1)
    assert model.line(FLAT) == pytest.approx(0.0, abs=FINE_STEP_DEG)


def test_breaking_putt_is_aimed_high_and_holed() -> None:
    model = SkillRollModel(RIGHT_TO_LEFT, 10.0, BASELINE)
    out = model(_perfect_strike(), _perfect_read())
    aim = model.line(RIGHT_TO_LEFT)
    assert aim < -0.5, "a right-to-left putt starts right of the hole"
    assert out.holed.all()


def test_under_reading_the_break_misses_low() -> None:
    lattice = ReadLattice(log_grade=(float(np.log(0.4)), 0.0))
    model = SkillRollModel(RIGHT_TO_LEFT, 10.0, BASELINE, lattice)
    half_read = ReadSample(np.full(N, 0.4), np.zeros(N), np.ones(N))
    out = model(_perfect_strike(), half_read)
    assert not out.holed.any() and np.all(out.strokes > 1.0)
    low, true = model.node_aims()[:, 0]
    assert low > true, "an under-read line starts closer to the hole"


def test_reading_the_green_slow_hits_it_long() -> None:
    model = SkillRollModel(FLAT, 10.0, BASELINE)  # stimp 4 read: below default hold
    out = model(_perfect_strike(), ReadSample(np.ones(N), np.zeros(N), np.full(N, 0.4)))
    assert not out.holed.any() and np.all(out.leave_m > 0.3)


def test_toe_strike_is_ignored() -> None:
    model = SkillRollModel(RIGHT_TO_LEFT, 12.0, BASELINE)
    execution = sample_execution(StrokeDispersion(1.0, 1.0, 0.05), N, 3)
    toed = dataclasses.replace(execution, strike_toe_mm=np.full(N, 9.0))
    a = model(execution, _perfect_read())
    b = model(toed, _perfect_read())
    np.testing.assert_array_equal(a.strokes, b.strokes)


def test_lines_are_planned_once_per_node(monkeypatch: pytest.MonkeyPatch) -> None:
    dispersion = ReadDispersion(0.2, 4.0, 0.08)
    model = SkillRollModel(
        RIGHT_TO_LEFT, 12.0, BASELINE, ReadLattice.from_dispersion(dispersion)
    )
    planned: list[PlaneGreen] = []
    line = model.line

    def counting_line(read: PlaneGreen, guess: float | None = None) -> float:
        planned.append(read)
        return line(read, guess)

    monkeypatch.setattr(model, "line", counting_line)
    read = sample_read(dispersion, 200, 4)
    first = model(_perfect_strike(200), read)
    second = model(_perfect_strike(200), read)
    assert len(planned) == 9
    np.testing.assert_array_equal(first.strokes, second.strokes)


def test_elimination_gains_through_the_variance_study() -> None:
    profile = PuttingSkillProfile(
        "mid",
        StrokeDispersion(face_sd_deg=1.5, path_sd_deg=2.0, speed_cv=0.08),
        ReadDispersion(grade_rel_sd=0.25, aspect_sd_deg=5.0, stimp_rel_sd=0.08),
    )
    lattice = ReadLattice.from_dispersion(profile.read)
    model = SkillRollModel(RIGHT_TO_LEFT, 12.0, BASELINE, lattice)
    config = AttributionConfig(
        factors=("face", "strike", "read_grade"), n_samples=400, seed=5
    )
    baseline, rows = elimination_gains(model, profile, config)
    gains = {row.factor: row.gain for row in rows}
    assert 1.0 < baseline < 3.0
    assert gains["strike"] == 0.0
    assert gains["face"] > 0.0 and gains["read_grade"] > 0.0
