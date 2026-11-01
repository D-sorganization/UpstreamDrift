"""Selectable putting providers: logistic default and simulated curve (#14177)."""

from __future__ import annotations

import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.sg_optimizer.course.conditions import GreenModel
from src.shared.python.sg_optimizer.shot_model.player_profile import PuttingSkill
from src.shared.python.sg_optimizer.shot_model.putting import (
    leave_distance_distribution,
    make_probability,
)
from src.shared.python.sg_optimizer.shot_model.putting_provider import (
    CurvePuttingProvider,
    LogisticPuttingProvider,
    PuttingProvider,
    SimulatedPuttingCurve,
    putting_provider,
)

pytestmark = pytest.mark.unit

# Shape of a low-dispersion simulated player (simulate_putting_curve, n=500).
TOUR_CURVE = SimulatedPuttingCurve(
    distances_ft=(2.0, 3.0, 6.0, 10.0, 20.0, 30.0, 60.0),
    make_rate=(1.0, 1.0, 0.95, 0.73, 0.32, 0.18, 0.06),
    leave_scale_ft=(0.96, 0.96, 0.96, 1.13, 1.69, 2.41, 4.53),
    leave_sigma_log=(0.97, 0.97, 0.97, 0.89, 0.89, 0.87, 0.86),
    samples_per_distance=500,
)
PROVIDERS: dict[str, PuttingProvider] = {
    "logistic": LogisticPuttingProvider(),
    "simulation": CurvePuttingProvider(TOUR_CURVE),
}
MEDIUM = GreenModel.preset("medium")


@pytest.fixture(params=sorted(PROVIDERS))
def provider(request: pytest.FixtureRequest) -> PuttingProvider:
    return PROVIDERS[request.param]


def test_make_probability_falls_with_distance(provider: PuttingProvider) -> None:
    p3, p10, p30 = (
        provider.make_probability(d, PuttingSkill(), MEDIUM) for d in (3, 10, 30)
    )
    assert p3 > p10 > p30
    assert p3 > 0.85 and p30 < 0.20


def test_player_multiplier_raises_make(provider: PuttingProvider) -> None:
    skilled = PuttingSkill(make_pct_multipliers={3.0: 0.9, 25.0: 1.2})
    long_skilled = provider.make_probability(20.0, skilled, MEDIUM)
    assert long_skilled > provider.make_probability(20.0, PuttingSkill(), MEDIUM)


def test_fast_greens_cost_long_putts_more(provider: PuttingProvider) -> None:
    slow, fast = GreenModel.preset("slow"), GreenModel.preset("fast")

    def ratio(d: float) -> float:
        p_fast = provider.make_probability(d, PuttingSkill(), fast)
        return p_fast / provider.make_probability(d, PuttingSkill(), slow)

    assert ratio(20.0) < ratio(4.0) < 1.01


def test_leave_widens_with_stimp_and_three_putts_rise(
    provider: PuttingProvider,
) -> None:
    slow = provider.leave_distribution(30.0, PuttingSkill(), GreenModel.preset("slow"))
    fast = provider.leave_distribution(30.0, PuttingSkill(), GreenModel.preset("fast"))
    assert fast.sigma_log > slow.sigma_log
    assert fast.expected_three_putt_probability(4.0) > (
        slow.expected_three_putt_probability(4.0)
    )


def test_three_putt_avoidance_tightens_the_leave(provider: PuttingProvider) -> None:
    good = provider.leave_distribution(
        30.0, PuttingSkill(three_putt_avoidance=1.5), MEDIUM
    )
    base = provider.leave_distribution(30.0, PuttingSkill(), MEDIUM)
    assert good.mean_ft < base.mean_ft


def test_invalid_distance_rejected(provider: PuttingProvider) -> None:
    with pytest.raises(ContractViolationError):
        provider.make_probability(0.0, PuttingSkill(), MEDIUM)


def test_default_provider_is_the_logistic_model() -> None:
    default = putting_provider()
    assert isinstance(default, LogisticPuttingProvider)
    assert default.make_probability(8.0, PuttingSkill(), MEDIUM) == make_probability(
        8.0, PuttingSkill(), MEDIUM
    )
    assert default.leave_distribution(
        25.0, PuttingSkill(), MEDIUM
    ) == leave_distance_distribution(25.0, PuttingSkill(), MEDIUM)


def test_simulation_provider_reads_the_curve_at_neutral_conditions() -> None:
    sim = putting_provider("simulation", TOUR_CURVE)
    neutral = GreenModel(stimp=10.0)
    assert sim.make_probability(10.0, PuttingSkill(), neutral) == pytest.approx(0.73)
    leave = sim.leave_distribution(20.0, PuttingSkill(), neutral)
    assert (leave.mean_ft, leave.sigma_log) == pytest.approx((1.69, 0.89))


def test_curve_interpolates_in_log_distance_and_holds_its_ends() -> None:
    mid = (2.0 * 3.0) ** 0.5  # halfway in log distance
    assert TOUR_CURVE.make_rate_at(mid) == pytest.approx(1.0)
    assert TOUR_CURVE.make_rate_at(1.0) == 1.0
    assert TOUR_CURVE.make_rate_at(90.0) == pytest.approx(0.06)
    halfway = (10.0 * 20.0) ** 0.5
    assert TOUR_CURVE.make_rate_at(halfway) == pytest.approx((0.73 + 0.32) / 2)


def test_selector_rejects_bad_combinations() -> None:
    with pytest.raises(ValueError):
        putting_provider("simulation")
    with pytest.raises(ContractViolationError):
        putting_provider("logistic", TOUR_CURVE)
    with pytest.raises(ContractViolationError):
        putting_provider("lookup")


@pytest.mark.parametrize(
    "fields",
    [
        {"distances_ft": (2.0,), "make_rate": (1.0,)},
        {"distances_ft": (3.0, 2.0)},
        {"make_rate": (1.2, 0.5)},
        {"leave_scale_ft": (0.0, 1.0)},
        {"leave_sigma_log": (-0.1, 0.5)},
        {"make_rate": (1.0, 0.5, 0.2)},
    ],
)
def test_curve_rejects_invalid_tables(fields: dict[str, tuple[float, ...]]) -> None:
    table: dict[str, tuple[float, ...]] = {
        "distances_ft": (2.0, 3.0),
        "make_rate": (1.0, 0.9),
        "leave_scale_ft": (1.0, 1.1),
        "leave_sigma_log": (0.5, 0.5),
        **fields,
    }
    with pytest.raises(ContractViolationError):
        SimulatedPuttingCurve(
            table["distances_ft"],
            table["make_rate"],
            table["leave_scale_ft"],
            table["leave_sigma_log"],
        )
