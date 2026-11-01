"""BatchRollModel (#14171): the batch integrator behind PuttOutcomeModel."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.putting_analytics.expected_putts import PuttAction
from src.shared.python.putting_analytics.roll_model import BatchRollModel
from src.shared.python.putting_dynamics.surfaces import HeightField, SurfaceSpec

pytestmark = pytest.mark.unit

FLAT = SurfaceSpec.flat_uniform(10.0)


def _noise(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    return rng.normal(0.0, 2.0, n), np.exp(rng.normal(0.0, 0.05, n))


def test_pace_decides_the_make_on_a_straight_putt() -> None:
    model = BatchRollModel(FLAT)
    assert model.simulate(6.0, PuttAction(0.0, 0.5), 2, 0).holed.all()
    assert not model.simulate(6.0, PuttAction(0.0, 12.0), 2, 0).holed.any()


def test_nominal_pace_rolls_the_planned_distance() -> None:
    model = BatchRollModel(FLAT, capture="point")
    miss = model.simulate(6.0, PuttAction(10.0, 2.0), 1, 0)
    assert miss.leave_ft[0] > 1.0
    assert model.nominal_speed_mps(6.0, 2.0) > model.nominal_speed_mps(6.0, 1.0)


def test_noisy_putts_are_reproducible_and_partly_holed() -> None:
    model = BatchRollModel(FLAT, execution=_noise)
    first = model.simulate(8.0, PuttAction(0.0, 1.5), 64, 11)
    again = model.simulate(8.0, PuttAction(0.0, 1.5), 64, 11)
    np.testing.assert_array_equal(first.leave_ft, again.leave_ft)
    assert 0.0 < first.make_rate < 1.0
    assert np.all(first.leave_ft[first.holed] == 0.0)


def test_contract_errors() -> None:
    steep = SurfaceSpec(
        HeightField.planar(8.0, 0.0), FLAT.friction_field, FLAT.friction
    )
    with pytest.raises(ValueError):
        BatchRollModel(steep).nominal_speed_mps(6.0, 1.0)
    with pytest.raises(ValueError):
        BatchRollModel(FLAT).simulate(0.0, PuttAction(0.0, 1.0), 4, 0)
