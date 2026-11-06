"""SHOW-B5 (#14569): the lie's spin factor reaches the ball flight in a B3 table.

``perturb_impact_rows`` returns a per-shot spin factor (rough cuts spin, a flyer
cuts it more) that ``IMPACT_FIELDS`` has no column for, so the B3 table path
used to drop it.  The modifier now exposes ``with_spin``; ``build_grid_table``
hands the factor to the landing simulator, which scales the launch spin.  A
fairway or tee lie is the identity and leaves the simulator call unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.physics.lie_interaction import make_lie_modifier
from src.shared.python.physics.rust_kernel import is_rust_available
from src.shared.python.physics.swing_ball_flight_pipeline import (
    SwingBallFlightPipeline,
)
from src.shared.python.sg_optimizer.shot_model.dispersion_lie import (
    lie_modifier_for_table,
)
from src.shared.python.sg_optimizer.shot_model.dispersion_table import (
    TableGrid,
    build_grid_table,
)
from src.shared.python.sg_optimizer.shot_model.physics_simulator import (
    PipelineLandingSimulator,
    swing_state_from_impact,
)
from src.shared.python.sg_optimizer.shot_model.skill_family import IMPACT_FIELDS
from src.shared.python.sg_optimizer.shot_model.skill_registry import (
    get_skill_family,
)

pytestmark = pytest.mark.unit

_DRIVER = np.array([100.0, 0.0, 0.0, -2.0, 0.0, 0.0])


def _rows(n: int = 400) -> np.ndarray:
    rows = np.zeros((n, len(IMPACT_FIELDS)))
    rows[:, 0] = 100.0
    return rows


def test_pipeline_scales_the_launch_spin_and_keeps_the_identity_exact() -> None:
    if not is_rust_available():
        pytest.skip("upstream-physics Rust kernel not available")
    pipe, swing = SwingBallFlightPipeline(), swing_state_from_impact(_DRIVER, "driver")
    base = pipe.run(swing)
    same = pipe.run(swing, spin_factor=1.0)
    assert same.launch_conditions.spin_rate == base.launch_conditions.spin_rate
    assert same.carry_m == base.carry_m
    half = pipe.run(swing, spin_factor=0.5)
    assert half.launch_conditions.spin_rate == pytest.approx(
        0.5 * base.launch_conditions.spin_rate
    )
    assert half.carry_m != pytest.approx(base.carry_m)
    for bad in (-0.1, float("nan")):
        with pytest.raises(ContractViolationError):
            pipe.run(swing, spin_factor=bad)


def test_with_spin_returns_the_call_rows_and_rough_cuts_spin() -> None:
    rows = _rows()
    rough = make_lie_modifier(0.6, np.random.default_rng(3))
    out, spin = rough.with_spin(rows, "rough")
    again = make_lie_modifier(0.6, np.random.default_rng(3))(rows, "rough")
    np.testing.assert_array_equal(out, again)
    assert spin.shape == (len(rows),) and float(spin.mean()) < 1.0
    _, ones = make_lie_modifier(0.6, np.random.default_rng(3)).with_spin(
        rows, "fairway"
    )
    np.testing.assert_array_equal(ones, np.ones(len(rows)))


def test_table_modifier_with_spin_is_deterministic_and_matches_its_call() -> None:
    mod, rows = lie_modifier_for_table(7), _rows()
    out, spin = mod.with_spin(rows, "rough")
    np.testing.assert_array_equal(out, mod(rows, "rough"))
    np.testing.assert_array_equal(spin, mod.with_spin(rows, "rough")[1])


def test_landing_simulator_applies_the_per_shot_spin_factor() -> None:
    if not is_rust_available():
        pytest.skip("upstream-physics Rust kernel not available")
    sim, impacts = PipelineLandingSimulator(), np.stack([_DRIVER, _DRIVER])
    base = sim(impacts, "driver", (0.0, 0.0))
    np.testing.assert_array_equal(
        sim(impacts, "driver", (0.0, 0.0), spin_factor=np.ones(2)), base
    )
    cut = sim(impacts, "driver", (0.0, 0.0), spin_factor=np.array([1.0, 0.6]))
    np.testing.assert_array_equal(cut[0], base[0])
    assert not np.allclose(cut[1], base[1])
    with pytest.raises(ContractViolationError):
        sim(impacts, "driver", (0.0, 0.0), spin_factor=np.ones(3))


def test_table_build_hands_the_spin_factor_to_the_simulator() -> None:
    seen: dict[str, object] = {}

    def sim(impacts, skill, wind, spin_factor=None):  # noqa: ANN001, ANN202
        seen[skill] = spin_factor
        return np.column_stack([impacts[:, 0] * 2.0, impacts[:, 1]])

    family = get_skill_family("tour")
    skill = next(iter(family.skills))
    for lie, cut in (("fairway", False), ("rough", True)):
        build_grid_table(
            family,
            sim,
            TableGrid((skill,), (lie,), ("calm",)),
            n=200,
            seed=1,
            lie_modifier=lie_modifier_for_table(1),
        )
        factor = seen[skill]
        if cut:
            assert factor is not None and float(np.mean(factor)) < 1.0
        else:
            assert factor is None
