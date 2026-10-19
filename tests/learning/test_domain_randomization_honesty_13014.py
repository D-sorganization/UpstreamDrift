"""Regression tests for issue #13014.

``DomainRandomizer`` must not silently skip physics parameters when the engine
lacks working getter/setter pairs. MuJoCo must expose a real randomization
surface so masses change and restore.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.learning.sim2real.domain_randomization import (
    DomainRandomizationConfig,
    DomainRandomizer,
)
from src.learning.sim2real.system_identification import UnsupportedParameterError

pytestmark = pytest.mark.unit

SYNTHETIC_HANGING_PENDULUM = """
<mujoco model="synthetic_hanging_pendulum">
  <option gravity="0 0 -9.81" timestep="0.001"/>
  <worldbody>
    <body name="link1" pos="0 0 0">
      <joint name="hinge1" type="hinge" axis="0 1 0" pos="0 0 0"/>
      <geom name="geom1" type="capsule" fromto="0 0 0 0 0 -1" size="0.05" mass="2.0"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor1" joint="hinge1" gear="1" ctrllimited="true" ctrlrange="-100 100"/>
  </actuator>
</mujoco>
"""


class BareEngine:
    """Physics engine with no domain-randomization parameter hooks."""


class TestDomainRandomizationFailsLoudly:
    def test_bare_engine_raises_when_physics_randomization_requested(self) -> None:
        config = DomainRandomizationConfig(
            randomize_delays=False,
            randomize_noise=False,
        )
        randomizer = DomainRandomizer(BareEngine(), config)
        with pytest.raises(UnsupportedParameterError, match="mass"):
            randomizer.randomize(seed=1)

    def test_bare_engine_allows_delay_only_randomization(self) -> None:
        config = DomainRandomizationConfig(
            randomize_mass=False,
            randomize_friction=False,
            randomize_damping=False,
            randomize_motor=False,
            randomize_gravity=False,
            randomize_delays=True,
            randomize_noise=False,
        )
        randomizer = DomainRandomizer(BareEngine(), config)
        result = randomizer.randomize(seed=1)
        assert "action_delay" in result
        assert "mass_scale" not in result

    def test_constructor_seed_is_reproducible_without_randomize_seed(self) -> None:
        from tests.learning.wave6_learning.test_domain_randomization import FakeEngine

        a = DomainRandomizer(FakeEngine(), seed=99)
        b = DomainRandomizer(FakeEngine(), seed=99)
        assert a.randomize()["mass_scale"] == b.randomize()["mass_scale"]


@pytest.mark.requires_mujoco
def test_mujoco_masses_change_and_restore() -> None:
    from src.shared.python.simulation_backends import has_mujoco

    if not has_mujoco():
        pytest.skip("mujoco not installed")

    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
        MuJoCoPhysicsEngine,
    )

    engine = MuJoCoPhysicsEngine()
    engine.load_from_string(SYNTHETIC_HANGING_PENDULUM, "xml")
    config = DomainRandomizationConfig(
        randomize_friction=False,
        randomize_damping=False,
        randomize_motor=False,
        randomize_gravity=False,
        randomize_delays=False,
        randomize_noise=False,
    )
    randomizer = DomainRandomizer(engine, config, seed=7)
    before = engine.get_link_masses().copy()
    randomizer.randomize()
    after = engine.get_link_masses()
    assert not np.allclose(before, after)
    randomizer.reset_to_nominal()
    np.testing.assert_allclose(engine.get_link_masses(), before)
