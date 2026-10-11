"""Native MuJoCo actuator classification tests for MyoSuite analysis."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

mujoco = pytest.importorskip("mujoco")


_MIXED_ACTUATOR_XML = """<mujoco model="mixed_actuators">
  <worldbody>
    <body name="body">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.3" size="0.04" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <muscle name="muscle" joint="hinge" lengthrange="0.1 1.0"/>
    <general name="filter" joint="hinge" dyntype="filter" dynprm="0.02"
             gaintype="fixed" gainprm="1" biastype="none"/>
    <motor name="motor" joint="hinge"/>
  </actuator>
</mujoco>"""


@pytest.mark.unit
def test_identifies_only_native_muscle_actuators() -> None:
    """Use the installed MuJoCo enum on a model without MyoSuite/Gym."""
    from src.engines.physics_engines.myosuite.python.muscle_analysis import (
        MyoSuiteMuscleAnalyzer,
    )

    model = mujoco.MjModel.from_xml_string(_MIXED_ACTUATOR_XML)
    data = mujoco.MjData(model)

    assert model.actuator_dyntype.tolist() == [
        mujoco.mjtDyn.mjDYN_MUSCLE,
        mujoco.mjtDyn.mjDYN_FILTER,
        mujoco.mjtDyn.mjDYN_NONE,
    ]

    analyzer = MyoSuiteMuscleAnalyzer(SimpleNamespace(model=model, data=data))

    assert analyzer.muscle_actuator_ids == [0]
    assert analyzer.muscle_names == ["muscle"]
