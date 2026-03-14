from unittest.mock import Mock

from src.api.routes.actuator_controls import (
    _demo_actuators,
    _get_actuator_info,
)


def test_demo_actuators():
    acts = _demo_actuators()
    assert len(acts) == 6
    assert acts[0].name == "hip_rotation"
    assert acts[0].min_value == -3.14


def test_get_actuator_info_no_engine():
    engine_manager = Mock()
    engine_manager.get_active_engine.return_value = None
    acts = _get_actuator_info(engine_manager)
    assert len(acts) == 6
    assert acts[0].name == "hip_rotation"


def test_get_actuator_info_with_engine():
    engine_manager = Mock()
    engine = Mock()
    engine.joint_names = ["arm", "leg"]
    engine.get_state.return_value = {"torques": [1.0, 2.0]}
    engine.get_joint_limits.return_value = [(-10, 10), (-20, 20)]
    engine_manager.get_active_engine.return_value = engine

    acts = _get_actuator_info(engine_manager)
    assert len(acts) == 2
    assert acts[0].name == "arm"
    assert acts[0].value == 1.0
    assert acts[0].min_value == -10
    assert acts[1].name == "leg"
    assert acts[1].value == 2.0
    assert acts[1].max_value == 20
