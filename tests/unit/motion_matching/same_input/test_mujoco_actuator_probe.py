"""Synthetic motor seam tests; limits in these fixtures are not physiological."""

from pathlib import Path
import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.full_body_model import (
    NativeMujocoFullBodyModel,
)
from src.shared.python.motion_matching.same_input.mujoco_actuator_probe import (
    ControlledMujocoDrive,
)

pytestmark = pytest.mark.unit

SPEC = Path("docs/development/full_body_models/full_body_spec_anthro_driver_v2.json")


@pytest.fixture(scope="module")
def model():
    return NativeMujocoFullBodyModel(SPEC.read_bytes())


def test_missing_and_stateful_actuators_fail_closed(model):
    with pytest.raises(ValueError, match="actuator"):
        ControlledMujocoDrive(model, "<actuator/>")
    name = model.coordinate_order[6]
    with pytest.raises(ValueError, match="stateful"):
        ControlledMujocoDrive(
            model,
            f'<actuator><general joint="{name}" dyntype="filter" dynprm="1"/></actuator>',
        )


def test_actual_motor_total_drives_existing_kkt_once(model):
    names = model.coordinate_order
    xml = "".join(
        f'<motor joint="{n}" gear="1" ctrllimited="true" ctrlrange="-1 1"/>'
        for n in names[6:]
    )
    system = ControlledMujocoDrive(model, f"<actuator>{xml}</actuator>")
    q = np.zeros(len(names))
    v = np.zeros(len(names))
    command = np.resize(np.array([2.0, -2.0, 0.25]), len(names) - 6)
    acceleration, applied = system.acceleration(q, v, command)
    expected = np.r_[np.zeros(6), np.clip(command, -1, 1)]
    assert np.array_equal(applied, expected)
    plain = model.accelerations(
        dict(zip(names, q, strict=True)),
        dict(zip(names, v, strict=True)),
        dict(zip(names, expected, strict=True)),
    )
    assert np.allclose(acceleration, [plain[n] for n in names], atol=1e-12, rtol=1e-12)
    assert not np.array_equal(applied[6:], command)


def test_control_and_input_validation(model):
    name = model.coordinate_order[6]
    system = ControlledMujocoDrive(
        model, f'<actuator><motor joint="{name}"/></actuator>'
    )
    q = np.zeros(len(model.coordinate_order))
    for ctrl in (np.array([np.nan]), np.array([1.0, 2.0])):
        with pytest.raises(ValueError):
            system.acceleration(q, q, ctrl)
    bad = q.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError):
        system.acceleration(bad, q, np.array([1.0]))


def test_force_limit_and_transmission_are_measured_after_saturation(model):
    name = model.coordinate_order[6]
    system = ControlledMujocoDrive(
        model,
        f'<actuator><motor joint="{name}" gear="2" forcelimited="true" forcerange="-0.5 0.5"/></actuator>',
    )
    q = np.zeros(len(model.coordinate_order))
    _, applied = system.acceleration(q, q, np.array([3.0]))
    assert applied[6] == pytest.approx(1.0)
    assert np.count_nonzero(applied) == 1


def test_repeated_calls_do_not_accumulate_actuation(model):
    name = model.coordinate_order[6]
    system = ControlledMujocoDrive(
        model, f'<actuator><motor joint="{name}"/></actuator>'
    )
    q = np.zeros(len(model.coordinate_order))
    a1, t1 = system.acceleration(q, q, np.array([1.0]))
    system.acceleration(q, q, np.array([-0.5]))
    a2, t2 = system.acceleration(q, q, np.array([1.0]))
    assert np.array_equal(t1, t2)
    assert np.array_equal(a1, a2)
