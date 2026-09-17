"""Real Pinocchio checks for the full-body polynomial RK4 boundary."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

from defusedxml import ElementTree as ET
import numpy as np
import pytest

from src.shared.python.motion_matching.contact_law import GroundPlane

_COLLECTED_WITH_PINOCCHIO_MOCK = isinstance(sys.modules.get("pinocchio"), Mock)
_CACHED_PLANT_STATE = None

if not _COLLECTED_WITH_PINOCCHIO_MOCK:
    from src.shared.python.motion_matching.full_body_step import (
        FullBodyStepOptions,
        NativeFullBodyStep,
    )
    from src.shared.python.motion_matching.polynomial_actuation import (
        ROOT_COORDINATES,
        FullBodyPolynomialControl,
    )

pytestmark = [pytest.mark.integration, pytest.mark.requires_pinocchio]


@pytest.fixture(scope="module", autouse=True)
def native_import_isolation(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Execute the native module in a clean process when unit mocks were collected."""
    if not _COLLECTED_WITH_PINOCCHIO_MOCK:
        return
    probe = subprocess.run(  # noqa: S603
        [sys.executable, "-c", "import pinocchio"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if probe.returncode != 0:
        pytest.skip(f"native Pinocchio unavailable: {probe.stderr}")
    report = tmp_path_factory.mktemp("native-step-report") / "junit.xml"
    command = [
        sys.executable,
        "-m",
        "pytest",
        str(Path(__file__).resolve()),
        "-q",
        "--timeout=60",
        f"--junitxml={report}",
    ]
    try:
        subprocess.run(  # noqa: S603
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.CalledProcessError as exc:
        pytest.fail(
            f"isolated native polynomial-step tests failed:\n{exc.stdout}\n{exc.stderr}"
        )
    suites = list(ET.parse(report).getroot().iter("testsuite"))
    counts = {
        field: sum(int(suite.attrib.get(field, 0)) for suite in suites)
        for field in ("tests", "failures", "errors", "skipped")
    }
    if counts != {"tests": 4, "failures": 0, "errors": 0, "skipped": 0}:
        pytest.fail(f"isolated native polynomial-step counts are invalid: {counts}")


@pytest.fixture
def native_plant_state(request: pytest.FixtureRequest):
    global _CACHED_PLANT_STATE
    if _COLLECTED_WITH_PINOCCHIO_MOCK:
        return None
    if _CACHED_PLANT_STATE is None:
        _CACHED_PLANT_STATE = request.getfixturevalue("plant_state")
    return _CACHED_PLANT_STATE


def _active_plane(plant, pose: dict[str, float]) -> None:
    zero = dict.fromkeys(pose, 0.0)
    samples = plant.contact_forces(pose, zero)
    lowest = min(float(sample.contact_point_m[2]) for sample in samples.values())
    plant.ground_plane = GroundPlane((0.0, 0.0, 1.0), lowest + 0.007)


def _check_real_step(
    plant,
    source_pose: dict[str, float],
    active_contact: bool,
    reversed_names: bool,
) -> None:
    pose = {**source_pose, "LWInputX": source_pose["LWInputX"] + 0.1}
    if active_contact:
        _active_plane(plant, pose)
    else:
        plant.ground_plane = GroundPlane((0.0, 0.0, 1.0), -10.0)
    names = tuple(plant.coordinate_order)
    if reversed_names:
        names = tuple(reversed(names))
        actuated = tuple(name for name in names if name not in ROOT_COORDINATES)
        control = FullBodyPolynomialControl(names, actuated, 1.0)
    else:
        control = FullBodyPolynomialControl.from_spec(
            plant.specification, duration_s=1.0
        )
    stepper = NativeFullBodyStep(plant, control, FullBodyStepOptions(2))
    rates = dict.fromkeys(names, 0.0)
    rates["TranslationInputX"] = 0.012
    rates["TranslationInputZ"] = -0.02
    rates["HipInputZ"] = 0.23
    rates["knee_angle_r"] = 0.31
    rates["ankle_angle_l"] = -0.18
    state = np.r_[
        [pose[name] for name in names],
        [rates[name] for name in names],
    ]
    parameters = np.zeros(control.n_parameters)
    parameters[::7] = 0.01
    dt_s = 1e-4
    result = stepper.linearize(state, parameters, time_s=0.2, dt_s=dt_s)
    value_only = stepper.step(state, parameters, time_s=0.2, dt_s=dt_s)
    np.testing.assert_array_equal(result.next_state, value_only)

    rng = np.random.default_rng(10265 + int(active_contact) + int(reversed_names))
    state_direction = rng.normal(size=state.size)
    state_direction /= np.linalg.norm(state_direction)
    epsilon = 2e-6
    plus = stepper.step(
        state + epsilon * state_direction, parameters, time_s=0.2, dt_s=dt_s
    )
    minus = stepper.step(
        state - epsilon * state_direction, parameters, time_s=0.2, dt_s=dt_s
    )
    measured = (plus - minus) / (2.0 * epsilon)
    np.testing.assert_allclose(
        result.dnext_dstate @ state_direction, measured, rtol=3e-4, atol=3e-6
    )

    coefficient_direction = np.zeros(control.n_parameters)
    actuator = control.actuated_names.index("knee_angle_r")
    coefficient_direction[7 * actuator : 7 * (actuator + 1)] = 1.0 / np.sqrt(7.0)
    epsilon = 1e-4
    plus = stepper.step(
        state, parameters + epsilon * coefficient_direction, time_s=0.2, dt_s=dt_s
    )
    minus = stepper.step(
        state, parameters - epsilon * coefficient_direction, time_s=0.2, dt_s=dt_s
    )
    measured = (plus - minus) / (2.0 * epsilon)
    assert np.linalg.norm(measured) > 1e-5
    np.testing.assert_allclose(
        result.dnext_dcoefficients @ coefficient_direction,
        measured,
        rtol=4e-4,
        atol=3e-7,
    )

    efforts = control.efforts(parameters, 0.2)
    assert all(efforts[name] == 0.0 for name in ROOT_COORDINATES)
    assert result.next_state.shape == state.shape
    assert np.isfinite(result.next_state).all()
    assert result.differentiable


@pytest.mark.parametrize("active_contact", [False, True])
@pytest.mark.parametrize("reversed_names", [False, True])
def test_real_step_linearization_matches_directional_rollouts(
    native_plant_state, active_contact: bool, reversed_names: bool
) -> None:
    if native_plant_state is None:
        return
    plant, pose = native_plant_state
    _check_real_step(plant, pose, active_contact, reversed_names)
