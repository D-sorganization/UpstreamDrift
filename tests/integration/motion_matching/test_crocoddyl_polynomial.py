"""Real Crocoddyl and Pinocchio qualification for polynomial actions."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from defusedxml import ElementTree as ET
import numpy as np
import pytest

_NATIVE_MODULES_MOCKED = any(
    isinstance(sys.modules.get(name), Mock) for name in ("pinocchio", "crocoddyl")
)
_CACHED_PLANT_STATE = None

if not _NATIVE_MODULES_MOCKED:
    try:
        import crocoddyl  # noqa: F401

        from src.shared.python.motion_matching.contact_law import GroundPlane
        from src.shared.python.motion_matching.crocoddyl_polynomial import (
            PolynomialCostQuadratic,
            build_polynomial_shooting_problem,
        )
        from src.shared.python.motion_matching.full_body_step import (
            FullBodyStepOptions,
            NativeFullBodyStep,
        )
        from src.shared.python.motion_matching.polynomial_actuation import (
            ROOT_COORDINATES,
            FullBodyPolynomialControl,
        )
    except (ImportError, OSError) as exc:
        pytest.skip(
            f"native Crocoddyl/Pinocchio stack unavailable: {exc}",
            allow_module_level=True,
        )

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_crocoddyl,
    pytest.mark.requires_pinocchio,
]


@pytest.fixture(scope="module", autouse=True)
def native_import_isolation(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Run all native cases in a clean process after unit mock collection."""
    if not _NATIVE_MODULES_MOCKED:
        return
    probe = subprocess.run(  # noqa: S603
        [sys.executable, "-c", "import crocoddyl, pinocchio, defusedxml"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if probe.returncode != 0:
        pytest.skip(f"qualified native runtime unavailable: {probe.stderr}")
    report = tmp_path_factory.mktemp("native-crocoddyl-report") / "junit.xml"
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
            timeout=150,
        )
    except subprocess.CalledProcessError as exc:
        pytest.fail(f"isolated native action tests failed:\n{exc.stdout}\n{exc.stderr}")
    suites = list(ET.parse(report).getroot().iter("testsuite"))
    counts = {
        field: sum(int(suite.attrib.get(field, 0)) for suite in suites)
        for field in ("tests", "failures", "errors", "skipped")
    }
    if counts != {"tests": 6, "failures": 0, "errors": 0, "skipped": 0}:
        pytest.fail(f"isolated native Crocoddyl counts are invalid: {counts}")


@pytest.fixture
def native_plant_state(request: pytest.FixtureRequest):
    global _CACHED_PLANT_STATE
    if _NATIVE_MODULES_MOCKED:
        return None
    if _CACHED_PLANT_STATE is None:
        _CACHED_PLANT_STATE = request.getfixturevalue("plant_state")
    return _CACHED_PLANT_STATE


class _ZeroCost:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def value(self, state, parameters, time_s):
        del state, parameters, time_s
        return 0.0

    def quadraticize(self, state, parameters, time_s):
        del state, parameters, time_s
        return PolynomialCostQuadratic(
            0.0, np.zeros(self.dimension), np.zeros((self.dimension, self.dimension))
        )


class _TargetCost:
    def __init__(self, physical: int, parameters: int, target: np.ndarray) -> None:
        self.physical = physical
        self.parameters = parameters
        self.target = np.asarray(target, dtype=float)

    def value(self, state, parameters, time_s):
        del parameters, time_s
        error = state - self.target
        return 0.5 * float(error @ error)

    def quadraticize(self, state, parameters, time_s):
        value = self.value(state, parameters, time_s)
        gradient = np.r_[state - self.target, np.zeros(self.parameters)]
        hessian = np.zeros(
            (self.physical + self.parameters, self.physical + self.parameters)
        )
        hessian[: self.physical, : self.physical] = np.eye(self.physical)
        return PolynomialCostQuadratic(value, gradient, hessian)


class _ParticleStep:
    def __init__(self) -> None:
        self.control = SimpleNamespace(n_coordinates=1, n_parameters=2, duration_s=1.0)

    @staticmethod
    def _blocks(time_s: float, dt_s: float):
        state = np.array([[1.0, dt_s], [0.0, 1.0]])
        coefficients = np.array(
            [
                [0.5 * dt_s**2, 0.5 * time_s * dt_s**2 + dt_s**3 / 6.0],
                [dt_s, time_s * dt_s + 0.5 * dt_s**2],
            ]
        )
        return state, coefficients

    def step(self, state, parameters, *, time_s, dt_s):
        state_block, coefficient_block = self._blocks(time_s, dt_s)
        return state_block @ state + coefficient_block @ parameters

    def linearize(self, state, parameters, *, time_s, dt_s):
        state_block, coefficient_block = self._blocks(time_s, dt_s)
        return SimpleNamespace(
            next_state=self.step(state, parameters, time_s=time_s, dt_s=dt_s),
            dnext_dstate=state_block,
            dnext_dcoefficients=coefficient_block,
            differentiable=True,
        )


class _BoundedStep:
    def __init__(self) -> None:
        names = (*ROOT_COORDINATES, "joint")
        self.control = FullBodyPolynomialControl(names, ("joint",), 1.0)

    def step(self, state, parameters, *, time_s, dt_s):
        del time_s
        result = np.asarray(state, dtype=float).copy()
        result[:7] += dt_s * parameters
        return result

    def linearize(self, state, parameters, *, time_s, dt_s):
        return SimpleNamespace(
            next_state=self.step(state, parameters, time_s=time_s, dt_s=dt_s),
            dnext_dstate=np.eye(14),
            dnext_dcoefficients=np.vstack((dt_s * np.eye(7), np.zeros((7, 7)))),
            differentiable=True,
        )


def test_real_fddp_finds_unique_global_particle_coefficients() -> None:
    if _NATIVE_MODULES_MOCKED:
        return
    stepper = _ParticleStep()
    running = _ZeroCost(4)
    terminal = _TargetCost(2, 2, np.array([1.0, 0.0]))
    bundle = build_polynomial_shooting_problem(
        stepper, np.linspace(0.0, 1.0, 11), np.zeros(2), running, terminal
    )
    warm = bundle.warm_start(np.zeros(2))
    solver = crocoddyl.SolverFDDP(bundle.problem)

    solved = solver.solve(warm.xs, warm.us, 50, True)

    assert solved
    np.testing.assert_allclose(solver.us[0], [6.0, -12.0], rtol=3e-5, atol=3e-5)
    diagnostic = bundle.diagnose(solver.xs, solver.us)
    assert diagnostic.max_physical_defect < 1e-9
    assert diagnostic.max_coefficient_defect < 1e-9
    assert np.isfinite(np.linalg.eigvalsh(solver.Quu[0])).all()


def test_real_box_fddp_bounds_degree_six_coefficients_and_effort() -> None:
    if _NATIVE_MODULES_MOCKED:
        return
    stepper = _BoundedStep()
    nphysical = 14
    nparameters = 7
    target = np.r_[np.full(7, 2.0), np.zeros(7)]
    running = _ZeroCost(nphysical + nparameters)
    terminal = _TargetCost(nphysical, nparameters, target)
    lower = np.full(nparameters, -0.5)
    upper = np.full(nparameters, 0.5)
    bundle = build_polynomial_shooting_problem(
        stepper,
        np.array([0.0, 1.0]),
        np.zeros(nphysical),
        running,
        terminal,
        coefficient_bounds=(lower, upper),
    )
    warm = bundle.warm_start(np.zeros(nparameters))
    solver = crocoddyl.SolverBoxFDDP(bundle.problem)

    solved = solver.solve(warm.xs, warm.us, 50, True)

    assert solved
    coefficients = np.asarray(solver.us[0])
    assert np.all(coefficients >= lower - 1e-12)
    assert np.all(coefficients <= upper + 1e-12)
    assert np.max(coefficients) > 0.49
    for time_s in np.linspace(0.0, 1.0, 101):
        effort = stepper.control.efforts(coefficients, float(time_s))["joint"]
        assert -0.5 <= effort <= 0.5
    assert bundle.diagnose(solver.xs, solver.us).max_bound_violation == 0.0


def _active_plane(plant, pose: dict[str, float]) -> None:
    zero = dict.fromkeys(pose, 0.0)
    samples = plant.contact_forces(pose, zero)
    lowest = min(float(sample.contact_point_m[2]) for sample in samples.values())
    plant.ground_plane = GroundPlane((0.0, 0.0, 1.0), lowest + 0.007)


@pytest.mark.parametrize("active_contact", [False, True])
@pytest.mark.parametrize("reversed_names", [False, True])
def test_real_full_body_action_matches_native_step_linearization(
    native_plant_state, active_contact: bool, reversed_names: bool
) -> None:
    if native_plant_state is None:
        return
    plant, source_pose = native_plant_state
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
    rates["TranslationInputZ"] = -0.02
    rates["HipInputZ"] = 0.23
    rates["knee_angle_r"] = 0.31
    state = np.r_[
        [pose[name] for name in names],
        [rates[name] for name in names],
    ]
    parameters = np.zeros(control.n_parameters)
    dimension = state.size + parameters.size
    cost = _ZeroCost(dimension)
    bundle = build_polynomial_shooting_problem(
        stepper, np.array([0.2, 0.2001]), state, cost, cost
    )
    action = bundle.problem.runningModels[1]
    data = action.createData()
    augmented = np.r_[state, parameters]

    action.calc(data, augmented, np.zeros(0))
    action.calcDiff(data, augmented, np.zeros(0))

    direct = stepper.linearize(state, parameters, time_s=0.2, dt_s=1e-4)
    np.testing.assert_allclose(
        data.xnext, np.r_[direct.next_state, parameters], rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        data.Fx[: state.size, : state.size],
        direct.dnext_dstate,
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        data.Fx[: state.size, state.size :],
        direct.dnext_dcoefficients,
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        data.Fx[state.size :, state.size :], np.eye(parameters.size)
    )
    assert direct.differentiable
