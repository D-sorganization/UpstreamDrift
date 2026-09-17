"""Real Pinocchio 3.8 and Pink 4.4 integration tests for FullBodyPinkTasks (#10276).

Follows strict TDD and DbC standards:
- Directional finite difference checks for marker FrameTask Jacobians
- Directional finite difference checks for weld RelativeFrameTask Jacobians and
  direct equivalence to FullBodyPinocchioModel.closure_position_linearization
- Hard equality constraint enforcement for locked coordinates
- Real Pink QP solve and post-integration manifold audit (closure and limits)
"""

from __future__ import annotations

import importlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from defusedxml import ElementTree as ET

_COLLECTED_WITH_PINOCCHIO_MOCK = isinstance(sys.modules.get("pinocchio"), Mock)

if not _COLLECTED_WITH_PINOCCHIO_MOCK:
    try:
        pin = importlib.import_module("pinocchio")
        pink = importlib.import_module("pink")
        importlib.import_module("quadprog")
        from src.engines.physics_engines.pinocchio.python.native_model import (
            FullBodyPinocchioModel,
        )
        from src.engines.physics_engines.pinocchio.python.pink_tasks import (
            ConfigurationState,
            FrameResiduals,
            FrameTaskBundle,
            FrameTaskOptions,
            FrameTaskRequest,
            FullBodyPinkTasks,
            StanceClosurePolicy,
        )
    except (ImportError, OSError) as exc:
        pytest.skip(
            f"native Pinocchio/Pink stack unavailable: {exc}",
            allow_module_level=True,
        )

pytestmark = [pytest.mark.integration, pytest.mark.requires_pinocchio]


@pytest.fixture(scope="module", autouse=True)
def native_import_isolation(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Run native checks outside the unit conftest's process-wide mock."""
    if not _COLLECTED_WITH_PINOCCHIO_MOCK:
        return
    dependencies = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.util; "
            "missing = [n for n in ('pinocchio', 'pink', 'quadprog') "
            "if importlib.util.find_spec(n) is None]; "
            "print(','.join(missing)); raise SystemExit(2 if missing else 0)",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if dependencies.returncode == 2:
        pytest.skip(
            f"Optional native dependencies absent: {dependencies.stdout.strip()}"
        )
    assert dependencies.returncode == 0, dependencies.stderr
    report = tmp_path_factory.mktemp("native-pink-tasks") / "results.xml"
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
            f"isolated native Pink tasks tests failed:\n{exc.stdout}\n{exc.stderr}"
        )
    cases = ET.parse(report).findall(".//testcase")
    assert len(cases) == 4, "All four native task integration tests must execute"
    assert all(
        case.find("skipped") is None
        and case.find("failure") is None
        and case.find("error") is None
        for case in cases
    ), "Skipped or failed native contracts do not qualify the task module"


@pytest.fixture
def real_plant_and_spec() -> tuple[FullBodyPinocchioModel, dict[str, Any], np.ndarray]:
    """Load canonical full-body model specification and initial valid pose."""
    repo_root = Path(__file__).resolve().parents[4]
    models_dir = repo_root / "docs" / "development" / "full_body_models"
    spec_path = models_dir / "full_body_spec_v1.json"
    candidate_path = (
        models_dir / "evidence" / "native_candidates" / "returned81_candidate.json"
    )

    if not spec_path.exists() or not candidate_path.exists():
        pytest.skip("Full-body model specification or candidate fixture missing")

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))

    plant = FullBodyPinocchioModel(spec)
    q_dict = dict.fromkeys(spec["coordinate_order"], 0.0)
    q_dict.update(zip(candidate["coordinate_names"], candidate["q0"], strict=True))
    q_dict["knee_angle_r"] = 0.15
    q_dict["knee_angle_l"] = 0.18
    q0 = plant.configuration(q_dict)
    return plant, spec, q0


def test_pink_marker_task_jacobian_matches_directional_differences(
    real_plant_and_spec: tuple[FullBodyPinocchioModel, dict[str, Any], np.ndarray],
) -> None:
    """FrameTask position Jacobian must match numerical directional finite differences."""
    plant, spec, q0 = real_plant_and_spec
    facade = FullBodyPinkTasks(spec, plant)

    # Pick an active marker
    marker_name = "HeadTop"
    pin.forwardKinematics(plant.model, plant.data, q0)
    pin.updateFramePlacements(plant.model, plant.data)
    fid = plant.model.getFrameId(marker_name)
    current_pos = plant.data.oMf[fid].translation.copy()
    target_pos = current_pos + np.array([0.05, -0.03, 0.02])

    request = FrameTaskRequest(
        marker_targets={marker_name: target_pos},
        validity_mask={marker_name: True},
        policy=StanceClosurePolicy(enforce_weld=False),
    )
    bundle = facade.build(request)
    assert len(bundle.tasks) >= 1
    task = bundle.tasks[0]

    config = pink.Configuration(plant.model, plant.model.createData(), q0)
    analytical_J = task.compute_jacobian(config)

    rng = np.random.default_rng(10276)
    eps = 1e-6
    for _ in range(5):
        direction = rng.normal(size=plant.model.nv)
        direction /= np.linalg.norm(direction)

        q_plus = pin.integrate(plant.model, q0, direction * eps)
        q_minus = pin.integrate(plant.model, q0, -direction * eps)

        conf_plus = pink.Configuration(plant.model, plant.model.createData(), q_plus)
        conf_minus = pink.Configuration(plant.model, plant.model.createData(), q_minus)

        err_plus = task.compute_error(conf_plus)
        err_minus = task.compute_error(conf_minus)

        numerical_deriv = (err_plus - err_minus) / (2.0 * eps)
        expected_deriv = analytical_J @ direction

        np.testing.assert_allclose(
            expected_deriv, numerical_deriv, rtol=1e-5, atol=1e-6
        )


def test_pink_weld_task_jacobian_matches_directional_differences_and_oracle(
    real_plant_and_spec: tuple[FullBodyPinocchioModel, dict[str, Any], np.ndarray],
) -> None:
    """RelativeFrameTask weld Jacobian must match finite differences and closure_position_linearization."""
    plant, spec, q0 = real_plant_and_spec
    q_dict = {name: float(q0[plant._coordinates[name]]) for name in plant._coordinates}
    lin = plant.closure_position_linearization(q_dict)

    facade = FullBodyPinkTasks(spec, plant)

    request = FrameTaskRequest(
        marker_targets={"HeadTop": np.zeros(3)},
        validity_mask={"HeadTop": True},
        policy=StanceClosurePolicy(enforce_weld=True),
    )
    bundle = facade.build(request)
    weld_tasks = [c for c in bundle.constraints if getattr(c, "is_weld_closure", False)]
    assert len(weld_tasks) == 1
    weld_task = weld_tasks[0]

    config = pink.Configuration(plant.model, plant.model.createData(), q0)
    analytical_J = weld_task.compute_jacobian(config)

    # 1. Compare against numerical directional differences
    rng = np.random.default_rng(10276)
    eps = 1e-6
    for _ in range(5):
        direction = rng.normal(size=plant.model.nv)
        direction /= np.linalg.norm(direction)

        q_plus = pin.integrate(plant.model, q0, direction * eps)
        q_minus = pin.integrate(plant.model, q0, -direction * eps)

        conf_plus = pink.Configuration(plant.model, plant.model.createData(), q_plus)
        conf_minus = pink.Configuration(plant.model, plant.model.createData(), q_minus)

        err_plus = weld_task.compute_error(conf_plus)
        err_minus = weld_task.compute_error(conf_minus)

        numerical_deriv = (err_plus - err_minus) / (2.0 * eps)
        expected_deriv = analytical_J @ direction

        np.testing.assert_allclose(
            expected_deriv, numerical_deriv, rtol=1e-5, atol=1e-6
        )

    # 2. Compare against plant.closure_position_linearization
    columns = [plant._velocity_indices[name] for name in lin.names]
    weld_J_oracle = lin.jacobian

    np.testing.assert_allclose(
        analytical_J[:, columns], weld_J_oracle, rtol=1e-7, atol=1e-8
    )


def test_pink_locked_coordinate_constraint(
    real_plant_and_spec: tuple[FullBodyPinocchioModel, dict[str, Any], np.ndarray],
) -> None:
    """Locked coordinate constraint must compute exact single-coordinate error and row-selector Jacobian."""
    plant, spec, q0 = real_plant_and_spec
    facade = FullBodyPinkTasks(spec, plant)

    coord_to_lock = "knee_angle_r"
    target_angle = 0.35
    request = FrameTaskRequest(
        marker_targets={"HeadTop": np.zeros(3)},
        validity_mask={"HeadTop": True},
        policy=StanceClosurePolicy(
            enforce_weld=False,
            locked_coordinates={coord_to_lock: target_angle},
        ),
    )
    bundle = facade.build(request)
    locked_tasks = [
        c
        for c in bundle.constraints
        if getattr(c, "coordinate_name", None) == coord_to_lock
    ]
    assert len(locked_tasks) == 1
    locked_task = locked_tasks[0]

    config = pink.Configuration(plant.model, plant.model.createData(), q0)
    err = locked_task.compute_error(config)
    jac = locked_task.compute_jacobian(config)

    q_idx = facade._coordinates[coord_to_lock]
    v_idx = facade._velocity_indices[coord_to_lock]

    expected_err = q0[q_idx] - target_angle
    np.testing.assert_allclose(err[0], expected_err)

    expected_jac = np.zeros((1, plant.model.nv))
    expected_jac[0, v_idx] = 1.0
    np.testing.assert_allclose(jac, expected_jac)


def test_pink_qp_solve_and_post_integration_audit(
    real_plant_and_spec: tuple[FullBodyPinocchioModel, dict[str, Any], np.ndarray],
) -> None:
    """Pink solve_ik must converge satisfying marker tracking, 6D weld closure, locked joints, and limits."""
    plant, spec, q0 = real_plant_and_spec
    facade = FullBodyPinkTasks(spec, plant)

    marker_name = "HeadTop"
    pin.forwardKinematics(plant.model, plant.data, q0)
    pin.updateFramePlacements(plant.model, plant.data)
    fid = plant.model.getFrameId(marker_name)
    current_pos = plant.data.oMf[fid].translation.copy()
    target_pos = current_pos + np.array([0.01, -0.01, 0.005])

    locked_coord = "knee_angle_r"
    locked_target = 0.25

    request = FrameTaskRequest(
        marker_targets={marker_name: target_pos},
        validity_mask={marker_name: True},
        policy=StanceClosurePolicy(
            enforce_weld=True,
            locked_coordinates={locked_coord: locked_target},
        ),
        options=FrameTaskOptions(solver="quadprog", dt=1.0 / 360.0),
    )
    bundle = facade.build(request)

    dt = 1.0 / 360.0
    conf = pink.Configuration(plant.model, plant.model.createData(), q0)
    velocity = pink.solve_ik(
        conf,
        bundle.tasks,
        dt=dt,
        constraints=bundle.constraints,
        limits=bundle.limits,
        solver="quadprog",
    )
    assert np.all(np.isfinite(velocity))

    q_next = pin.integrate(plant.model, q0, velocity * dt)
    assert np.all(np.isfinite(q_next))

    conf_state = ConfigurationState(q=q_next)
    residuals = facade.audit(conf_state, request)

    # 1. Marker error must be reduced substantially (target was 1.5 cm away, residual < 0.2 mm)
    assert marker_name in residuals.marker_errors_m
    assert residuals.marker_errors_m[marker_name] < 2e-4

    # 2. Weld closure error must be machine precision (< 1e-6 m and < 1e-6 rad)
    assert residuals.weld_translation_error_m < 1e-6
    assert residuals.weld_rotation_error_rad < 1e-6

    # 3. Locked coordinate must match target exactly
    locked_idx = facade._coordinates[locked_coord]
    np.testing.assert_allclose(q_next[locked_idx], locked_target, atol=1e-7)

    # 4. Joint bounds must have zero violations
    assert len(residuals.bound_violations) == 0
