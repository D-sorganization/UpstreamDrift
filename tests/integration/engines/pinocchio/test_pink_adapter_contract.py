"""Real Pinocchio 3.8 and Pink 4.4 contract tests for the IK adapters."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from defusedxml import ElementTree as ET

_COLLECTED_WITH_PINOCCHIO_MOCK = isinstance(sys.modules.get("pinocchio"), Mock)

if not _COLLECTED_WITH_PINOCCHIO_MOCK:
    try:
        pin = importlib.import_module("pinocchio")
        importlib.import_module("pink")
        NoSolutionFound = importlib.import_module("pink.exceptions").NoSolutionFound
        VelocityLimit = importlib.import_module("pink.limits").VelocityLimit
        PostureTask = importlib.import_module("pink.tasks").PostureTask
        PINKBackend = importlib.import_module(
            "src.engines.physics_engines.pinocchio.python.dtack.backends.pink_backend"
        ).PINKBackend
        PinkSolver = importlib.import_module(
            "src.engines.physics_engines.pinocchio.python.dtack.ik.pink_solver"
        ).PinkSolver
    except (ImportError, OSError) as exc:
        pytest.skip(
            f"native Pinocchio/Pink stack unavailable: {exc}", allow_module_level=True
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
    report = tmp_path_factory.mktemp("native-pink") / "results.xml"
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
        pytest.fail(f"isolated native Pink tests failed:\n{exc.stdout}\n{exc.stderr}")
    # This XML is produced exclusively by the just-completed local subprocess.
    cases = ET.parse(report).findall(".//testcase")
    assert len(cases) == 4, "All four native adapter contracts must execute"
    assert all(
        case.find("skipped") is None
        and case.find("failure") is None
        and case.find("error") is None
        for case in cases
    ), "Skipped or failed native contracts do not qualify the adapter"


@pytest.fixture
def revolute_urdf(tmp_path: Path) -> Path:
    path = tmp_path / "one_joint.urdf"
    path.write_text(
        """<?xml version="1.0"?>
<robot name="one_joint">
  <link name="base"/>
  <link name="arm"/>
  <link name="contact">
    <collision>
      <geometry><box size="0.1 0.1 0.1"/></geometry>
    </collision>
  </link>
  <joint name="shoulder" type="revolute">
    <parent link="base"/>
    <child link="arm"/>
    <origin xyz="0.4 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.5" upper="1.5" effort="20" velocity="10"/>
  </joint>
  <joint name="contact_mount" type="fixed">
    <parent link="arm"/>
    <child link="contact"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )
    return path


def _posture(target: float) -> PostureTask:
    task = PostureTask(cost=1.0)
    task.set_target(np.array([target]))
    return task


def test_backend_hard_posture_reaches_target_and_refreshes_frames(
    revolute_urdf: Path,
) -> None:
    if _COLLECTED_WITH_PINOCCHIO_MOCK:
        return
    backend = PINKBackend(revolute_urdf)
    target = 0.25

    assert len(backend.robot.collision_model.geometryObjects) == 1
    assert backend.configuration.collision_model is backend.robot.collision_model

    result = backend.solve_ik(
        {}, np.array([-0.3]), dt=0.1, constraints=[_posture(target)], limits=[]
    )

    np.testing.assert_allclose(result, [target], atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(backend.configuration.q, result, atol=0.0, rtol=0.0)
    fresh_data = backend.robot.model.createData()
    pin.forwardKinematics(backend.robot.model, fresh_data, result)
    pin.updateFramePlacements(backend.robot.model, fresh_data)
    frame_id = backend.robot.model.getFrameId("arm")
    np.testing.assert_allclose(
        backend.configuration.data.oMf[frame_id].homogeneous,
        fresh_data.oMf[frame_id].homogeneous,
        atol=1e-12,
        rtol=0.0,
    )


def test_hard_posture_conflicting_with_velocity_limit_reports_infeasibility(
    revolute_urdf: Path,
) -> None:
    if _COLLECTED_WITH_PINOCCHIO_MOCK:
        return
    backend = PINKBackend(revolute_urdf)
    velocity_limit = VelocityLimit(backend.robot.model, np.array([0.1]))

    with pytest.raises(NoSolutionFound) as caught:
        backend.solve_ik(
            {},
            np.array([0.0]),
            dt=0.1,
            constraints=[_posture(0.25)],
            limits=[velocity_limit],
        )

    assert any("Pink IK solve failed" in note for note in caught.value.__notes__)
    np.testing.assert_allclose(backend.configuration.q, [0.0], atol=0.0, rtol=0.0)

    recovered = backend.solve_ik(
        {}, np.array([0.8]), dt=0.1, constraints=[_posture(-0.2)], limits=[]
    )

    np.testing.assert_allclose(recovered, [-0.2], atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(backend.configuration.q, recovered, atol=0.0, rtol=0.0)


def test_sequential_distant_starts_match_fresh_backend_solves(
    revolute_urdf: Path,
) -> None:
    if _COLLECTED_WITH_PINOCCHIO_MOCK:
        return
    reused = PINKBackend(revolute_urdf)
    target = -0.15
    for start in (np.array([-0.8]), np.array([0.9])):
        reused_result = reused.solve_ik(
            {}, start, dt=0.1, constraints=[_posture(target)], limits=[]
        )
        fresh = PINKBackend(revolute_urdf)
        fresh_result = fresh.solve_ik(
            {}, start, dt=0.1, constraints=[_posture(target)], limits=[]
        )
        np.testing.assert_allclose(reused_result, fresh_result, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(reused.configuration.q, reused_result)


def test_free_flyer_configuration_uses_nq_while_velocity_uses_nv() -> None:
    if _COLLECTED_WITH_PINOCCHIO_MOCK:
        return
    model = pin.Model()
    joint_id = model.addJoint(
        0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), "floating_base"
    )
    model.appendBodyToJoint(joint_id, pin.Inertia.Random(), pin.SE3.Identity())
    q_init = pin.neutral(model)
    visual_model = pin.GeometryModel()
    collision_model = pin.GeometryModel()
    solver = PinkSolver(model, model.createData(), visual_model, collision_model)

    result = solver.solve(q_init, [], 0.1, limits=[])

    assert model.nq == 7
    assert model.nv == 6
    assert result.shape == (model.nq,)
    np.testing.assert_allclose(result, q_init, atol=1e-12, rtol=0.0)
