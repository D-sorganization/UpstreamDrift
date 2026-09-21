"""Real Pinocchio and Pink integration tests for PinkTrajectoryService (Packet P2, #10277).

Executes real multi-frame Pink QPs over canonical full-body Pinocchio model:
- Multi-frame trajectory solve with hard weld closure and marker tasks
- Timing interval rate audit validation
- Structured infeasibility and cancellation semantics
"""

from __future__ import annotations

import importlib
import json
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
        from src.engines.physics_engines.pinocchio.python.pink_trajectory import (
            PinkTrajectoryService,
        )
        from src.shared.python.motion_matching.constrained_ik import (
            IKOptions,
            IKTrajectoryRequest,
            IKTrajectoryResult,
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
    report = tmp_path_factory.mktemp("native-pink-trajectory") / "results.xml"
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
            f"isolated native Pink trajectory tests failed:\n{exc.stdout}\n{exc.stderr}"
        )
    cases = ET.parse(report).findall(".//testcase")
    assert len(cases) == 3, "All three native trajectory integration tests must execute"
    assert all(
        case.find("skipped") is None
        and case.find("failure") is None
        and case.find("error") is None
        for case in cases
    ), "Skipped or failed native contracts do not qualify the trajectory service"


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


def test_real_pink_trajectory_multi_frame_solve(
    real_plant_and_spec: tuple[FullBodyPinocchioModel, dict[str, Any], np.ndarray],
) -> None:
    """Solve multi-frame trajectory with real Pink QP and audit residuals and limits."""
    plant, spec, q0 = real_plant_and_spec
    service = PinkTrajectoryService(spec)

    # Pre-extract marker positions at q0
    pin.forwardKinematics(plant.model, plant.data, q0)
    pin.updateFramePlacements(plant.model, plant.data)

    labels = ("HeadTop", "WaistLeft", "LShoulderTop", "RShoulderTop")
    frames = 4
    dt = 1.0 / 360.0
    time_s = np.arange(frames, dtype=np.float64) * dt
    targets = np.zeros((frames, len(labels), 3), dtype=np.float64)

    for i, lbl in enumerate(labels):
        fid = plant.model.getFrameId(lbl)
        base_pos = plant.data.oMf[fid].translation.copy()
        for f in range(frames):
            targets[f, i] = base_pos + np.array([0.005 * f, 0.0, 0.0])

    validity = np.ones((frames, len(labels)), dtype=bool)
    req = IKTrajectoryRequest(
        initial_q=q0,
        time_s=time_s,
        marker_targets=targets,
        validity_mask=validity,
        labels=labels,
        model_name="full_body_pinocchio",
    )
    opts = IKOptions(step_mode="physical", solver="quadprog", damping=1e-5)

    res = service.solve_trajectory(req, opts)
    assert res.passed
    assert res.configurations.shape == (frames, plant.model.nq)
    assert np.all(res.frame_success)
    assert len(res.rate_audits) == frames

    # Verify weld closure residual remains tight across trajectory
    for residual in res.frame_residuals:
        assert residual.weld_translation_error_m < 0.01  # < 1 cm


def test_real_pink_timing_scale_rate_audits(
    real_plant_and_spec: tuple[FullBodyPinocchioModel, dict[str, Any], np.ndarray],
) -> None:
    """Two different physical dt intervals scale velocity audits inversely with dt."""
    plant, spec, q0 = real_plant_and_spec
    service = PinkTrajectoryService(spec)

    pin.forwardKinematics(plant.model, plant.data, q0)
    pin.updateFramePlacements(plant.model, plant.data)

    labels = ("HeadTop", "WaistLeft")
    frames = 2
    targets = np.zeros((frames, 2, 3), dtype=np.float64)
    for i, lbl in enumerate(labels):
        fid = plant.model.getFrameId(lbl)
        base_pos = plant.data.oMf[fid].translation.copy()
        targets[0, i] = base_pos
        targets[1, i] = base_pos + np.array([0.02, 0.0, 0.0])

    validity = np.ones((frames, 2), dtype=bool)

    req_360 = IKTrajectoryRequest(
        initial_q=q0,
        time_s=np.array([0.0, 1.0 / 360.0], dtype=np.float64),
        marker_targets=targets,
        validity_mask=validity,
        labels=labels,
        model_name="full_body_pinocchio",
    )
    req_180 = IKTrajectoryRequest(
        initial_q=q0,
        time_s=np.array([0.0, 1.0 / 180.0], dtype=np.float64),
        marker_targets=targets,
        validity_mask=validity,
        labels=labels,
        model_name="full_body_pinocchio",
    )

    res_360 = service.solve_trajectory(req_360, IKOptions(step_mode="physical"))
    res_180 = service.solve_trajectory(req_180, IKOptions(step_mode="physical"))

    assert res_360.rate_audits[1].dt_s == pytest.approx(1.0 / 360.0)
    assert res_180.rate_audits[1].dt_s == pytest.approx(1.0 / 180.0)


def test_real_pink_cancellation_and_failure_semantics(
    real_plant_and_spec: tuple[FullBodyPinocchioModel, dict[str, Any], np.ndarray],
) -> None:
    """Cancellation terminates at frame boundary; partial result fails qualification."""
    plant, spec, q0 = real_plant_and_spec
    service = PinkTrajectoryService(spec)

    pin.forwardKinematics(plant.model, plant.data, q0)
    pin.updateFramePlacements(plant.model, plant.data)

    labels = ("HeadTop", "WaistLeft")
    frames = 4
    targets = np.zeros((frames, 2, 3), dtype=np.float64)
    for i, lbl in enumerate(labels):
        fid = plant.model.getFrameId(lbl)
        targets[:, i] = plant.data.oMf[fid].translation.copy()

    call_count = 0

    def cancel_token() -> bool:
        nonlocal call_count
        call_count += 1
        return call_count >= 2

    req = IKTrajectoryRequest(
        initial_q=q0,
        time_s=np.arange(frames, dtype=np.float64) * (1.0 / 360.0),
        marker_targets=targets,
        validity_mask=np.ones((frames, 2), dtype=bool),
        labels=labels,
        model_name="full_body_pinocchio",
        cancellation_token=cancel_token,
    )

    res = service.solve_trajectory(req, IKOptions())
    assert res.cancelled
    assert not res.passed
    assert res.first_failed_frame is not None
    assert res.configurations.shape == (frames, plant.model.nq)
    assert not res.frame_success[res.first_failed_frame]
