"""Integration test: Cross-engine forward kinematics equivalence.

Issue #4542 -- Cross-engine FK equivalence (5 mm RMSE) for character builder URDFs.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path


import numpy as np
import pytest

from humanoid_character_builder.interfaces import CharacterBuilder


def get_active_joints(urdf_path: Path) -> dict[str, tuple[float, float]]:
    """Extract all non-fixed joints and their limits from URDF."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    joints = {}
    for joint in root.findall("joint"):
        jtype = joint.get("type")
        if jtype in ("revolute", "prismatic"):
            name = joint.get("name")
            limit = joint.find("limit")
            if limit is not None:
                lower = float(limit.get("lower", "-3.14"))
                upper = float(limit.get("upper", "3.14"))
                joints[name] = (lower, upper)
        elif jtype == "continuous":
            name = joint.get("name")
            joints[name] = (-math.pi, math.pi)

    return joints


def compute_mujoco_fk(
    urdf_path: Path, configs: list[dict[str, float]], body_names: list[str]
) -> np.ndarray:
    """Compute FK for all configs in MuJoCo. Returns shape (N, B, 3)."""
    mujoco = pytest.importorskip("mujoco", reason="mujoco not installed")

    try:
        model = mujoco.MjModel.from_xml_path(str(urdf_path))
    except AttributeError:
        pytest.skip("mujoco.MjModel.from_xml_path not available in this version")

    data = mujoco.MjData(model)

    body_ids = []
    for name in body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        body_ids.append(body_id)

    results = np.zeros((len(configs), len(body_names), 3))

    for i, config in enumerate(configs):
        mujoco.mj_resetData(model, data)
        for jname, val in config.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid >= 0:
                qpos_adr = model.jnt_qposadr[jid]
                data.qpos[qpos_adr] = val

        mujoco.mj_kinematics(model, data)
        for b, bid in enumerate(body_ids):
            results[i, b] = data.xpos[bid]

    return results


def compute_drake_fk(
    urdf_path: Path, configs: list[dict[str, float]], body_names: list[str]
) -> np.ndarray:
    """Compute FK for all configs in Drake. Returns shape (N, B, 3)."""
    pytest.importorskip("pydrake.multibody", reason="pydrake not installed")
    from pydrake.multibody.parsing import Parser
    from pydrake.multibody.plant import MultibodyPlant

    plant = MultibodyPlant(time_step=0.0)
    parser = Parser(plant)
    if hasattr(parser, "AddModels"):
        parser.AddModels(str(urdf_path))
    else:
        parser.AddModelFromFile(str(urdf_path))
    plant.Finalize()

    context = plant.CreateDefaultContext()

    results = np.zeros((len(configs), len(body_names), 3))

    for i, config in enumerate(configs):
        for jname, val in config.items():
            if plant.HasJointNamed(jname):
                joint = plant.GetJointByName(jname)
                # Ensure it's 1DOF
                if joint.num_positions() == 1:
                    joint.set_angle(context, val)

        # evaluate FK
        for b, bname in enumerate(body_names):
            if plant.HasBodyNamed(bname):
                body = plant.GetBodyByName(bname)
                pose = plant.EvalBodyPoseInWorld(context, body)
                results[i, b] = pose.translation()

    return results


def compute_pinocchio_fk(
    urdf_path: Path, configs: list[dict[str, float]], body_names: list[str]
) -> np.ndarray:
    """Compute FK for all configs in Pinocchio. Returns shape (N, B, 3)."""
    pin = pytest.importorskip("pinocchio", reason="pinocchio not installed")
    if not hasattr(pin, "__version__"):
        pytest.skip("pinocchio appears to be a stub/mock")
    if not hasattr(pin, "buildModelFromUrdf"):
        pytest.skip("pinocchio buildModelFromUrdf not available")

    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()

    results = np.zeros((len(configs), len(body_names), 3))

    for i, config in enumerate(configs):
        q = pin.neutral(model)
        for jname, val in config.items():
            if model.existJointName(jname):
                jid = model.getJointId(jname)
                q_idx = model.idx_qs[jid]
                q[q_idx] = val

        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        for b, bname in enumerate(body_names):
            # Use frame placement for both bodies and frames.
            # model.getBodyId() returns a body/frame index, but data.oMi stores
            # joint placements. Using oMf ensures we read the correct transform.
            if model.existBodyName(bname):
                bid = model.getBodyId(bname)
                results[i, b] = data.oMf[bid].translation
            elif model.existFrame(bname):
                fid = model.getFrameId(bname)
                results[i, b] = data.oMf[fid].translation

    return results


def log_result_to_report(engine_pair: str, max_rmse: float) -> None:
    """Log the RMSE reference numbers to the central JSON report."""
    report_file = Path("reports/urdf_cross_engine.json")
    report_file.parent.mkdir(parents=True, exist_ok=True)

    if report_file.exists():
        try:
            with open(report_file, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}

    data[engine_pair] = {
        "max_rmse_m": float(max_rmse),
        "max_rmse_mm": float(max_rmse * 1000.0),
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


@pytest.fixture(scope="module")
def shared_urdf_and_configs(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, list[dict[str, float]], list[str]]:
    """Generate URDF and random joint configurations once for the module."""
    tmp_dir = tmp_path_factory.mktemp("cross_engine_fk")
    builder = CharacterBuilder()
    params = CharacterBuilder.create_from_preset("average")
    result = builder.build(params, mesh_output_dir=tmp_dir)
    urdf_path = result.export_urdf(tmp_dir)

    joints = get_active_joints(urdf_path)

    # Sample 100 random configs
    random.seed(42)
    configs = []
    for _ in range(100):
        cfg = {}
        for jname, (lower, upper) in joints.items():
            cfg[jname] = random.uniform(lower, upper)
        configs.append(cfg)

    body_names = [
        "left_hand",
        "right_hand",
        "left_foot",
        "right_foot",
        "head",
        "pelvis",
    ]

    return urdf_path, configs, body_names


def test_cross_engine_fk_mujoco_drake(
    shared_urdf_and_configs: tuple[Path, list[dict[str, float]], list[str]],
) -> None:
    urdf_path, configs, body_names = shared_urdf_and_configs

    res_mj = compute_mujoco_fk(urdf_path, configs, body_names)
    res_dk = compute_drake_fk(urdf_path, configs, body_names)

    # Calculate RMSE across configs
    diff = res_mj - res_dk
    rmse = np.sqrt(np.mean(diff**2, axis=0))  # shape: (B, 3)
    max_rmse = np.max(np.linalg.norm(rmse, axis=1))  # max over bodies

    log_result_to_report("mujoco_vs_drake", max_rmse)

    # Assert max RMSE <= 5 mm (0.005 m)
    assert max_rmse <= 0.005, f"MuJoCo vs Drake RMSE {max_rmse:.4f} m > 5 mm threshold"


def test_cross_engine_fk_mujoco_pinocchio(
    shared_urdf_and_configs: tuple[Path, list[dict[str, float]], list[str]],
) -> None:
    urdf_path, configs, body_names = shared_urdf_and_configs

    res_mj = compute_mujoco_fk(urdf_path, configs, body_names)
    res_pn = compute_pinocchio_fk(urdf_path, configs, body_names)

    diff = res_mj - res_pn
    rmse = np.sqrt(np.mean(diff**2, axis=0))
    max_rmse = np.max(np.linalg.norm(rmse, axis=1))

    log_result_to_report("mujoco_vs_pinocchio", max_rmse)

    assert max_rmse <= 0.005, (
        f"MuJoCo vs Pinocchio RMSE {max_rmse:.4f} m > 5 mm threshold"
    )


def test_cross_engine_fk_drake_pinocchio(
    shared_urdf_and_configs: tuple[Path, list[dict[str, float]], list[str]],
) -> None:
    urdf_path, configs, body_names = shared_urdf_and_configs

    res_dk = compute_drake_fk(urdf_path, configs, body_names)
    res_pn = compute_pinocchio_fk(urdf_path, configs, body_names)

    diff = res_dk - res_pn
    rmse = np.sqrt(np.mean(diff**2, axis=0))
    max_rmse = np.max(np.linalg.norm(rmse, axis=1))

    log_result_to_report("drake_vs_pinocchio", max_rmse)

    assert max_rmse <= 0.005, (
        f"Drake vs Pinocchio RMSE {max_rmse:.4f} m > 5 mm threshold"
    )
