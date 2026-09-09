"""Reviewed landmark bindings for bundled native models; extensible via URDF."""

from pathlib import Path

from src.motion_capture.reconstruct.model.registry import MODELS, RegisteredModel

from .urdf_models import load_urdf_model
from .mjcf_models import load_mjcf_model

GOLFER_LINKS = {
    "pelvis": "mid_hip",
    "thorax3": "neck",
    "upper_arm_left": "left_shoulder",
    "forearm_left": "left_elbow",
    "hand_left": "left_wrist",
    "upper_arm_right": "right_shoulder",
    "forearm_right": "right_elbow",
    "hand_right": "right_wrist",
    "left_thigh": "left_hip",
    "left_shank": "left_knee",
    "left_foot": "left_ankle",
    "right_thigh": "right_hip",
    "right_shank": "right_knee",
    "right_foot": "right_ankle",
}
SIMPLE_LINKS = {
    "pelvis": "mid_hip",
    "head": "nose",
    "left_upper_arm": "left_shoulder",
    "right_upper_arm": "right_shoulder",
    "left_forearm": "left_elbow",
    "right_forearm": "right_elbow",
    "left_thigh": "left_hip",
    "right_thigh": "right_hip",
    "left_shin": "left_knee",
    "right_shin": "right_knee",
    "left_foot": "left_ankle",
    "right_foot": "right_ankle",
}
DRAKE_LINKS = {
    "pelvis": "mid_hip",
    "upper_torso_hub": "neck",
    "left_upper_arm": "left_shoulder",
    "left_forearm": "left_elbow",
    "left_hand": "left_wrist",
    "right_upper_arm": "right_shoulder",
    "right_forearm": "right_elbow",
    "right_hand": "right_wrist",
}
HUMAN_LINKS = {
    "Pelvis": "mid_hip",
    "Neck": "neck",
    "Head": "nose",
    "LeftUpperArm": "left_shoulder",
    "LeftForeArm": "left_elbow",
    "LeftHand": "left_wrist",
    "RightUpperArm": "right_shoulder",
    "RightForeArm": "right_elbow",
    "RightHand": "right_wrist",
    "LeftUpperLeg": "left_hip",
    "LeftLowerLeg": "left_knee",
    "LeftFoot": "left_ankle",
    "RightUpperLeg": "right_hip",
    "RightLowerLeg": "right_knee",
    "RightFoot": "right_ankle",
}

# Paths refer to repository-owned model geometry, not engine launchers. External
# provider packs can use load_urdf_model with their own explicit bindings.
URDF_PRESETS = {
    "pinocchio_golfer": (
        "src/engines/physics_engines/pinocchio/models/generated/golfer.urdf",
        GOLFER_LINKS,
    ),
    "pinocchio_golfer_ik": (
        "src/engines/physics_engines/pinocchio/models/generated/golfer_ik.urdf",
        GOLFER_LINKS,
    ),
    "drake_golfer": (
        "src/engines/physics_engines/drake/models/generated/golfer.urdf",
        DRAKE_LINKS,
    ),
    "simple_humanoid": (
        "src/shared/python/model_generation/library/bundled/simple_humanoid/humanoid.urdf",
        SIMPLE_LINKS,
    ),
    "human_subject": (
        "src/tools/model_explorer/bundled_assets/human_models/human_subject_with_meshes/model.urdf",
        HUMAN_LINKS,
    ),
}
MJCF_PRESETS = {
    "mujoco_humanoid": (
        "src/shared/python/model_generation/library/bundled/mujoco_humanoid/humanoid.xml",
        {
            "torso": "neck",
            "pelvis": "mid_hip",
            "right_thigh": "right_hip",
            "right_shin": "right_knee",
            "right_foot": "right_ankle",
            "left_thigh": "left_hip",
            "left_shin": "left_knee",
            "left_foot": "left_ankle",
            "right_upper_arm": "right_shoulder",
            "right_lower_arm": "right_elbow",
            "left_upper_arm": "left_shoulder",
            "left_lower_arm": "left_elbow",
        },
    ),
}


def _catalog(
    repo_root: Path,
) -> tuple[dict[str, RegisteredModel], dict[str, dict[str, str]]]:
    result = dict(MODELS)
    inventory = {
        name: {"status": "available", "source": "articulated registry"}
        for name in result
    }
    for presets, loader in (
        (URDF_PRESETS, load_urdf_model),
        (MJCF_PRESETS, load_mjcf_model),
    ):
        for name, (relative, mapping) in presets.items():
            path = repo_root / relative
            try:
                result[name] = loader(path, name=name, landmark_map=mapping)
                inventory[name] = {"status": "available", "source": relative}
            except (ImportError, ValueError, OSError) as exc:
                inventory[name] = {
                    "status": "unavailable",
                    "source": relative,
                    "reason": str(exc),
                }
    inventory["myosuite_body"] = {
        "status": "unavailable",
        "reason": "Bundled myobody/myoupperbody are labeled placeholder models, not MyoSuite anatomy.",
    }
    inventory["opensim_golfer"] = {
        "status": "unavailable",
        "reason": "Native OpenSim custom-joint/muscle constraints require an OpenSim adapter; no surrogate fit is substituted.",
    }
    return result, inventory


def available_models(repo_root: Path) -> dict[str, RegisteredModel]:
    """Return registry models plus available, validated native URDF presets.

    Missing or invalid optional models are omitted and explained by
    model_inventory; no model is replaced by a mock engine or another skeleton.
    """
    return _catalog(repo_root)[0]


def model_inventory(repo_root: Path) -> dict[str, dict[str, str]]:
    """Return capability evidence, including why a named model cannot be fitted."""
    return _catalog(repo_root)[1]
