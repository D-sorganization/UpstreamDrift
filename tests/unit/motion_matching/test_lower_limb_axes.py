"""Lower-limb joint axes of the full-body specification must be anatomical.

Forward kinematics (MuJoCo) at the zero pose: hip flexion turns the thigh
about the pelvis mediolateral axis (the knee moves in the sagittal plane),
the knee turns about an axis perpendicular to the femur's long axis, and the
ankle pin axis is close to the mediolateral direction. The v1 document had
its hip and knee permutations swapped; v2 fixes them and this test guards it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.full_body_model import (
    NativeMujocoFullBodyModel,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_v2.json"


@pytest.fixture(scope="module")
def adapter() -> NativeMujocoFullBodyModel:
    return NativeMujocoFullBodyModel(SPEC.read_bytes())


def _anchor(adapter: NativeMujocoFullBodyModel, joint: str) -> np.ndarray:
    return adapter.data.xanchor[adapter.model.joint(joint).id].copy()


def _axis(adapter: NativeMujocoFullBodyModel, joint: str) -> np.ndarray:
    return adapter.data.xaxis[adapter.model.joint(joint).id].copy()


@pytest.mark.parametrize("side", ["r", "l"])
def test_knee_and_hip_axes_are_anatomical(
    adapter: NativeMujocoFullBodyModel, side: str
) -> None:
    zero = dict.fromkeys(adapter.coordinate_order, 0.0)
    adapter.frame_poses(zero)
    hip = _anchor(adapter, f"hip_flexion_{side}")
    knee = _anchor(adapter, f"knee_angle_{side}")
    ankle = _anchor(adapter, f"ankle_angle_{side}")
    femur = (knee - hip) / np.linalg.norm(knee - hip)
    tibia = (ankle - knee) / np.linalg.norm(ankle - knee)
    knee_axis = _axis(adapter, f"knee_angle_{side}")
    flexion_axis = _axis(adapter, f"hip_flexion_{side}")
    rotation_axis = _axis(adapter, f"hip_rotation_{side}")
    mediolateral = _anchor(adapter, "hip_flexion_r") - _anchor(adapter, "hip_flexion_l")
    mediolateral /= np.linalg.norm(mediolateral)
    # Knee flexes about an axis perpendicular to the femur, close to mediolateral.
    assert abs(knee_axis @ femur) < 0.2
    assert abs(knee_axis @ mediolateral) > 0.8
    # Hip flexion also turns about the mediolateral axis; hip rotation about the femur.
    assert abs(flexion_axis @ mediolateral) > 0.8
    assert abs(rotation_axis @ femur) > 0.9
    # At zero the shank continues the thigh (Rajagopal straight leg).
    assert femur @ tibia > 0.95
    # Flexing the knee moves the ankle in the plane normal to the knee axis.
    bent = dict(zero)
    bent[f"knee_angle_{side}"] = -1.0
    adapter.frame_poses(bent)
    moved = _anchor(adapter, f"ankle_angle_{side}") - ankle
    assert np.linalg.norm(moved) > 0.2
    assert abs(moved @ knee_axis) < 0.02


def test_ankle_pin_axis_is_mostly_mediolateral(
    adapter: NativeMujocoFullBodyModel,
) -> None:
    adapter.frame_poses(dict.fromkeys(adapter.coordinate_order, 0.0))
    mediolateral = _anchor(adapter, "hip_flexion_r") - _anchor(adapter, "hip_flexion_l")
    mediolateral /= np.linalg.norm(mediolateral)
    for side in ("r", "l"):
        assert abs(_axis(adapter, f"ankle_angle_{side}") @ mediolateral) > 0.8
