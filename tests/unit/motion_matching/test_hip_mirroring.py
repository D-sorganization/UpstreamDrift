"""The left hip is the sagittal mirror of the right (OSV-6, #11737).

OpenSim's Rajagopal hips mirror the left side through ``LinearFunction``
coefficients of -1 on ``hip_adduction_l`` and ``hip_rotation_l``. The spec must
keep that convention, so equal coordinate values on both sides give mirror-image
femur orientations and the shared ranges of motion are anatomical on both sides.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.execution.spec_builder import (
    hip_axis_signs,
)

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_OSIM = _ROOT / "src/engines/physics_engines/opensim/models/golf_humanoid.osim"
_SPEC = _ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"

#: Sagittal reflections: pelvis ``Hip`` frame (x fwd, y left, z up) and the
#: OpenSim femur frame (z lateral).
_PELVIS_MIRROR = np.diag([1.0, -1.0, 1.0])
_FEMUR_MIRROR = np.diag([1.0, 1.0, -1.0])


def _axis_rotation(axis: str, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return {
        "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
        "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
        "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
    }[axis]


def _hip_rotation(joint: dict, q: np.ndarray) -> np.ndarray:
    """Femur-to-pelvis rotation: ``parent_to_base * R(q) * inv(child_to_follower)``."""
    rotation = np.asarray(joint["parent_to_base"])[:3, :3]
    for primitive, angle in zip(joint["primitives"], q, strict=True):
        rotation = rotation @ _axis_rotation(primitive["primitive"][1].lower(), angle)
    return rotation @ np.asarray(joint["child_to_follower"])[:3, :3].T


def _hips() -> dict[str, dict]:
    from src.shared.python.humanoid_character_builder.presets.loader import (
        load_character_preset,
    )
    from src.shared.python.humanoid_character_builder.spec_params import (
        compile_full_body_spec,
    )

    document = compile_full_body_spec(load_character_preset("anthro_driver").parameters)
    return {j["name"]: j for j in document["joints"] if j["name"] in ("hip_r", "hip_l")}


def test_osim_mirrors_only_the_left_hip() -> None:
    assert hip_axis_signs(_OSIM) == {"r": 1.0, "l": -1.0}


@pytest.mark.parametrize(
    "q",
    [(0.3, 0.0, 0.0), (0.0, 0.25, 0.0), (0.0, 0.0, 0.4), (0.5, -0.3, 0.35)],
)
def test_equal_coordinates_give_mirror_image_femurs(q: tuple) -> None:
    hips = _hips()
    right = _hip_rotation(hips["hip_r"], np.asarray(q))
    left = _hip_rotation(hips["hip_l"], np.asarray(q))
    np.testing.assert_allclose(left, _PELVIS_MIRROR @ right @ _FEMUR_MIRROR, atol=1e-12)


def test_zero_pose_is_unchanged_by_the_mirror() -> None:
    hips = _hips()
    zero = np.zeros(3)
    np.testing.assert_allclose(
        _hip_rotation(hips["hip_l"], zero), _hip_rotation(hips["hip_r"], zero)
    )


def test_zero_twist_calibration_is_the_same_physical_twist_when_mirrored() -> None:
    """``hip_zero_twist_deg`` keeps its meaning: same femur pose either way."""
    from src.shared.python.motion_matching.execution.spec_builder import HIP_MIRROR
    from src.shared.python.motion_matching.hip_calibration import (
        HipCalibration,
        HipRotationZero,
        apply_hip_calibration,
        hip_is_mirrored,
    )

    import json

    document = json.loads(_SPEC.read_text())
    unmirrored = json.loads(_SPEC.read_text())
    for joint in unmirrored["joints"]:
        if joint["name"] == "hip_l":
            for key in ("parent_to_base", "child_to_follower"):
                joint[key] = (np.asarray(joint[key]) @ HIP_MIRROR).tolist()
            assert not hip_is_mirrored(joint)
    calibration = HipCalibration(
        centre_r=(0.0, -0.09, 0.0),
        centre_l=(0.0, 0.09, 0.0),
        radius_r_m=0.4,
        radius_l_m=0.4,
        residual_sd_r_m=0.001,
        residual_sd_l_m=0.001,
        frames=10,
        pelvis_axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        waist_fit_max_residual_m=0.001,
    )
    twist = HipRotationZero(offset_r_deg=12.0, offset_l_deg=-9.0)
    alignment = np.eye(4)
    poses = []
    for doc in (document, unmirrored):
        out = apply_hip_calibration(doc, calibration, alignment, zero_twist_deg=twist)
        hip_l = next(j for j in out["joints"] if j["name"] == "hip_l")
        poses.append(_hip_rotation(hip_l, np.zeros(3)))
    np.testing.assert_allclose(poses[0], poses[1], atol=1e-12)


def test_mirror_document_hip_is_idempotent() -> None:
    import json

    from src.shared.python.motion_matching.execution.spec_builder import (
        mirror_document_hip,
    )

    document = json.loads(_SPEC.read_text())
    before = json.dumps(document, sort_keys=True)
    assert mirror_document_hip(document, "l") is False
    assert json.dumps(document, sort_keys=True) == before
    with pytest.raises(ValueError, match="hip_x"):
        mirror_document_hip(document, "x")


@pytest.mark.parametrize("club", ["driver", "iron7"])
def test_committed_anthro_spec_is_mirrored(club: str) -> None:
    import json

    spec = _SPEC.with_name(f"full_body_spec_anthro_{club}.json")
    joints = {j["name"]: j for j in json.loads(spec.read_text())["joints"]}
    q = np.array([0.5, -0.3, 0.35])
    np.testing.assert_allclose(
        _hip_rotation(joints["hip_l"], q),
        _PELVIS_MIRROR @ _hip_rotation(joints["hip_r"], q) @ _FEMUR_MIRROR,
        atol=1e-9,
    )
