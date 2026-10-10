"""Knee flexion is negative on both sides of the anthro spec (#12057).

The spec follows the gait2392 convention: the knee hinge axis (base ``z`` column
in the femur frame) points to the subject's right on both sides, so a positive
angle swings the shank anteriorly (extension) and flexion is negative inside the
declared ``[-120, 10]`` degree range.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.execution.spec_builder import (
    HIP_MIRROR,
    knee_follows_convention,
    leg_extension,
    mirror_document_knee,
)

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_SPEC_DIR = _ROOT / "docs/development/full_body_models"
_PELVIS = "solid_reference:GolfSwing3D_Kinetic/Hips and Torso Inputs/LowerTorso"
_FLEX = np.deg2rad(20.0)


def _load(club: str) -> dict:
    path = _SPEC_DIR / f"full_body_spec_anthro_{club}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _ankle_in_pelvis(spec: dict, side: str, knee_rad: float) -> np.ndarray:
    from src.tools.tour_matching_viewer.core import body_poses_from_state

    q = dict.fromkeys(spec["coordinate_order"], 0.0)
    q[f"knee_angle_{side}"] = knee_rad
    poses = body_poses_from_state(spec, q)
    pelvis = poses[_PELVIS]
    ankle = poses[f"talus_{side}"][:3, 3]
    return pelvis[:3, :3].T @ (ankle - pelvis[:3, 3])


def _unmirror(spec: dict, side: str) -> dict:
    """Spec with ``knee_<side>`` returned to its pre-fix (un-mirrored) layout."""
    out = copy.deepcopy(spec)
    joint = next(j for j in out["joints"] if j["name"] == f"knee_{side}")
    for key in ("parent_to_base", "child_to_follower"):
        joint[key] = (np.asarray(joint[key]) @ HIP_MIRROR).tolist()
    return out


@pytest.mark.parametrize("club", ["driver", "iron7"])
@pytest.mark.parametrize("side", ["r", "l"])
def test_negative_knee_flexion_moves_the_ankle_posteriorly(
    club: str, side: str
) -> None:
    spec = _load(club)
    lo, hi = spec["coordinate_ranges_deg"][f"knee_angle_{side}"]
    assert lo <= -np.rad2deg(_FLEX) and hi >= 0.0
    zero = _ankle_in_pelvis(spec, side, 0.0)
    flexed = _ankle_in_pelvis(spec, side, -_FLEX)
    extended = _ankle_in_pelvis(spec, side, _FLEX)
    assert flexed[0] - zero[0] < -0.02, "flexion must swing the shank back"
    assert extended[0] - zero[0] > 0.02


@pytest.mark.parametrize("club", ["driver", "iron7"])
def test_committed_knees_follow_the_convention(club: str) -> None:
    for joint in _load(club)["joints"]:
        if joint["name"] in ("knee_r", "knee_l"):
            assert knee_follows_convention(joint), joint["name"]


@pytest.mark.parametrize("club", ["driver", "iron7"])
def test_mirror_leaves_the_zero_pose_unchanged(club: str) -> None:
    spec = _load(club)
    old = _unmirror(spec, "r")
    assert not any(
        knee_follows_convention(j) for j in old["joints"] if j["name"] == "knee_r"
    )
    np.testing.assert_allclose(
        _ankle_in_pelvis(spec, "r", 0.0), _ankle_in_pelvis(old, "r", 0.0), atol=1e-12
    )
    # The mirror flips the sign of the angle, nothing else.
    np.testing.assert_allclose(
        _ankle_in_pelvis(spec, "r", -_FLEX),
        _ankle_in_pelvis(old, "r", _FLEX),
        atol=1e-12,
    )


def test_mirror_document_knee_is_idempotent_and_validates() -> None:
    spec = _unmirror(_load("driver"), "r")
    assert mirror_document_knee(spec, "r") is True
    after = json.dumps(spec, sort_keys=True)
    assert mirror_document_knee(spec, "r") is False
    assert json.dumps(spec, sort_keys=True) == after
    assert mirror_document_knee(spec, "l") is False
    with pytest.raises(ValueError, match="knee_x"):
        mirror_document_knee(spec, "x")
    with pytest.raises(ValueError, match="knee_r"):
        mirror_document_knee({"joints": []}, "r")


@pytest.mark.parametrize("axis_sign", [1.0, -1.0])
def test_leg_extension_gives_positive_z_knee_axes_on_both_sides(
    axis_sign: float,
) -> None:
    """Whatever the OpenSim knee-axis sign, the built hinge axis points to +z."""
    from src.shared.python.motion_matching.execution import spec_builder as sb

    bodies = {
        f"{name}_{side}": {
            "mass": 1.0,
            "com": (0.0,) * 3,
            "inertia": (1, 1, 1, 0, 0, 0),
        }
        for side in ("r", "l")
        for name in sb.LEG_BODIES
    }
    zero = {
        f"{w}_{k}": [0.0, 0.0, 0.0]
        for w in ("parent", "child")
        for k in ("orientation", "translation")
    }
    joints = {
        f"{name}_{side}": dict(zero)
        for side in ("r", "l")
        for name in ("hip", "walker_knee", "ankle", "subtalar", "mtp")
    }
    # Ry(+/-90 deg) puts the OpenSim primary (x) axis along -z / +z.
    for side, sign in (("r", axis_sign), ("l", -axis_sign)):
        for which in ("parent", "child"):
            joints[f"walker_knee_{side}"][f"{which}_orientation"] = [
                0.0,
                sign * np.pi / 2,
                0.0,
            ]
    extension, _ = leg_extension(bodies, joints, "pelvis", np.eye(4))
    knees = {e.name: np.asarray(e.parent_to_base) for e in extension.joints}
    assert knees["knee_r"][2, 2] > 0.5
    assert knees["knee_l"][2, 2] > 0.5


def test_rajagopal_knee_sign_is_minus_one_on_both_sides() -> None:
    """Spec flexion is negative, Rajagopal's is positive, on both sides."""
    from src.engines.physics_engines.opensim.python.musculoskeletal_graft_validation import (  # noqa: E501
        RAJAGOPAL_SIGNS,
    )

    assert RAJAGOPAL_SIGNS["r"]["knee"] == -1.0
    assert RAJAGOPAL_SIGNS["l"]["knee"] == -1.0


def test_myosuite_map_flips_the_right_knee() -> None:
    """MyoSuite knee flexion is positive; the spec's is negative (right side)."""
    path = (
        _ROOT / "src/engines/physics_engines/myosuite/python/coordinate_map_anthro.json"
    )
    rows = json.loads(path.read_text(encoding="utf-8"))["mappings"]
    signs = {r["source"]: r["sign"] for r in rows}
    assert signs["knee_angle_r"] == -1.0
