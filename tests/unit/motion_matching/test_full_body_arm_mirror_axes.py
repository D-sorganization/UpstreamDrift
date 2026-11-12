"""Left/right arm joint axes are sagittal mirrors at the zero pose (G1, #14746).

A rotation about axis ``a`` reflects through the sagittal plane
``M = diag(1, -1, 1)`` into a rotation about ``-M @ a`` (rotations are
pseudovectors). So a positive left coordinate and the same positive right
coordinate must give mirrored motion: ``axis_R == -M @ axis_L``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit

MODELS = Path(__file__).resolve().parents[3] / "docs/development/full_body_models"
SPECS = ["full_body_spec_anthro_driver.json", "full_body_spec_anthro_driver_v2.json"]
MIRROR = np.diag([1.0, -1.0, 1.0])
AXIS_COL = {"Rx": 0, "Ry": 1, "Rz": 2}
CHAIN = (
    ("Scapula", "Hubto{s}S"),
    ("Shoulder", "{s}UpperArm"),
    ("Elbow", None),
    ("Forearm", "{s}LowerForearm"),
    ("Wrist", None),
)


def _joint_axes(spec: dict, side: str) -> dict[str, np.ndarray]:
    """Axis of every arm coordinate in the torso frame at the zero pose."""
    joints = {j["name"]: j for j in spec["joints"]}
    word = "Left" if side == "L" else "Right"
    frame = np.eye(3)
    axes: dict[str, np.ndarray] = {}
    for key in ("Scapula", "Shoulder", "Elbow", "Forearm", "Wrist"):
        joint = next(j for n, j in joints.items() if f"{word} {key}" in n)
        pb = np.asarray(joint["parent_to_base"], float)[:3, :3]
        cf = np.asarray(joint["child_to_follower"], float)[:3, :3]
        base = frame @ pb
        for prim in joint["primitives"]:
            axes[prim["coordinate"]] = base[:, AXIS_COL[prim["primitive"]]]
        frame = base @ np.linalg.inv(cf)
    return axes


@pytest.mark.parametrize("spec_name", SPECS)
def test_right_arm_axes_mirror_left(spec_name: str) -> None:
    path = MODELS / spec_name
    if not path.exists():
        pytest.skip(f"{spec_name} not present")
    spec = json.loads(path.read_text(encoding="utf-8"))
    left, right = _joint_axes(spec, "L"), _joint_axes(spec, "R")
    bad = []
    for name, axis in left.items():
        other = right["R" + name[1:]]
        if not np.allclose(other, -MIRROR @ axis, atol=1e-9):
            bad.append((name, np.round(axis, 3).tolist(), np.round(other, 3).tolist()))
    assert not bad, f"non-mirrored right-arm coordinates (L axis, R axis): {bad}"


def _zero_pose_frames(spec: dict) -> dict[str, np.ndarray]:
    """Child body frame at q = 0: parent @ parent_to_base @ inv(child_to_follower)."""
    by_child = {j["child"]: j for j in spec["joints"]}
    frames: dict[str, np.ndarray] = {}

    def frame(body: str) -> np.ndarray:
        if body in frames:
            return frames[body]
        joint = by_child.get(body)
        if joint is None:
            return np.eye(4)
        pose = (
            frame(joint["parent"])
            @ np.asarray(joint["parent_to_base"], float)
            @ np.linalg.inv(np.asarray(joint["child_to_follower"], float))
        )
        frames[body] = pose
        return pose

    for body in by_child:
        frame(body)
    return frames


def _unmirrored(spec: dict) -> dict:
    """The same spec with the right arm back in the native axis convention."""
    from src.shared.python.motion_matching.arm_mirror import ARM_FLIP

    out = json.loads(json.dumps(spec))
    for key, flip in ARM_FLIP.items():
        joint = next(j for j in out["joints"] if f"Right {key}" in j["name"])
        for field in ("parent_to_base", "child_to_follower"):
            joint[field] = (np.asarray(joint[field], float) @ flip).tolist()
    return out


@pytest.mark.parametrize("spec_name", [*SPECS, "full_body_spec_anthro_iron7.json"])
def test_mirror_leaves_zero_pose_frames_and_dorsal_axes_unchanged(
    spec_name: str,
) -> None:
    """The half turns relabel the axes only: bodies and dorsal axes do not move."""
    from src.shared.python.model_appearance.grip_pose import Hand
    from src.shared.python.model_appearance.hand_spec import (
        wrist_follower_dorsal_axis,
    )

    from src.shared.python.motion_matching.arm_mirror import (
        arm_is_mirrored,
        mirror_document_arm,
    )

    mirrored = json.loads((MODELS / spec_name).read_text(encoding="utf-8"))
    if not arm_is_mirrored(mirrored):  # pragma: no cover
        mirror_document_arm(mirrored)
    native = _unmirrored(mirrored)
    after, before = _zero_pose_frames(mirrored), _zero_pose_frames(native)
    assert after.keys() == before.keys()
    for body, pose in after.items():
        assert np.allclose(pose, before[body], atol=1e-9), body
    for hand in (Hand.LEAD, Hand.TRAIL):
        assert np.allclose(
            wrist_follower_dorsal_axis(mirrored, hand),
            wrist_follower_dorsal_axis(native, hand),
            atol=1e-9,
        ), hand


FIXTURES = Path(__file__).resolve().parents[2] / "fixtures/club_face"


def _rot(prim: str, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    out = np.eye(4)
    i, j = {"Rx": (1, 2), "Ry": (2, 0), "Rz": (0, 1)}[prim]
    out[i, i], out[i, j], out[j, i], out[j, j] = c, -s, s, c
    return out


def _frames_at(spec: dict, q: np.ndarray) -> dict[str, np.ndarray]:
    """World pose of every body reached through rotational joints at pose ``q``."""
    order = {name: k for k, name in enumerate(spec["coordinate_order"])}
    by_child = {j["child"]: j for j in spec["joints"]}
    frames: dict[str, np.ndarray] = {}

    def frame(body: str) -> np.ndarray:
        if body not in frames:
            joint = by_child.get(body)
            if joint is None:
                frames[body] = np.eye(4)
            else:
                motion = np.eye(4)
                for prim in joint["primitives"]:
                    k = order.get(prim["coordinate"])
                    if k is not None and prim["primitive"] in ("Rx", "Ry", "Rz"):
                        motion = motion @ _rot(prim["primitive"], q[k])
                frames[body] = (
                    frame(joint["parent"])
                    @ np.asarray(joint["parent_to_base"], float)
                    @ motion
                    @ np.linalg.inv(np.asarray(joint["child_to_follower"], float))
                )
        return frames[body]

    for body in by_child:
        frame(body)
    return frames


def test_native_and_mirrored_replay_give_identical_body_frames() -> None:
    """Migration proof: native spec + native q == mirrored spec + migrated q.

    Samples 30 of the 908 driver-swing frames, equal to 1e-9, so the
    stored-coordinate negation is exactly compensated by the axes.
    """
    mirrored = json.loads((MODELS / SPECS[0]).read_text(encoding="utf-8"))
    native = _unmirrored(mirrored)
    swing_new = np.load(FIXTURES / "swing_q_driver.npz")["q"]
    swing_old = np.load(FIXTURES / "swing_q_driver_native.npz")["q"]
    assert swing_new.shape == swing_old.shape
    assert swing_new.shape[0] == 908
    for k in np.linspace(0, swing_new.shape[0] - 1, 30).astype(int):
        after = _frames_at(mirrored, swing_new[k])
        before = _frames_at(native, swing_old[k])
        assert after.keys() == before.keys()
        for body, pose in after.items():
            assert np.allclose(pose, before[body], atol=1e-9), (int(k), body)


@pytest.mark.parametrize("fixture", ["swing_q_iron7.npz", "tracked_q_iron7.npz"])
def test_iron7_migrated_q_replays_to_the_native_body_frames(fixture: str) -> None:
    """Iron7 (SHOW-A17, #14753): negating the six flipped columns is exact.

    The committed iron7 q are stored in the mirrored convention; the native
    spec with those columns negated back must give the same body frames.
    """
    from src.shared.python.motion_matching.arm_mirror import FLIPPED_COORDINATES

    mirrored = json.loads(
        (MODELS / "full_body_spec_anthro_iron7.json").read_text(encoding="utf-8")
    )
    native = _unmirrored(mirrored)
    order = mirrored["coordinate_order"]
    cols = [i for i, name in enumerate(order) if name in FLIPPED_COORDINATES]
    assert len(cols) == len(FLIPPED_COORDINATES)
    q_new = np.load(FIXTURES / fixture)["q"].astype(float)
    q_old = q_new.copy()
    q_old[:, cols] *= -1.0
    for k in np.linspace(0, q_new.shape[0] - 1, 20).astype(int):
        after = _frames_at(mirrored, q_new[k])
        before = _frames_at(native, q_old[k])
        for body, pose in after.items():
            assert np.allclose(pose, before[body], atol=1e-9), (int(k), body)


def test_restart_jitter_is_mirror_equivariant() -> None:
    """A mirrored spec draws the reflected restart starts of the native spec.

    The mirror is a re-parameterisation (``q' = S q`` on the flipped arm
    coordinates), so the multi-start jitter must satisfy ``jitter' = S jitter``
    for the same RNG stream; otherwise the two conventions sample different
    starts and the optimizer lands in different minima (#14753).
    """
    from src.shared.python.motion_matching.arm_mirror import (
        FLIPPED_COORDINATES,
        document_arm_mirrored,
        restart_jitter,
    )

    mirrored = json.loads((MODELS / SPECS[0]).read_text(encoding="utf-8"))
    native = _unmirrored(mirrored)
    assert document_arm_mirrored(mirrored) and not document_arm_mirrored(native)
    order = tuple(mirrored["coordinate_order"])
    noise_native = restart_jitter(np.random.default_rng(7), 0.5, False, order)
    noise_mirrored = restart_jitter(np.random.default_rng(7), 0.5, True, order)
    assert noise_native.shape == (len(order) - 6,)
    sign = np.array([-1.0 if n in FLIPPED_COORDINATES else 1.0 for n in order[6:]])
    assert np.array_equal(noise_mirrored, sign * noise_native)
    assert np.abs(noise_native).max() <= 0.5


@pytest.mark.parametrize("document", [{}, {"joints": []}, {"coordinate_order": []}])
def test_stub_specs_are_not_mirrored(document: dict) -> None:
    from src.shared.python.motion_matching.arm_mirror import document_arm_mirrored

    assert document_arm_mirrored(document) is False


def test_mujoco_kinematics_reports_the_spec_mirror_state() -> None:
    """The pipeline's kinematics must know the convention its jitter reflects.

    ``FullBodyMarkerKinematics`` is built from an adapter and carries no spec
    document, so it has to inherit the flag; otherwise the restarts never
    reflect and the mirrored optimizer diverges from the native one (#14753).
    """
    pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_ik import (
        FullBodyMarkerKinematics,
    )
    from src.engines.physics_engines.mujoco.python.full_body_model import (
        NativeMujocoFullBodyModel,
    )

    mirrored = json.loads((MODELS / SPECS[0]).read_text(encoding="utf-8"))
    for doc, want in ((mirrored, True), (_unmirrored(mirrored), False)):
        adapter = NativeMujocoFullBodyModel(json.dumps(doc).encode())
        body = next(iter(adapter.body_frames))
        kin = FullBodyMarkerKinematics(adapter, {"m": (body, (0.0, 0.0, 0.0))})
        assert kin.arm_mirrored is want
