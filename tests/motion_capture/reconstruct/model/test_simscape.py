"""Axis-convention identification (#9714): synthetic proof, then the real logs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.motion_capture.reconstruct.model.simscape import (
    JOINTS,
    Frames,
    identify_joint,
    load_trials,
    markdown,
    validate,
)

pytestmark = pytest.mark.unit

DATASETS = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "engines"
    / "Simscape_Multibody_Models"
    / "3D_Golf_Model"
    / "matlab"
    / "Scripts"
    / "Dataset Generator"
)


def _synthetic_frames(order: str, signs: tuple[int, ...], seed: int = 0) -> Frames:
    """Parent world rotations, a joint with known primitives and fixed frames."""
    rng = np.random.default_rng(seed)
    t = 60
    parent = Rotation.random(t, random_state=1).as_matrix()
    a = Rotation.from_rotvec([0.3, -0.2, 1.1]).as_matrix()
    b = Rotation.from_rotvec([-1.0, 0.4, 0.2]).as_matrix()
    angles = rng.uniform(-90, 90, (t, len(order)))
    prim = Rotation.from_euler(
        order.upper(), np.radians(angles) * np.asarray(signs)
    ).as_matrix()
    rel = np.einsum("ij,tjk,kl->til", a, prim, b)
    child = np.einsum("tij,tjk->tik", parent, rel)
    offset = np.array([0.0, 0.0, -0.254])
    p_parent = rng.normal(size=(t, 3))
    p_child = p_parent + np.einsum("tij,j->ti", parent, offset)
    parent_name, child_name, candidates = JOINTS["left_shoulder"]
    cols = candidates[0]
    angle_cols = {c: angles[:, k] for k, c in enumerate(cols[: len(order)])}
    # Fill every other joint's columns so load-shaped access never fails.
    for _, _, other in JOINTS.values():
        for group in other:
            for c in group:
                angle_cols.setdefault(c, np.zeros(t))
    rot = {
        name: np.broadcast_to(np.eye(3), (t, 3, 3)).copy()
        for name in ("Spine", "Torso", "RScap", "RS", "LF", "RF")
    }
    pos = {name: np.zeros((t, 3)) for name in rot}
    rot[parent_name], rot[child_name] = parent, child
    pos[parent_name], pos[child_name] = p_parent, p_child
    return Frames(rotation=rot, position=pos, angles_deg=angle_cols, frames=t)


@pytest.mark.parametrize(
    "order,signs", [("xyz", (1, 1, 1)), ("yzx", (-1, -1, 1)), ("zxy", (1, -1, -1))]
)
@pytest.mark.timeout(180)  # exhaustive hypothesis search; CI's default is 60 s
def test_identification_recovers_a_known_convention(
    order: str, signs: tuple[int, ...]
) -> None:
    frames = _synthetic_frames(order, signs)
    found = identify_joint(frames, "left_shoulder", seeds=4)  # exact data; CI budget
    assert found.residual_rad < 1e-6
    assert found.offset_spread_m < 1e-9
    np.testing.assert_allclose(
        found.offset_parent_frame_m, [0.0, 0.0, -0.254], atol=1e-9
    )
    # The recovered primitives reproduce the relative rotation exactly; the
    # sequence itself may be an equivalent one (Euler sequences are not unique
    # once constant frames are free), so the test checks the residual, not
    # the label.
    assert found.reference == "parent"
    assert found.angle_columns == JOINTS["left_shoulder"][2][0]


def test_markdown_lists_every_joint() -> None:
    frames = _synthetic_frames("xyz", (1, 1, 1))
    report = {
        "joints": {
            "j": {
                "parent": "P",
                "child": "C",
                "reference": "parent",
                "sequence": "x+",
                "residual_rad": 1e-4,
                "pre_frame_rotvec": [0, 0, 0],
                "post_frame_rotvec": [0, 0, 0],
                "offset_parent_frame_m": [0, 0, 0.1],
                "offset_spread_m": 1e-5,
            }
        }
    }
    text = markdown(report)
    assert text.startswith("| joint") and "| j | P → C |" in text
    assert frames.frames == 60


@pytest.mark.slow
@pytest.mark.skipif(
    not DATASETS.is_dir(), reason="dataset generator logs not checked out"
)
def test_shoulder_gimbal_and_strut_validate_on_the_real_logs() -> None:
    files = sorted(DATASETS.glob("golf_swing_dataset_*/trial_*.csv"))[:4]
    frames = load_trials(files)
    for side in ("left", "right"):
        found = identify_joint(frames, f"{side}_shoulder")
        assert found.residual_rad < 1e-3
        assert abs(abs(found.offset_parent_frame_m[2]) - 0.254) < 1e-3
        assert found.offset_spread_m < 1e-3
    report = validate(files)
    assert set(report["joints"]) == set(JOINTS)
