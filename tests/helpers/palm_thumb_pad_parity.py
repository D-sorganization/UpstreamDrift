"""Palm-thumb pad force series per engine (HRG-H2 parity, #13190 PR 2).

Each engine contributes only its forward kinematics of the two grip-hand
frames (``grip_bushing[side]`` ``hand_frame`` on ``hand_body``); the force is
always the shared :func:`palm_thumb_pad_force`. Frame velocities are central
differences of the frame poses with the same step for every source, so an
engine differs from the spec reference only through its kinematics.

The spec has no closed-grip configuration (at zero and at the address seed the
grip-hand frames are ~0.38 m apart), so each side is put on its grip by one
fixed rigid map ``A_s = G_s @ inv(S_s(q0))``: ``G_s`` is the grip interface
frame and ``S_s(q0)`` the spec-analytic hand frame at zero. ``A_s`` comes from
the spec alone and is applied to every source alike, so an engine's frame
error (offset or motion) passes through to the force unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from src.shared.python.grip_contact import BushingState, GripInterface
from src.shared.python.grip_contact.contact_topology import GripStyle, contact_topology
from src.shared.python.grip_contact.hand_drift import HandDrift
from src.shared.python.grip_contact.palm_thumb_pad import (
    hand_drift_gate,
    palm_thumb_pad_force,
)
from src.shared.python.grip_contact.topology_contact import build_topology_model
from src.shared.python.motion_matching.grip_contract import GripSpec
from tests.helpers.hand_consistency import (
    engine_frame_poses,
    separated_flagship_doc,
    spec_frame_poses,
)

PAD_QUANTITIES = ("normal_force", "force_on_trail")
#: Trail-wrist oscillation: amplitude (rad), frequency (Hz), samples over one period.
WRIST_COORDINATES = ("RWInputX", "RWInputY")
AMPLITUDE_RAD = 0.002
FREQUENCY_HZ = 10.0
SAMPLES = 21
_DIFF_STEP_S = 1.0e-5
_PAD_FRAMES = {"L": "H2PAD_L", "R": "H2PAD_R"}

PoseReader = Callable[[dict[str, float]], dict[str, Any]]


def pad_parity_doc() -> dict[str, Any]:
    """The separated flagship spec plus one spec frame on each grip-hand frame."""
    doc = json.loads(json.dumps(separated_flagship_doc()))
    for side, name in _PAD_FRAMES.items():
        grip = doc["grip_bushing"][side]
        doc["frames"].append(
            {"name": name, "body": grip["hand_body"], "placement": grip["hand_frame"]}
        )
    return doc


def wrist_trajectory(doc: dict[str, Any]) -> Callable[[float], dict[str, float]]:
    """``q(t)``: zeros plus a trail-wrist circle through zero at ``t = 0``."""
    zeros = dict.fromkeys(doc["coordinate_order"], 0.0)
    omega = 2.0 * np.pi * FREQUENCY_HZ

    def q_at(t: float) -> dict[str, float]:
        x, y = WRIST_COORDINATES
        return {
            **zeros,
            x: AMPLITUDE_RAD * float(np.sin(omega * t)),
            y: AMPLITUDE_RAD * float(1.0 - np.cos(omega * t)),
        }

    return q_at


def _grip_anchors(doc: dict[str, Any], interface: GripInterface) -> dict:
    neutral = spec_frame_poses(doc, dict.fromkeys(doc["coordinate_order"], 0.0))
    anchors = {}
    for side, grip in (("L", interface.left), ("R", interface.right)):
        g = np.eye(4)
        g[:3, :3], g[:3, 3] = grip.rotation, grip.position_m
        anchors[side] = g @ np.linalg.inv(np.asarray(neutral[_PAD_FRAMES[side]]))
    return anchors


def _state(poses_of: PoseReader, q_at: Callable, t: float, side: str, anchor):
    before, now, after = (
        anchor @ np.asarray(poses_of(q_at(t + dt))[_PAD_FRAMES[side]], dtype=float)
        for dt in (-_DIFF_STEP_S, 0.0, _DIFF_STEP_S)
    )
    span = 2.0 * _DIFF_STEP_S
    turn = Rotation.from_matrix(after[:3, :3] @ before[:3, :3].T).as_rotvec()
    return BushingState(
        now[:3, :3].copy(),
        now[:3, 3].copy(),
        (after[:3, 3] - before[:3, 3]) / span,
        turn / span,
    )


def pad_series(poses_of: PoseReader, doc: dict[str, Any]) -> dict[str, np.ndarray]:
    """Normal force (N) and force on the trail hand (N) along the trajectory."""
    interface = GripInterface.from_spec(doc)
    model = build_topology_model(
        interface, topology=contact_topology(GripSpec(style=GripStyle.OVERLAP))
    )
    anchors = _grip_anchors(doc, interface)
    zero = np.zeros(3)
    gate = hand_drift_gate(HandDrift(along_mm=zero, across_mm=zero, rotation_deg=zero))
    q_at = wrist_trajectory(doc)
    normal, on_trail = [], []
    for t in np.linspace(0.0, 1.0 / FREQUENCY_HZ, SAMPLES):
        lead, trail = (_state(poses_of, q_at, t, s, anchors[s]) for s in "LR")
        pad = palm_thumb_pad_force(model, lead, trail, gate=gate)
        if not pad.available:
            raise AssertionError(f"pad unavailable at t={t:.4f} s: {pad.reason}")
        normal.append(pad.normal_force_n)
        on_trail.append(pad.force_on_trail_n)
    return {"normal_force": np.asarray(normal), "force_on_trail": np.asarray(on_trail)}


def spec_pad_series(doc: dict[str, Any]) -> dict[str, np.ndarray]:
    """The reference series: the pad law on the spec-analytic frames."""
    return pad_series(lambda q: spec_frame_poses(doc, q), doc)


def engine_pad_series(engine: str, doc: dict[str, Any]) -> dict[str, np.ndarray]:
    """The pad law on ``engine``'s own forward kinematics of the same frames."""
    return pad_series(engine_frame_poses(engine, doc), doc)
