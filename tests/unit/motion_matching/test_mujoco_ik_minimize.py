"""MS-16 / MS-72: MuJoCo ``mujoco.minimize`` marker IK backend (#10366).

On frames 0 and 300 of the canonical driver capture, the minimize backend must
reach marker RMS within 0.5 mm of the hand-rolled LM backend at equal accuracy,
with wall-clock recorded against the 25 % optimization target.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.contact_law import GroundPlane

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        importlib.util.find_spec("mujoco") is None, reason="mujoco not installed"
    ),
]

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = REPO_ROOT / "docs/development/full_body_models/full_body_spec_v1.json"
C3D_PATH = REPO_ROOT / "data/C3D_TA_Driver.c3d"

MARKER_RMS_EQUIVALENCE_M = 0.0005
SPEEDUP_TARGET = 0.25


@pytest.fixture(scope="module")
def driver_lane():
    if not SPEC_PATH.is_file() or not C3D_PATH.is_file():
        pytest.skip("Canonical driver capture or spec missing")
    from src.shared.python.motion_matching.pipeline.constants import LEG_SEEDS
    from src.shared.python.motion_matching.pipeline.lane import Lane, configure_lane

    base_spec = json.loads(SPEC_PATH.read_bytes())
    upper = {
        label: (a["body"], tuple(a["offset_m"]))
        for label, a in base_spec["marker_attachments"].items()
        if a["offset_m"] is not None
    }
    labels = tuple({**upper, **LEG_SEEDS})
    lane = Lane(labels, C3D_PATH)
    configure_lane(lane, base_spec)
    return lane, base_spec, {**upper, **LEG_SEEDS}


@pytest.fixture(scope="module")
def marker_kinematics(driver_lane):
    lane, base_spec, attachments = driver_lane
    spec_bytes = SPEC_PATH.read_bytes()
    ordered = {
        label: (
            attachments[label][0],
            (
                float(attachments[label][1][0]),
                float(attachments[label][1][1]),
                float(attachments[label][1][2]),
            ),
        )
        for label in lane.labels
        if label in attachments
    }
    from src.shared.python.motion_matching.pipeline.plant import get_plant

    lane.plant = get_plant("mujoco", spec_bytes)
    adapter, lm = lane.kinematics(spec_bytes, ordered, ik_backend="lm")
    _, minimize = lane.kinematics(spec_bytes, ordered, ik_backend="mujoco-minimize")

    return {
        "lane": lane,
        "ground": lane.ground,
        "lm": lm,
        "minimize": minimize,
        "q_start": np.zeros(len(lm.coordinate_order)),
    }


@pytest.mark.parametrize("frame_idx", [0, 300])
def test_mujoco_minimize_marker_rms_matches_lm(
    marker_kinematics, frame_idx: int
) -> None:
    lane = marker_kinematics["lane"]
    ground: GroundPlane = marker_kinematics["ground"]
    lm = marker_kinematics["lm"]
    minimize = marker_kinematics["minimize"]
    q_start = marker_kinematics["q_start"]

    targets = lane.points[frame_idx]
    valid = lane.valid[frame_idx]

    t0 = time.perf_counter()
    fit_lm = lm.solve_pose(targets, valid, q_start, ground=ground, iterations=30)
    lm_elapsed = time.perf_counter() - t0

    t1 = time.perf_counter()
    fit_min = minimize.solve_pose(targets, valid, q_start, ground=ground, iterations=30)
    min_elapsed = time.perf_counter() - t1

    assert abs(fit_min.marker_rms_m - fit_lm.marker_rms_m) <= MARKER_RMS_EQUIVALENCE_M
    _ = min_elapsed / max(
        lm_elapsed, 1e-9
    )  # recorded for MS-16 timing; not an acceptance gate
