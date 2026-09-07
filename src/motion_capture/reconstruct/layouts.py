"""Map detector layouts onto the 15-joint reconstruct skeleton.

MediaPipe reports 33 landmarks and BODY_25 reports 25; the fit works on the
15 joints both share plus ``mid_hip`` and ``neck``, which are derived as the
midpoints of the hips and of the shoulders. A derived joint carries the
*minimum* confidence of its parents and is unobserved when either parent is,
so no midpoint is invented from one side. Everything else in the payload is
copied; the layout name records the source so provenance survives.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.shared.python.core.contracts import require

from .skeleton import JOINT_NAMES

DERIVED: dict[str, tuple[str, str]] = {
    "mid_hip": ("left_hip", "right_hip"),
    "neck": ("left_shoulder", "right_shoulder"),
}


def to_reconstruct_layout(payload: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of a ``view-observations`` payload on the reconstruct joint set.

    Precondition: the source layout names every non-derived reconstruct joint
    and both parents of each derived joint. Postcondition: every row has 15
    keypoints in :data:`JOINT_NAMES` order.
    """
    source = list(payload["detector_layout"]["keypoint_names"])
    index = {name: i for i, name in enumerate(source)}
    needed = [n for n in JOINT_NAMES if n not in DERIVED] + [
        p for pair in DERIVED.values() for p in pair
    ]
    missing = sorted({n for n in needed if n not in index})
    require(not missing, "source layout lacks joints", missing)
    frames: list[dict[str, Any]] = []
    for row in payload["frames"]:
        px = np.asarray(row["keypoints_px"], dtype=float)
        conf = np.asarray(row["confidence"], dtype=float)
        out_px = np.zeros((len(JOINT_NAMES), 2))
        out_conf = np.zeros(len(JOINT_NAMES))
        for j, name in enumerate(JOINT_NAMES):
            if name in DERIVED:
                a, b = (index[p] for p in DERIVED[name])
                c = float(min(conf[a], conf[b]))
                out_conf[j] = c
                out_px[j] = 0.5 * (px[a] + px[b]) if c > 0 else 0.0
            else:
                out_px[j] = px[index[name]]
                out_conf[j] = conf[index[name]]
        new = dict(row)
        new["keypoints_px"] = out_px.tolist()
        new["confidence"] = out_conf.tolist()
        frames.append(new)
    out = dict(payload)
    out["frames"] = frames
    out["detector_layout"] = {
        "name": "reconstruct_15",
        "keypoint_names": list(JOINT_NAMES),
    }
    out["provenance"] = {
        **dict(payload.get("provenance", {})),
        "source_layout": payload["detector_layout"].get("name"),
        "derived_joints": {k: list(v) for k, v in DERIVED.items()},
    }
    return out
