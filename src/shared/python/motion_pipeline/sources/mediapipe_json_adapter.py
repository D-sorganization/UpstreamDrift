"""MediaPipe Pose JSON adapter (33-landmark schema).

Matches the dump format produced by ``mediapipe_estimator.py``::

    {
        "schema": "MediaPipe_33",
        "fps": 30.0,
        "frames": [
            {"frame_index": 0, "timestamp": 0.0,
             "landmarks": [{"x": .., "y": .., "z": .., "visibility": ..}, ...]},
            ...
        ]
    }
"""

from __future__ import annotations

import json
from pathlib import Path

from src.shared.python.motion_pipeline.contracts import (
    Calibration,
    Keypoint,
    KeypointFrame,
    KeypointSequence,
)
from src.shared.python.motion_pipeline.sources.base import (
    MocapSourceAdapter,
    SourceMetadata,
)
from src.shared.python.motion_pipeline.sources.registry import register_adapter


def _landmark_to_keypoint(lm: object) -> Keypoint | None:
    """Convert a MediaPipe landmark in any canonical form to a Keypoint.

    Accepts:
      * dict form: ``{"x":..,"y":..,"z":..,"visibility":..}``
      * list/tuple form: ``[x, y, z, visibility, presence?]`` (3 to 5 entries).

    Returns ``None`` when the landmark cannot be parsed.
    """
    if isinstance(lm, dict):
        try:
            x = float(lm.get("x", 0.0))
            y = float(lm.get("y", 0.0))
        except (TypeError, ValueError):
            return None
        z_raw = lm.get("z")
        z = float(z_raw) if z_raw is not None else None
        vis = lm.get("visibility")
        if vis is None:
            vis = lm.get("presence", 1.0)
        try:
            confidence = max(0.0, min(1.0, float(vis)))
        except (TypeError, ValueError):
            confidence = 1.0
        return Keypoint(x=x, y=y, z=z, confidence=confidence)
    if isinstance(lm, (list, tuple)) and len(lm) >= 2:
        try:
            x = float(lm[0])
            y = float(lm[1])
        except (TypeError, ValueError):
            return None
        z = float(lm[2]) if len(lm) >= 3 and lm[2] is not None else None
        if len(lm) >= 4:
            try:
                confidence = max(0.0, min(1.0, float(lm[3])))
            except (TypeError, ValueError):
                confidence = 1.0
        else:
            confidence = 1.0
        return Keypoint(x=x, y=y, z=z, confidence=confidence)
    return None


@register_adapter
class MediaPipeJSONAdapter(MocapSourceAdapter):
    """MediaPipe Pose JSON adapter."""

    format_name = "mediapipe_json"
    file_extensions = (".json",)

    @classmethod
    def supports(cls, path: Path) -> bool:
        p = Path(path)
        if p.suffix.lower() != ".json":
            return False
        if "mediapipe" in p.name.lower():
            return True
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return (
            isinstance(data, dict)
            and data.get("schema") == "MediaPipe_33"
            and isinstance(data.get("frames"), list)
        )

    def metadata(self, path: Path) -> SourceMetadata:
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("MediaPipe JSON must be an object")
        fps = float(data.get("fps", 30.0)) or 30.0
        frames = data.get("frames", []) or []
        return SourceMetadata(
            format_name=self.format_name,
            fps=fps,
            frame_count=len(frames),
            unit_system="normalized",
            keypoint_schema="MediaPipe_33",
        )

    def load(
        self,
        path: Path,
        calibration: Calibration | None = None,
    ) -> KeypointSequence:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"MediaPipe JSON not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "frames" not in data:
            raise ValueError(
                f"MediaPipe JSON {p} missing 'frames'; not a MediaPipe dump"
            )
        fps = float(data.get("fps", 30.0)) or 30.0
        out: list[KeypointFrame] = []
        for idx, raw in enumerate(data["frames"]):
            if not isinstance(raw, dict):
                continue
            landmarks = (
                raw.get("landmarks")
                or raw.get("pose_landmarks")
                or raw.get("pose_world_landmarks")
                or []
            )
            kps: list[Keypoint] = []
            for lm in landmarks:
                kp = _landmark_to_keypoint(lm)
                if kp is not None:
                    kps.append(kp)
            if not kps:
                continue
            t = float(raw.get("timestamp", idx / fps))
            out.append(
                KeypointFrame(
                    timestamp=t,
                    keypoints=kps,
                    schema_name="MediaPipe_33",
                    frame_index=int(raw.get("frame_index", idx)),
                )
            )
        if not out:
            raise ValueError(f"MediaPipe JSON {p} produced no usable frames")
        return KeypointSequence(
            id=f"mediapipe-{p.stem}",
            frames=out,
            calibration=calibration,
            metadata={"source_file": str(p)},
        )
