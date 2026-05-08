"""OpenPose JSON adapter (BODY_25 keypoints).

Two on-disk variants are supported:

1. **Per-frame files** (``*_keypoints.json``) - one JSON object with a
   ``people`` list and a ``version`` field. The adapter loads the file and
   reports a single-frame :class:`KeypointSequence`.
2. **Concatenated array** - a JSON array whose entries are per-frame
   objects of the same shape, useful for pre-aggregated dumps.

Multi-person frames default to the first person; pass ``person_index`` to
the constructor to select another.
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


@register_adapter
class OpenPoseJSONAdapter(MocapSourceAdapter):
    """OpenPose ``*_keypoints.json`` reader."""

    format_name = "openpose_json"
    file_extensions = (".json",)

    def __init__(self, person_index: int = 0, fps: float = 30.0) -> None:
        self.person_index = person_index
        self.fps = fps

    @classmethod
    def supports(cls, path: Path) -> bool:
        p = Path(path)
        if p.suffix.lower() != ".json":
            return False
        # Filename heuristic: OpenPose names files <stem>_keypoints.json
        name = p.name.lower()
        if "_keypoints" in name or "openpose" in name:
            return True
        # Fallback: peek at content
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if isinstance(data, dict) and "people" in data and "version" in data:
            return True
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return "people" in data[0]
        return False

    def _frames_from_payload(self, data: object) -> list[dict]:
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        raise ValueError("OpenPose JSON must be an object or list of objects")

    def metadata(self, path: Path) -> SourceMetadata:
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        frames = self._frames_from_payload(data)
        return SourceMetadata(
            format_name=self.format_name,
            fps=self.fps,
            frame_count=len(frames),
            unit_system="pixels",
            keypoint_schema="BODY_25",
            notes=f"person_index={self.person_index}",
        )

    def load(
        self,
        path: Path,
        calibration: Calibration | None = None,
    ) -> KeypointSequence:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"OpenPose JSON not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        raw_frames = self._frames_from_payload(data)
        if not raw_frames:
            raise ValueError(f"OpenPose JSON {p} has no frames")

        kp_frames: list[KeypointFrame] = []
        for idx, frame in enumerate(raw_frames):
            people = frame.get("people", [])
            if not people:
                # Empty frame -- emit a 1-keypoint zero placeholder so the
                # frame is not lost, with confidence 0 (occluded entire body).
                kp_frames.append(
                    KeypointFrame(
                        timestamp=idx / self.fps,
                        keypoints=[Keypoint(x=0.0, y=0.0, confidence=0.0)],
                        schema_name="BODY_25",
                        frame_index=idx,
                    )
                )
                continue
            person = people[min(self.person_index, len(people) - 1)]
            flat = person.get("pose_keypoints_2d") or person.get("pose_keypoints")
            if not flat:
                continue
            keypoints: list[Keypoint] = []
            for i in range(0, len(flat), 3):
                triplet = flat[i : i + 3]
                if len(triplet) < 3:
                    continue
                x, y, c = float(triplet[0]), float(triplet[1]), float(triplet[2])
                keypoints.append(Keypoint(x=x, y=y, confidence=max(0.0, min(1.0, c))))
            if keypoints:
                kp_frames.append(
                    KeypointFrame(
                        timestamp=idx / self.fps,
                        keypoints=keypoints,
                        schema_name="BODY_25",
                        frame_index=idx,
                    )
                )
        if not kp_frames:
            raise ValueError(f"OpenPose JSON {p} produced no usable frames")

        return KeypointSequence(
            id=f"openpose-{p.stem}",
            frames=kp_frames,
            calibration=calibration,
            metadata={"source_file": str(p), "person_index": self.person_index},
        )
