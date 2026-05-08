"""AlphaPose JSON adapter.

AlphaPose dumps a JSON array where each element looks like::

    {
        "image_id": "0001.jpg",
        "category_id": 1,
        "keypoints": [x1, y1, c1, x2, y2, c2, ...],
        "score": 0.93,
        "idx": [0]
    }

Default schema is COCO-17. Multi-frame ordering is by ``image_id`` if it
parses as an integer-prefixed filename, otherwise by file order.
"""

from __future__ import annotations

import json
import re
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

_NUMERIC = re.compile(r"(\d+)")


def _frame_key(image_id: str, default: int) -> int:
    m = _NUMERIC.search(str(image_id))
    return int(m.group(1)) if m else default


@register_adapter
class AlphaPoseJSONAdapter(MocapSourceAdapter):
    """AlphaPose JSON adapter (COCO-17 by default)."""

    format_name = "alphapose_json"
    file_extensions = (".json",)

    def __init__(self, fps: float = 30.0, schema: str = "COCO_17") -> None:
        self.fps = fps
        self.schema = schema

    @classmethod
    def supports(cls, path: Path) -> bool:
        p = Path(path)
        if p.suffix.lower() != ".json":
            return False
        name = p.name.lower()
        if "alphapose" in name:
            return True
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if not isinstance(data, list) or not data:
            return False
        first = data[0]
        return (
            isinstance(first, dict)
            and "keypoints" in first
            and "image_id" in first
            and "category_id" in first
        )

    def metadata(self, path: Path) -> SourceMetadata:
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("AlphaPose JSON must be a list of detections")
        frame_ids = {item.get("image_id") for item in data if isinstance(item, dict)}
        return SourceMetadata(
            format_name=self.format_name,
            fps=self.fps,
            frame_count=len(frame_ids),
            unit_system="pixels",
            keypoint_schema=self.schema,
        )

    def load(
        self,
        path: Path,
        calibration: Calibration | None = None,
    ) -> KeypointSequence:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"AlphaPose JSON not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not data:
            raise ValueError("AlphaPose JSON must be a non-empty list")

        # Group by frame: first detection per image_id wins (top score)
        by_frame: dict[str, dict] = {}
        for item in data:
            if not isinstance(item, dict) or "keypoints" not in item:
                continue
            fid = str(item.get("image_id", ""))
            score = float(item.get("score", 0.0))
            existing = by_frame.get(fid)
            if existing is None or score > float(existing.get("score", -1.0)):
                by_frame[fid] = item

        ordered = sorted(by_frame.items(), key=lambda kv: _frame_key(kv[0], 0))
        frames: list[KeypointFrame] = []
        for idx, (_fid, item) in enumerate(ordered):
            flat = item["keypoints"]
            keypoints: list[Keypoint] = []
            for i in range(0, len(flat), 3):
                triplet = flat[i : i + 3]
                if len(triplet) < 3:
                    continue
                x = float(triplet[0])
                y = float(triplet[1])
                c = max(0.0, min(1.0, float(triplet[2])))
                keypoints.append(Keypoint(x=x, y=y, confidence=c))
            if not keypoints:
                continue
            frames.append(
                KeypointFrame(
                    timestamp=idx / self.fps,
                    keypoints=keypoints,
                    schema_name=self.schema,  # type: ignore[arg-type]
                    frame_index=idx,
                )
            )
        if not frames:
            raise ValueError(f"AlphaPose JSON {p} produced no usable frames")
        return KeypointSequence(
            id=f"alphapose-{p.stem}",
            frames=frames,
            calibration=calibration,
            metadata={"source_file": str(p)},
        )
