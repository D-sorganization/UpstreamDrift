"""HRNet (COCO-17) JSON adapter.

HRNet research dumps follow either the AlphaPose-like COCO list shape or a
``{"frames": [{"frame_index": 0, "keypoints": [...]}, ...]}`` wrapper. We
support both. Schema defaults to COCO-17.
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
class HRNetJSONAdapter(MocapSourceAdapter):
    """HRNet JSON adapter."""

    format_name = "hrnet_json"
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
        if "hrnet" in name:
            return True
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if isinstance(data, dict) and data.get("schema") == "HRNet":
            return True
        if isinstance(data, dict) and "frames" in data:
            frames = data["frames"]
            if isinstance(frames, list) and frames and isinstance(frames[0], dict):
                return "keypoints" in frames[0] and "frame_index" in frames[0]
        return False

    def _normalise(self, data: object) -> list[dict]:
        if isinstance(data, dict) and "frames" in data:
            frames = data["frames"]
            if isinstance(frames, list):
                return [f for f in frames if isinstance(f, dict)]
        if isinstance(data, list):
            return [f for f in data if isinstance(f, dict)]
        raise ValueError("HRNet JSON must be a list or {'frames': [...]} object")

    def metadata(self, path: Path) -> SourceMetadata:
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        frames = self._normalise(data)
        return SourceMetadata(
            format_name=self.format_name,
            fps=self.fps,
            frame_count=len(frames),
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
            raise FileNotFoundError(f"HRNet JSON not found: {p}")
        raw = self._normalise(json.loads(p.read_text(encoding="utf-8")))
        if not raw:
            raise ValueError(f"HRNet JSON {p} has no frames")

        frames: list[KeypointFrame] = []
        for idx, item in enumerate(sorted(raw, key=lambda d: d.get("frame_index", 0))):
            flat = item.get("keypoints", [])
            kps: list[Keypoint] = []
            for i in range(0, len(flat), 3):
                triplet = flat[i : i + 3]
                if len(triplet) < 3:
                    continue
                kps.append(
                    Keypoint(
                        x=float(triplet[0]),
                        y=float(triplet[1]),
                        confidence=max(0.0, min(1.0, float(triplet[2]))),
                    )
                )
            if not kps:
                continue
            t = float(item.get("timestamp", idx / self.fps))
            frames.append(
                KeypointFrame(
                    timestamp=t,
                    keypoints=kps,
                    schema_name=self.schema,  # type: ignore[arg-type]
                    frame_index=int(item.get("frame_index", idx)),
                )
            )
        if not frames:
            raise ValueError(f"HRNet JSON {p} produced no usable frames")
        return KeypointSequence(
            id=f"hrnet-{p.stem}",
            frames=frames,
            calibration=calibration,
            metadata={"source_file": str(p)},
        )
