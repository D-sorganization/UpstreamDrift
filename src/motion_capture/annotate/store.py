"""Sparse manual landmark annotations for one view (#9798).

``annotations/<view>.json`` (``manual-annotations/1.0.0``)::

    {"schema_version": ..., "view": "face_on", "width": 1920, "height": 1200,
     "fps": 60.0, "joints": [...JOINT_NAMES...],
     "frames": {"12": {"left_wrist": {"x_px": 811.5, "y_px": 640.0, "note": null}}},
     "skipped": {"12": ["right_wrist"]},
     "provenance": {... "parameters": {"annotator": "..."} ...}}

A point is a click the user was confident about; a skip records that the
joint was occluded on that frame. Everything else is simply absent: sparse
by construction. Qt-free.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.shared.python.core.contracts import require

from ..provenance import stamp, write_json
from ..reconstruct.skeleton import JOINT_NAMES

SCHEMA_VERSION = "manual-annotations/1.0.0"
ANNOTATIONS_DIR = "annotations"


@dataclass(frozen=True)
class Point:
    x_px: float
    y_px: float
    note: str | None = None
    interpolated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"x_px": self.x_px, "y_px": self.y_px, "note": self.note}


class AnnotationSet:
    """Points and skips per (frame, joint) for one view."""

    def __init__(
        self,
        view: str,
        width: int,
        height: int,
        fps: float,
        joints: Iterable[str] = JOINT_NAMES,
        annotator: str = "unknown",
        base_set: str | None = None,
    ) -> None:
        require(view.strip() != "", "view must be named")
        require(
            width > 0 and height > 0, "image size must be positive", (width, height)
        )
        require(fps > 0, "fps must be positive", fps)
        self.view, self.width, self.height, self.fps = view, width, height, fps
        self.joints: tuple[str, ...] = tuple(joints)
        require(len(self.joints) >= 1, "at least one joint")
        self.annotator = annotator
        #: Observation set this file corrects (#9803); ``None`` for fresh tracking.
        self.base_set = base_set
        self._points: dict[int, dict[str, Point]] = {}
        self._skipped: dict[int, set[str]] = {}
        self.created_utc: str | None = None

    # -- editing -------------------------------------------------------------
    def _check(self, frame: int, joint: str) -> None:
        require(frame >= 0, "frame must be >= 0", frame)
        require(joint in self.joints, "unknown joint", joint)

    def set_point(
        self, frame: int, joint: str, x_px: float, y_px: float, note: str | None = None
    ) -> Point:
        """Record a click; clears a skip on the same (frame, joint).

        Precondition: the point lies inside the image.
        """
        self._check(frame, joint)
        require(0 <= x_px < self.width and 0 <= y_px < self.height, "point off image")
        point = Point(float(x_px), float(y_px), note)
        self._points.setdefault(frame, {})[joint] = point
        self._skipped.get(frame, set()).discard(joint)
        return point

    def clear_point(self, frame: int, joint: str) -> None:
        self._check(frame, joint)
        self._points.get(frame, {}).pop(joint, None)
        if frame in self._points and not self._points[frame]:
            del self._points[frame]

    def skip(self, frame: int, joint: str) -> None:
        """Mark the joint occluded on the frame; removes any point."""
        self.clear_point(frame, joint)
        self._skipped.setdefault(frame, set()).add(joint)

    def unskip(self, frame: int, joint: str) -> None:
        self._check(frame, joint)
        self._skipped.get(frame, set()).discard(joint)

    # -- queries -------------------------------------------------------------
    def point(self, frame: int, joint: str) -> Point | None:
        return self._points.get(frame, {}).get(joint)

    def is_skipped(self, frame: int, joint: str) -> bool:
        return joint in self._skipped.get(frame, set())

    def has_entry(self, frame: int, joint: str) -> bool:
        return self.point(frame, joint) is not None or self.is_skipped(frame, joint)

    def annotated_frames(self) -> list[int]:
        return sorted(self._points)

    def points_at(self, frame: int) -> dict[str, Point]:
        return dict(self._points.get(frame, {}))

    def skipped_at(self, frame: int) -> set[str]:
        return set(self._skipped.get(frame, set()))

    def count(self) -> int:
        return sum(len(p) for p in self._points.values())

    def coverage(self) -> dict[str, dict[str, Any]]:
        """Per joint: annotated and skipped counts, first and last frame."""
        out: dict[str, dict[str, Any]] = {}
        for joint in self.joints:
            frames = sorted(f for f, pts in self._points.items() if joint in pts)
            skipped = sum(1 for s in self._skipped.values() if joint in s)
            out[joint] = {
                "annotated": len(frames),
                "skipped": skipped,
                "first": frames[0] if frames else None,
                "last": frames[-1] if frames else None,
            }
        return out

    def next_missing(self, frame: int, joint: str, stride: int = 1) -> int:
        """First frame ``>= frame`` on the stride with neither point nor skip."""
        require(stride >= 1, "stride must be >= 1", stride)
        f = frame
        while self.has_entry(f, joint):
            f += stride
        return f

    def interpolate(self, frame: int, joint: str) -> Point | None:
        """Linear guess between the nearest annotated frames on either side.

        Returns the stored point when the frame is annotated; ``None`` when
        no neighbour exists on one side (no extrapolation).
        """
        self._check(frame, joint)
        exact = self.point(frame, joint)
        if exact is not None:
            return exact
        frames = [f for f, pts in self._points.items() if joint in pts]
        before = [f for f in frames if f < frame]
        after = [f for f in frames if f > frame]
        if not before or not after:
            return None
        f0, f1 = max(before), min(after)
        p0, p1 = self._points[f0][joint], self._points[f1][joint]
        s = (frame - f0) / (f1 - f0)
        return Point(
            p0.x_px + s * (p1.x_px - p0.x_px),
            p0.y_px + s * (p1.y_px - p0.y_px),
            None,
            interpolated=True,
        )

    # -- persistence ---------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "view": self.view,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "joints": list(self.joints),
            "base_set": self.base_set,
            "frames": {
                str(f): {j: p.to_dict() for j, p in sorted(pts.items())}
                for f, pts in sorted(self._points.items())
            },
            "skipped": {
                str(f): sorted(s) for f, s in sorted(self._skipped.items()) if s
            },
        }

    def save(self, path: Path, base: Path | None = None) -> Path:
        """Write the file with provenance; ``updated_utc`` is the stamp time.

        Postcondition: :meth:`load` of the file equals this set.
        """
        payload = stamp(
            self.to_dict(),
            schema_version=SCHEMA_VERSION,
            module=__name__,
            parameters={
                "annotator": self.annotator,
                "points": self.count(),
                "created_utc": self.created_utc,
            },
            base=base,
        )
        created = self.created_utc or payload["provenance"]["created_utc"]
        payload["provenance"]["parameters"]["created_utc"] = created
        payload["provenance"]["updated_utc"] = payload["provenance"]["created_utc"]
        self.created_utc = created
        return write_json(path, payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AnnotationSet:
        require(
            payload.get("schema_version") == SCHEMA_VERSION,
            "unknown annotation schema",
            payload.get("schema_version"),
        )
        prov = payload.get("provenance") or {}
        params = prov.get("parameters") or {}
        out = cls(
            str(payload["view"]),
            int(payload["width"]),
            int(payload["height"]),
            float(payload["fps"]),
            payload.get("joints", JOINT_NAMES),
            annotator=str(params.get("annotator", "unknown")),
            base_set=payload.get("base_set"),
        )
        out.created_utc = params.get("created_utc") or prov.get("created_utc")
        for frame, pts in payload.get("frames", {}).items():
            for joint, p in pts.items():
                out.set_point(int(frame), joint, p["x_px"], p["y_px"], p.get("note"))
        for frame, joints in payload.get("skipped", {}).items():
            for joint in joints:
                out.skip(int(frame), joint)
        return out

    @classmethod
    def load(cls, path: Path) -> AnnotationSet:
        require(path.is_file(), "annotation file must exist", str(path))
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AnnotationSet) and self.to_dict() == other.to_dict()

    __hash__ = None  # type: ignore[assignment]


def annotation_path(session: Path, view: str) -> Path:
    return session / ANNOTATIONS_DIR / f"{view}.json"
