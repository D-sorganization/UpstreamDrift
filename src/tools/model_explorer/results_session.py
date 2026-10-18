"""Qt-free results playback state for the Model Explorer (#12878, #12915).

Holds one loaded results archive, the scrub position and the single export
path. The widget delegates here, so playback and export can be tested without
a display and the same code serves every engine. Two kinds of run load
through :meth:`ResultsSession.load`:

* joint-space runs (``q`` on the shared full-body specification, see
  :mod:`.pose_results`), played on the shared-appearance golfer, and
* marker archives (one frame source, one writer: ``motion_matching.export``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.export import export_video
from src.shared.python.motion_matching.video_frame_source import (
    MarkerFrameSource,
    load_marker_frame_source,
)
from src.tools.model_explorer.pose_results import (
    PoseFrameSource,
    engine_title,
    is_pose_results,
    load_pose_frame_source,
)

__all__ = ["ResultsSession", "detect_engine_from_archive"]

_VIDEO_SUFFIXES = (".mp4", ".gif")
FrameSource = MarkerFrameSource | PoseFrameSource
_KNOWN_ENGINES = ("mujoco", "drake", "pinocchio", "opensim", "myosuite", "simscape")


def detect_engine_from_archive(path: str | Path) -> str | None:
    """Extract engine name from an archive's contents, receipts, or metadata."""
    target = Path(path)
    if not target.is_file():
        return None

    # 1. Inspect npz keys
    if target.suffix.lower() == ".npz":
        try:
            with np.load(target, allow_pickle=False) as data:
                files = set(data.files)
                if "engine" in files:
                    val = data["engine"]
                    raw = (
                        str(val.item() if getattr(val, "ndim", 0) == 0 else val[0])
                        .strip()
                        .lower()
                    )
                    if raw and raw not in ("unknown", "engine"):
                        return raw
                if "manifest_json" in files:
                    try:
                        manifest = json.loads(str(data["manifest_json"]))
                        eng = manifest.get("engine") or manifest.get(
                            "metadata", {}
                        ).get("engine")
                        if eng and str(eng).strip().lower() not in (
                            "unknown",
                            "engine",
                        ):
                            return str(eng).strip().lower()
                    except (ValueError, TypeError):
                        pass
                if "manifest" in files:
                    try:
                        manifest = json.loads(str(data["manifest"]))
                        eng = manifest.get("engine") or manifest.get("reference_engine")
                        if eng and str(eng).strip().lower() not in (
                            "unknown",
                            "engine",
                        ):
                            return str(eng).strip().lower()
                    except (ValueError, TypeError):
                        pass
                if "backend" in files:
                    val = data["backend"]
                    raw = (
                        str(val.item() if getattr(val, "ndim", 0) == 0 else val[0])
                        .strip()
                        .lower()
                    )
                    if raw and raw not in ("unknown", "engine"):
                        return raw
        except (OSError, ValueError):
            pass

    # 2. Sibling receipt JSON
    for receipt_candidate in (
        target.with_suffix(".json"),
        target.parent / "receipt.json",
    ):
        if receipt_candidate.is_file():
            try:
                receipt = json.loads(receipt_candidate.read_text(encoding="utf-8"))
                eng = receipt.get("engine") or receipt.get("backend")
                if eng and str(eng).strip().lower() not in ("unknown", "engine"):
                    return str(eng).strip().lower()
            except (ValueError, OSError):
                pass

    # 3. Known engine names in stem or parent folder
    for part in target.stem.lower().replace("-", "_").split("_"):
        if part in _KNOWN_ENGINES:
            return part
    for part in target.parent.name.lower().replace("-", "_").split("_"):
        if part in _KNOWN_ENGINES:
            return part

    return None


class ResultsSession:
    """A loaded results archive plus its playback cursor.

    Invariant: ``0 <= current_frame < frame_count`` and ``frame_count >= 2``.
    """

    def __init__(self, path: Path, source: FrameSource, engine: str = "engine") -> None:
        if not engine.strip():
            raise ValueError("engine must be a non-empty name")
        self.path = path
        self.engine = engine
        self._source = source
        self._current_frame = 0
        self._playback: Any | None = None

    @classmethod
    def load(cls, path: str | Path, engine: str | None = None) -> ResultsSession:
        """Load ``path``; raise ``FileNotFoundError`` or ``VideoSourceError``.

        Joint-space runs take their engine from the run itself; marker
        archives read engine metadata if present or fall back to ``engine``.
        """
        resolved = Path(path)
        if is_pose_results(resolved):
            pose = load_pose_frame_source(resolved)
            return cls(resolved, pose, pose.engine)
        detected = detect_engine_from_archive(resolved)
        chosen_engine = (
            engine
            if (engine is not None and engine != "engine")
            else (detected or "engine")
        )
        return cls(resolved, load_marker_frame_source(resolved), chosen_engine)

    @property
    def markers(self) -> NDArray[np.float64] | None:
        """Current marker trajectories across all frames, shape (n, k, 3), or None."""
        if hasattr(self._source, "markers"):
            return self._source.markers
        return None

    @property
    def pose_source(self) -> PoseFrameSource | None:
        """The joint-space run, or ``None`` for a marker archive."""
        return self._source if isinstance(self._source, PoseFrameSource) else None

    @property
    def title(self) -> str:
        """What is being shown, e.g. ``Playback of Drake result (...)``."""
        pose = self.pose_source
        if pose is not None:
            return pose.playback_label
        return f"{engine_title(self.engine)} marker overlay"

    @property
    def frame_count(self) -> int:
        return self._source.frame_count

    @property
    def current_frame(self) -> int:
        return self._current_frame

    @property
    def default_video_name(self) -> str:
        """Suggested export file name: the archive stem with ``.mp4``."""
        return f"{self.path.stem}.mp4"

    @property
    def current_time_s(self) -> float:
        return float(self._source.time_s[self._current_frame])

    def scrub(self, index: int) -> None:
        """Move the playback cursor to ``index``.

        Raises:
            ValueError: ``index`` is outside ``[0, frame_count)``.
        """
        if not 0 <= index < self.frame_count:
            raise ValueError(
                f"frame {index} is outside the results (0..{self.frame_count - 1})"
            )
        self._current_frame = index

    def playback(self) -> Any:
        """The shared :class:`GolferPlayback` of a joint-space run (lazy).

        Raises ``ValueError`` for a marker archive (nothing to pose).
        """
        pose = self.pose_source
        if pose is None:
            raise ValueError("marker archives have no model pose to play back")
        if self._playback is None:
            from src.tools.model_explorer.golfer_playback import GolferPlayback

            self._playback = GolferPlayback(pose)
        return self._playback

    def export_video(self, out_path: str | Path, *, stride: int = 5) -> Path:
        """Write the whole playback to ``out_path`` (``.mp4`` or ``.gif``).

        Joint-space runs are filmed on the golfer in real time; marker
        archives go through the marker-overlay exporter.

        Raises:
            ValueError: the suffix is neither ``.mp4`` nor ``.gif``.
        """
        target = Path(out_path)
        if target.suffix.lower() not in _VIDEO_SUFFIXES:
            raise ValueError(
                f"Unsupported video format '{target.suffix}'; use .mp4 or .gif"
            )
        if isinstance(self._source, PoseFrameSource):
            return Path(self.playback().export_video(target))
        return export_video(self._source, self.engine, target, stride=stride)

    def close(self) -> None:
        """Release the renderer, if one was created; idempotent."""
        playback, self._playback = self._playback, None
        if playback is not None:
            playback.close()
