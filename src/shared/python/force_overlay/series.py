"""Time-indexed immutable series of ForceTorqueFrames (ADR-0052, #11286).

Provides:
- Strictly increasing times_s and single-engine enforcement.
- Pure linear interpolation at arbitrary sample times with gap thresholding.
- Lossless NPZ serialization with allow_pickle=False and boolean mask arrays.
- JSON-safe dict serialization.
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import io
import math
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from src.shared.python.body_part_viz import AxialLoadFrame
from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)

_ALLOWED_SERIES_KEYS = frozenset({"schema_version", "engine", "frames"})


def _interpolate_wrenches(
    f0: ForceTorqueFrame, f1: ForceTorqueFrame, alpha: float
) -> tuple[OverlayWrench, ...]:
    """Linearly interpolate matching wrenches between two frames."""
    w0_map = {w.label: w for w in f0.wrenches}
    w1_map = {w.label: w for w in f1.wrenches}
    interp: list[OverlayWrench] = []

    for label, w0 in w0_map.items():
        if label not in w1_map:
            continue
        w1 = w1_map[label]

        # Interpolate point_m
        p = (
            float((1.0 - alpha) * w0.point_m[0] + alpha * w1.point_m[0]),
            float((1.0 - alpha) * w0.point_m[1] + alpha * w1.point_m[1]),
            float((1.0 - alpha) * w0.point_m[2] + alpha * w1.point_m[2]),
        )

        # Interpolate force_n
        fn: tuple[float, float, float] | None = None
        if w0.force_n is not None and w1.force_n is not None:
            fn = (
                float((1.0 - alpha) * w0.force_n[0] + alpha * w1.force_n[0]),
                float((1.0 - alpha) * w0.force_n[1] + alpha * w1.force_n[1]),
                float((1.0 - alpha) * w0.force_n[2] + alpha * w1.force_n[2]),
            )

        # Interpolate torque_nm
        tnm: tuple[float, float, float] | None = None
        if w0.torque_nm is not None and w1.torque_nm is not None:
            tnm = (
                float((1.0 - alpha) * w0.torque_nm[0] + alpha * w1.torque_nm[0]),
                float((1.0 - alpha) * w0.torque_nm[1] + alpha * w1.torque_nm[1]),
                float((1.0 - alpha) * w0.torque_nm[2] + alpha * w1.torque_nm[2]),
            )

        # If both halves become None, wrench cannot be represented -> omit
        if fn is None and tnm is None:
            continue

        interp.append(
            OverlayWrench(
                kind=w0.kind,
                label=label,
                body=w0.body,
                point_m=p,
                force_n=fn,
                torque_nm=tnm,
                source=w0.source,
            )
        )
    return tuple(interp)


def _interpolate_axial_loads(
    f0: ForceTorqueFrame, f1: ForceTorqueFrame, alpha: float, t: float
) -> AxialLoadFrame | None:
    """Linearly interpolate axial load frames if both are present."""
    al0_frame = f0.axial_loads
    al1_frame = f1.axial_loads
    if al0_frame is None or al1_frame is None:
        return None
    al0 = al0_frame.values_n
    al1 = al1_frame.values_n
    all_segs = sorted(set(al0.keys()) | set(al1.keys()))
    interp_values: dict[str, float | None] = {}
    for seg in all_segs:
        v0 = al0.get(seg)
        v1 = al1.get(seg)
        if v0 is not None and v1 is not None:
            interp_values[seg] = float((1.0 - alpha) * v0 + alpha * v1)
        else:
            interp_values[seg] = None
    return AxialLoadFrame(
        time_s=t,
        values_n=interp_values,
        source=al0_frame.source,
    )


@dataclass(frozen=True)
class ForceTorqueSeries:
    """Immutable, time-indexed sequence of ForceTorqueFrames from a single engine."""

    frames: tuple[ForceTorqueFrame, ...] = ()
    engine: str = ""

    def __post_init__(self) -> None:
        frames_tuple = tuple(self.frames)
        if frames_tuple:
            eng = frames_tuple[0].engine
            for i, f in enumerate(frames_tuple):
                if not isinstance(f, ForceTorqueFrame):
                    raise TypeError(
                        f"All frames must be ForceTorqueFrame instances, got {type(f)}"
                    )
                if f.engine != eng:
                    raise ValueError(
                        f"All frames must share the same engine, got '{eng}' and '{f.engine}'"
                    )
                if i > 0 and f.time_s <= frames_tuple[i - 1].time_s:
                    raise ValueError(
                        f"times_s must be strictly increasing, got "
                        f"{frames_tuple[i - 1].time_s} then {f.time_s}"
                    )
            object.__setattr__(self, "engine", eng)
        else:
            if not isinstance(self.engine, str):
                raise ValueError("engine must be a string")
        object.__setattr__(self, "frames", frames_tuple)

    @property
    def times_s(self) -> tuple[float, ...]:
        """Return tuple of sample timestamps."""
        return tuple(f.time_s for f in self.frames)

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self):
        return iter(self.frames)

    def __getitem__(self, index: int) -> ForceTorqueFrame:
        return self.frames[index]

    def frame_at(self, t: float, max_gap_s: float = 0.1) -> ForceTorqueFrame | None:
        """Sample or linearly interpolate a frame at time t.

        Rules:
        - Exact time match returns the exact frame instance.
        - Between neighbours with (t1 - t0) <= max_gap_s:
          - Linearly interpolates point_m.
          - Each half (force_n, torque_nm) present in both neighbours is linearly interpolated.
          - A half present in only one neighbour becomes None.
          - Labels present in only one neighbour are omitted.
          - Wrenches whose force and torque both become None are omitted.
          - Axial loads interpolate per segment; None if either neighbour is None.
        - Out of range or gap > max_gap_s returns None.
        """
        if not math.isfinite(t):
            raise ValueError("t must be a finite number")
        if not self.frames:
            return None

        t0 = self.frames[0].time_s
        tn = self.frames[-1].time_s
        if t < t0 - 1e-12 or t > tn + 1e-12:
            return None

        # Check for exact match within numerical tolerance
        for f in self.frames:
            if math.isclose(f.time_s, t, rel_tol=0, abs_tol=1e-12):
                return f

        times = [f.time_s for f in self.frames]
        idx = bisect.bisect_right(times, t)
        if idx == 0 or idx >= len(self.frames):
            return None

        f0 = self.frames[idx - 1]
        f1 = self.frames[idx]

        gap = f1.time_s - f0.time_s
        if gap > max_gap_s:
            return None

        alpha = (t - f0.time_s) / gap
        interp_wrenches = _interpolate_wrenches(f0, f1, alpha)
        interp_axial_loads = _interpolate_axial_loads(f0, f1, alpha, t)

        return ForceTorqueFrame(
            time_s=t,
            engine=self.engine,
            wrenches=interp_wrenches,
            axial_loads=interp_axial_loads,
            world_frame=f0.world_frame,
            units=f0.units,
        )

    def to_npz(self, path_or_file: str | Path | BinaryIO) -> None:
        """Serialize series to NPZ format without pickling (allow_pickle=False safe)."""
        n_frames = len(self.frames)
        times_s = np.array([f.time_s for f in self.frames], dtype=np.float64)

        # Collect unique wrench labels
        all_labels: list[str] = []
        seen_labels: set[str] = set()
        for f in self.frames:
            for w in f.wrenches:
                if w.label not in seen_labels:
                    seen_labels.add(w.label)
                    all_labels.append(w.label)

        n_wrenches = len(all_labels)
        label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}

        wrench_labels = np.array(all_labels, dtype=str)
        wrench_present = np.zeros((n_frames, n_wrenches), dtype=bool)
        wrench_kinds = np.zeros((n_frames, n_wrenches), dtype="U32")
        wrench_bodies = np.zeros((n_frames, n_wrenches), dtype="U64")
        wrench_sources = np.zeros((n_frames, n_wrenches), dtype="U64")
        point_m = np.zeros((n_frames, n_wrenches, 3), dtype=np.float64)
        force_n = np.zeros((n_frames, n_wrenches, 3), dtype=np.float64)
        force_mask = np.zeros((n_frames, n_wrenches), dtype=bool)
        torque_nm = np.zeros((n_frames, n_wrenches, 3), dtype=np.float64)
        torque_mask = np.zeros((n_frames, n_wrenches), dtype=bool)

        for i, f in enumerate(self.frames):
            for w in f.wrenches:
                j = label_to_idx[w.label]
                wrench_present[i, j] = True
                wrench_kinds[i, j] = w.kind.value
                wrench_bodies[i, j] = w.body
                wrench_sources[i, j] = w.source
                point_m[i, j] = w.point_m
                if w.force_n is not None:
                    force_n[i, j] = w.force_n
                    force_mask[i, j] = True
                if w.torque_nm is not None:
                    torque_nm[i, j] = w.torque_nm
                    torque_mask[i, j] = True

        # Axial loads
        has_axial_loads = np.array(
            [f.axial_loads is not None for f in self.frames], dtype=bool
        )
        all_segments: list[str] = []
        seen_segs: set[str] = set()
        for f in self.frames:
            axial = f.axial_loads
            if axial is not None:
                for seg in axial.values_n:
                    if seg not in seen_segs:
                        seen_segs.add(seg)
                        all_segments.append(seg)

        n_segments = len(all_segments)
        seg_to_idx = {seg: s for s, seg in enumerate(all_segments)}
        axial_segments = np.array(all_segments, dtype=str)
        axial_values = np.zeros((n_frames, n_segments), dtype=np.float64)
        axial_mask = np.zeros((n_frames, n_segments), dtype=bool)
        axial_sources = np.zeros((n_frames,), dtype="U64")

        for i, f in enumerate(self.frames):
            axial = f.axial_loads
            if axial is not None:
                axial_sources[i] = axial.source
                vals = axial.values_n
                for seg, val in vals.items():
                    s = seg_to_idx[seg]
                    if val is not None:
                        axial_values[i, s] = val
                        axial_mask[i, s] = True

        world_frame = self.frames[0].world_frame if self.frames else "world_Zup"

        np.savez(
            path_or_file,
            times_s=times_s,
            engine=np.array(self.engine, dtype=str),
            world_frame=np.array(world_frame, dtype=str),
            wrench_labels=wrench_labels,
            wrench_present=wrench_present,
            wrench_kinds=wrench_kinds,
            wrench_bodies=wrench_bodies,
            wrench_sources=wrench_sources,
            point_m=point_m,
            force_n=force_n,
            force_mask=force_mask,
            torque_nm=torque_nm,
            torque_mask=torque_mask,
            has_axial_loads=has_axial_loads,
            axial_segments=axial_segments,
            axial_values=axial_values,
            axial_mask=axial_mask,
            axial_sources=axial_sources,
        )

    @classmethod
    def from_npz(cls, path_or_file: str | Path | BinaryIO) -> ForceTorqueSeries:
        """Deserialize series from NPZ format with allow_pickle=False."""
        with np.load(path_or_file, allow_pickle=False) as data:
            times_s = data["times_s"]
            engine = str(data["engine"])
            world_frame = str(data["world_frame"])
            wrench_labels = data["wrench_labels"]
            wrench_present = data["wrench_present"]
            wrench_kinds = data["wrench_kinds"]
            wrench_bodies = data["wrench_bodies"]
            wrench_sources = data["wrench_sources"]
            point_m = data["point_m"]
            force_n = data["force_n"]
            force_mask = data["force_mask"]
            torque_nm = data["torque_nm"]
            torque_mask = data["torque_mask"]
            has_axial_loads = data["has_axial_loads"]
            axial_segments = data["axial_segments"]
            axial_values = data["axial_values"]
            axial_mask = data["axial_mask"]
            axial_sources = data["axial_sources"]

            n_frames = len(times_s)
            n_wrenches = len(wrench_labels)
            frames: list[ForceTorqueFrame] = []

            for i in range(n_frames):
                wrenches: list[OverlayWrench] = []
                for j in range(n_wrenches):
                    if not wrench_present[i, j]:
                        continue
                    w_kind = WrenchKind(str(wrench_kinds[i, j]))
                    w_label = str(wrench_labels[j])
                    w_body = str(wrench_bodies[i, j])
                    w_source = str(wrench_sources[i, j])
                    w_point = (
                        float(point_m[i, j, 0]),
                        float(point_m[i, j, 1]),
                        float(point_m[i, j, 2]),
                    )
                    w_force: tuple[float, float, float] | None = None
                    if force_mask[i, j]:
                        w_force = (
                            float(force_n[i, j, 0]),
                            float(force_n[i, j, 1]),
                            float(force_n[i, j, 2]),
                        )
                    w_torque: tuple[float, float, float] | None = None
                    if torque_mask[i, j]:
                        w_torque = (
                            float(torque_nm[i, j, 0]),
                            float(torque_nm[i, j, 1]),
                            float(torque_nm[i, j, 2]),
                        )

                    wrenches.append(
                        OverlayWrench(
                            kind=w_kind,
                            label=w_label,
                            body=w_body,
                            point_m=w_point,
                            force_n=w_force,
                            torque_nm=w_torque,
                            source=w_source,
                        )
                    )

                axial_frame: AxialLoadFrame | None = None
                if has_axial_loads[i]:
                    values_n: dict[str, float | None] = {}
                    for s, seg_name in enumerate(axial_segments):
                        seg_str = str(seg_name)
                        if axial_mask[i, s]:
                            values_n[seg_str] = float(axial_values[i, s])
                        else:
                            values_n[seg_str] = None
                    axial_frame = AxialLoadFrame(
                        time_s=float(times_s[i]),
                        values_n=values_n,
                        source=str(axial_sources[i]),
                    )

                frames.append(
                    ForceTorqueFrame(
                        time_s=float(times_s[i]),
                        engine=engine,
                        wrenches=tuple(wrenches),
                        axial_loads=axial_frame,
                        world_frame=world_frame,
                    )
                )

        return cls(frames=tuple(frames), engine=engine)

    def to_dict(self) -> dict[str, Any]:
        """Serialize series to JSON-safe dictionary."""
        return {
            "schema_version": "force-torque-series-v1",
            "engine": self.engine,
            "frames": [f.to_dict() for f in self.frames],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ForceTorqueSeries:
        """Deserialize series from dictionary, rejecting unknown keys."""
        extra = set(data.keys()) - _ALLOWED_SERIES_KEYS
        if extra:
            raise ValueError(f"Unknown keys in series data: {extra}")

        schema_version = data.get("schema_version")
        if schema_version != "force-torque-series-v1":
            raise ValueError(
                f"Invalid schema_version: expected 'force-torque-series-v1', got '{schema_version}'"
            )

        frames = tuple(ForceTorqueFrame.from_dict(f) for f in data.get("frames", ()))
        return cls(frames=frames, engine=str(data.get("engine", "")))
