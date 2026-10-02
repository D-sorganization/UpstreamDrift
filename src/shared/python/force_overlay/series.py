"""Time-series interpolation and storage for force/torque overlay frames (ADR-0052, #11286)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import io
import json
import math
from pathlib import Path
from typing import Any, Final

import numpy as np

from src.shared.python.body_part_viz.axial_loads import AxialLoadFrame
from .contracts import ForceTorqueFrame, OverlayWrench


@dataclass(frozen=True)
class ForceTorqueSeries:
    """Time-indexed sequence of ForceTorqueFrames with linear interpolation."""

    engine: str
    frames: tuple[ForceTorqueFrame, ...] = ()

    SCHEMA_VERSION: Final[str] = "force-torque-series-v1"

    def __post_init__(self) -> None:
        if not self.engine or not isinstance(self.engine, str):
            raise ValueError("engine must be a non-empty string")

        f_tuple = tuple(self.frames)
        for i, f in enumerate(f_tuple):
            if not isinstance(f, ForceTorqueFrame):
                raise TypeError(
                    f"All frames must be ForceTorqueFrame, got {type(f).__name__}"
                )
            if f.engine != self.engine:
                raise ValueError(
                    f"Frame at index {i} engine mismatch: expected {self.engine!r}, got {f.engine!r}"
                )
            if i > 0 and f.time_s <= f_tuple[i - 1].time_s:
                raise ValueError(
                    f"Frame times must be strictly increasing: frame[{i - 1}]={f_tuple[i - 1].time_s} "
                    f">= frame[{i}]={f.time_s}"
                )
        object.__setattr__(self, "frames", f_tuple)

    def frame_at(self, t: float, max_gap_s: float = 0.05) -> ForceTorqueFrame | None:
        """Interpolate frame at timestamp t within max_gap_s tolerance."""
        if not self.frames:
            return None
        t_req = float(t)
        if t_req < self.frames[0].time_s or t_req > self.frames[-1].time_s:
            return None

        # Binary search for interval
        times = [f.time_s for f in self.frames]
        idx = int(np.searchsorted(times, t_req))

        if idx < len(times) and math.isclose(
            times[idx], t_req, rel_tol=0, abs_tol=1e-12
        ):
            return self.frames[idx]
        if idx > 0 and math.isclose(times[idx - 1], t_req, rel_tol=0, abs_tol=1e-12):
            return self.frames[idx - 1]

        f0 = self.frames[idx - 1]
        f1 = self.frames[idx]
        gap = f1.time_s - f0.time_s
        if gap > max_gap_s or gap <= 0:
            return None

        alpha = (t_req - f0.time_s) / gap

        # Interpolate wrenches
        w0_map = {w.label: w for w in f0.wrenches}
        w1_map = {w.label: w for w in f1.wrenches}
        common_labels = sorted(set(w0_map.keys()) & set(w1_map.keys()))

        interp_wrenches: list[OverlayWrench] = []
        for label in common_labels:
            w0 = w0_map[label]
            w1 = w1_map[label]
            if w0.kind != w1.kind or w0.body != w1.body:
                continue

            pt = (
                (1.0 - alpha) * w0.point_m[0] + alpha * w1.point_m[0],
                (1.0 - alpha) * w0.point_m[1] + alpha * w1.point_m[1],
                (1.0 - alpha) * w0.point_m[2] + alpha * w1.point_m[2],
            )

            f_n: tuple[float, float, float] | None = None
            if w0.force_n is not None and w1.force_n is not None:
                f_n = (
                    (1.0 - alpha) * w0.force_n[0] + alpha * w1.force_n[0],
                    (1.0 - alpha) * w0.force_n[1] + alpha * w1.force_n[1],
                    (1.0 - alpha) * w0.force_n[2] + alpha * w1.force_n[2],
                )

            t_nm: tuple[float, float, float] | None = None
            if w0.torque_nm is not None and w1.torque_nm is not None:
                t_nm = (
                    (1.0 - alpha) * w0.torque_nm[0] + alpha * w1.torque_nm[0],
                    (1.0 - alpha) * w0.torque_nm[1] + alpha * w1.torque_nm[1],
                    (1.0 - alpha) * w0.torque_nm[2] + alpha * w1.torque_nm[2],
                )

            if f_n is None and t_nm is None:
                continue

            interp_wrenches.append(
                OverlayWrench(
                    kind=w0.kind,
                    label=label,
                    body=w0.body,
                    point_m=pt,
                    force_n=f_n,
                    torque_nm=t_nm,
                    source=w0.source,
                )
            )

        # Interpolate axial loads
        interp_axial: AxialLoadFrame | None = None
        if f0.axial_loads is not None and f1.axial_loads is not None:
            common_segs = set(f0.axial_loads.values_n.keys()) | set(
                f1.axial_loads.values_n.keys()
            )
            interp_vals: dict[str, float | None] = {}
            for seg in sorted(common_segs):
                v0 = f0.axial_loads.values_n.get(seg)
                v1 = f1.axial_loads.values_n.get(seg)
                if v0 is not None and v1 is not None:
                    interp_vals[seg] = (1.0 - alpha) * v0 + alpha * v1
                else:
                    interp_vals[seg] = None
            interp_axial = AxialLoadFrame(
                time_s=t_req,
                values_n=interp_vals,
                source=f0.axial_loads.source,
            )

        return ForceTorqueFrame(
            time_s=t_req,
            engine=self.engine,
            wrenches=tuple(interp_wrenches),
            axial_loads=interp_axial,
            world_frame=f0.world_frame,
            units=f0.units,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize series to dictionary."""
        return {
            "schema_version": self.SCHEMA_VERSION,
            "engine": self.engine,
            "frames": [f.to_dict() for f in self.frames],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ForceTorqueSeries:
        """Deserialize series from dictionary."""
        return cls(
            engine=data["engine"],
            frames=tuple(ForceTorqueFrame.from_dict(f) for f in data.get("frames", ())),
        )

    def to_npz(self, file: str | Path | io.BytesIO) -> None:
        """Save to npz without object pickles, encoding availability as boolean masks."""
        json_data = json.dumps(self.to_dict()).encode("utf-8")
        np.savez(file, json_bytes=np.frombuffer(json_data, dtype=np.uint8))

    @classmethod
    def from_npz(cls, file: str | Path | io.BytesIO) -> ForceTorqueSeries:
        """Load from npz with allow_pickle=False."""
        with np.load(file, allow_pickle=False) as npz:
            raw = npz["json_bytes"].tobytes().decode("utf-8")
            data = json.loads(raw)
            return cls.from_dict(data)
