"""OpenSim STO / MOT adapter.

STO (states) and MOT (motion) files share a header/columns/data layout:

    <name>
    version=1
    nRows=<int>
    nColumns=<int>
    inDegrees=yes|no
    endheader
    time   <col1>   <col2>   ...
    <values...>

Adapter returns a :class:`MotionTrajectory` wrapping a :class:`JointTrajectory`
with a synthetic 1-DOF-per-column skeleton; a richer mapping to a real
:class:`SkeletonRig` is the responsibility of downstream stages.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.shared.python.motion_pipeline.contracts import (
    Calibration,
    JointDef,
    JointStateFrame,
    JointTrajectory,
    MotionTrajectory,
    SkeletonRig,
)
from src.shared.python.motion_pipeline.sources.base import (
    MocapSourceAdapter,
    SourceMetadata,
)
from src.shared.python.motion_pipeline.sources.registry import register_adapter


@register_adapter
class OpenSimSTOMOTAdapter(MocapSourceAdapter):
    """Parser for OpenSim ``.sto`` and ``.mot`` text files."""

    format_name = "opensim_sto_mot"
    file_extensions = (".sto", ".mot")

    @classmethod
    def supports(cls, path: Path) -> bool:
        p = Path(path)
        if p.suffix.lower() not in cls.file_extensions:
            return False
        try:
            with open(p, encoding="utf-8") as f:
                head = f.read(4096)
        except (OSError, UnicodeDecodeError):
            return False
        return "endheader" in head.lower()

    def _parse_header(self, text: str) -> tuple[dict[str, str], list[str], list[str]]:
        lines = text.splitlines()
        meta: dict[str, str] = {}
        end_idx = -1
        for i, line in enumerate(lines):
            if line.strip().lower() == "endheader":
                end_idx = i
                break
            if "=" in line:
                k, _, v = line.partition("=")
                meta[k.strip()] = v.strip()
        if end_idx < 0:
            raise ValueError("OpenSim STO/MOT missing 'endheader' line")
        if end_idx + 1 >= len(lines):
            raise ValueError("OpenSim STO/MOT has no column header line")
        column_line = lines[end_idx + 1]
        columns = column_line.split()
        data_lines = [ln for ln in lines[end_idx + 2 :] if ln.strip()]
        return meta, columns, data_lines

    def metadata(self, path: Path) -> SourceMetadata:
        p = Path(path)
        meta, columns, data_lines = self._parse_header(p.read_text(encoding="utf-8"))
        n_rows = len(data_lines)
        # Try to infer FPS from the first two timestamps
        fps = 100.0
        if len(data_lines) >= 2:
            try:
                t0 = float(data_lines[0].split()[0])
                t1 = float(data_lines[1].split()[0])
                if t1 > t0:
                    fps = 1.0 / (t1 - t0)
            except (ValueError, IndexError):
                pass
        in_degrees = meta.get("inDegrees", "no").strip().lower() == "yes"
        unit_system = "degrees" if in_degrees else "radians"
        return SourceMetadata(
            format_name=self.format_name,
            fps=fps,
            frame_count=n_rows,
            unit_system=unit_system,  # type: ignore[arg-type]
            notes=f"columns={len(columns) - 1}",
        )

    def load(
        self,
        path: Path,
        calibration: Calibration | None = None,
    ) -> MotionTrajectory:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"OpenSim STO/MOT file not found: {p}")
        meta, columns, data_lines = self._parse_header(p.read_text(encoding="utf-8"))
        if not columns:
            raise ValueError("OpenSim STO/MOT has empty column header")
        if columns[0].lower() != "time":
            raise ValueError(f"Expected first column 'time', got {columns[0]!r}")
        joint_cols = columns[1:]
        if not joint_cols:
            raise ValueError("OpenSim STO/MOT has no joint columns")

        in_degrees = meta.get("inDegrees", "no").strip().lower() == "yes"
        joints: dict[str, JointDef] = {}
        # Synthetic skeleton: a single root with one chained joint per column.
        prev = None
        for name in joint_cols:
            joints[name] = JointDef(name=name, parent=prev, axes=["X"])
            if prev is not None:
                joints[prev].children.append(name)
            prev = name
        root_name = joint_cols[0]
        skeleton = SkeletonRig(
            id=f"opensim-{p.stem}",
            joints=joints,
            root_joint=root_name,
            metadata={"source_file": str(p)},
        )

        frames: list[JointStateFrame] = []
        for idx, raw in enumerate(data_lines):
            tokens = raw.split()
            if len(tokens) != len(columns):
                raise ValueError(
                    f"STO/MOT row {idx} has {len(tokens)} cols, expected {len(columns)}"
                )
            try:
                t = float(tokens[0])
                vals = [float(v) for v in tokens[1:]]
            except ValueError as e:
                raise ValueError(f"Non-numeric data in STO/MOT row {idx}") from e
            if in_degrees:
                vals = [float(np.deg2rad(v)) for v in vals]
            frames.append(JointStateFrame(timestamp=t, q=vals, frame_index=idx))
        if not frames:
            raise ValueError(f"OpenSim STO/MOT {p} has no data rows")

        traj = JointTrajectory(
            id=f"opensim-traj-{p.stem}",
            skeleton=skeleton,
            frames=frames,
            metadata={"source_file": str(p), "header": meta},
        )
        return MotionTrajectory(
            id=f"opensim-motion-{p.stem}",
            skeleton=skeleton,
            trajectory=traj,
            source_provenance={"format": self.format_name, "source_file": str(p)},
        )
