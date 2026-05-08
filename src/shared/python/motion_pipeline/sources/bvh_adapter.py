"""BVH (BioVision Hierarchy) adapter for the motion capture pipeline.

Part of issues #4561 / #4563. Handles BVH files from Move.ai, Rokoko, Blender,
and similar joint-hierarchy + Euler-rotation sources.

A BVH file has two sections:

- ``HIERARCHY`` - joint tree with channels (3-rot or 6-channel root)
- ``MOTION``    - per-frame channel values (degrees for rotations)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from src.shared.python.motion_pipeline.contracts import (
    Calibration,
    JointDef,
    JointStateFrame,
    JointTrajectory,
    SkeletonRig,
)
from src.shared.python.motion_pipeline.sources.base import (
    MocapSourceAdapter,
    SourceMetadata,
)
from src.shared.python.motion_pipeline.sources.registry import register_adapter


@register_adapter
class BVHAdapter(MocapSourceAdapter):
    """Adapter for BVH joint-hierarchy mocap files."""

    format_name = "bvh"
    file_extensions = (".bvh",)

    def __init__(self, rotation_order: str = "XYZ", up_axis: str = "+Y") -> None:
        self.rotation_order = rotation_order
        self.up_axis = up_axis

    @classmethod
    def supports(cls, path: Path) -> bool:
        p = Path(path)
        if p.suffix.lower() not in cls.file_extensions:
            return False
        try:
            with open(p, encoding="utf-8") as f:
                first = f.readline().strip().upper()
            return first == "HIERARCHY"
        except (OSError, UnicodeDecodeError):
            return False

    def metadata(self, path: Path) -> SourceMetadata:
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        frame_time_match = re.search(
            r"FRAME\s+TIME:\s*([\d.eE+-]+)", text, re.IGNORECASE
        )
        frames_match = re.search(r"FRAMES:\s*(\d+)", text, re.IGNORECASE)
        frame_time = (
            float(frame_time_match.group(1)) if frame_time_match else 1.0 / 30.0
        )
        if frame_time <= 0:
            frame_time = 1.0 / 30.0
        frame_count = int(frames_match.group(1)) if frames_match else 0
        fps = 1.0 / frame_time
        return SourceMetadata(
            format_name=self.format_name,
            fps=fps,
            frame_count=frame_count,
            unit_system="degrees",
            keypoint_schema=None,
            marker_set_name=None,
            notes=f"rotation_order={self.rotation_order}, up_axis={self.up_axis}",
        )

    def load(
        self,
        path: Path,
        calibration: Calibration | None = None,
    ) -> JointTrajectory:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"BVH file not found: {p}")
        content = p.read_text(encoding="utf-8")
        hierarchy, motion = self._split_sections(content)
        skeleton = self._parse_hierarchy(hierarchy)
        frames = self._parse_motion(motion, skeleton)
        return JointTrajectory(
            id=f"bvh-{p.stem}",
            skeleton=skeleton,
            frames=frames,
            metadata={
                "source_file": str(p),
                "rotation_order": self.rotation_order,
                "up_axis": self.up_axis,
            },
        )

    # ------------------------------------------------------------------
    # Parsing helpers

    @staticmethod
    def _split_sections(content: str) -> tuple[str, str]:
        m = re.search(r"\nMOTION\s*\n", content, re.IGNORECASE)
        if not m:
            raise ValueError("BVH file missing MOTION section")
        return content[: m.start()].strip(), content[m.end() :].strip()

    def _parse_hierarchy(self, hierarchy: str) -> SkeletonRig:
        joints: dict[str, JointDef] = {}
        root_joint: str | None = None
        stack: list[str] = []

        for raw in hierarchy.splitlines():
            line = raw.strip()
            root_match = re.match(r"ROOT\s+(\S+)", line, re.IGNORECASE)
            if root_match:
                name = root_match.group(1)
                root_joint = name
                joints[name] = JointDef(name=name, parent=None)
                stack = [name]
                continue
            joint_match = re.match(r"JOINT\s+(\S+)", line, re.IGNORECASE)
            if joint_match:
                name = joint_match.group(1)
                parent = stack[-1] if stack else None
                joints[name] = JointDef(name=name, parent=parent)
                if parent and parent in joints:
                    joints[parent].children.append(name)
                stack.append(name)
                continue
            if re.match(r"END\s+SITE", line, re.IGNORECASE):
                # End sites are anonymous leaves; skip without altering stack
                # the closing brace will pop the parent only if pushed.
                continue
            if line == "{":
                continue
            if line == "}":
                if stack:
                    stack.pop()
                continue

        if not root_joint:
            raise ValueError("BVH file missing ROOT joint")
        return SkeletonRig(
            id="bvh-skeleton",
            joints=joints,
            root_joint=root_joint,
            up_axis=self.up_axis,
            metadata={"rotation_order": self.rotation_order},
        )

    @staticmethod
    def _parse_motion(motion: str, skeleton: SkeletonRig) -> list[JointStateFrame]:
        lines = motion.splitlines()
        frame_time = 1.0 / 30.0
        data_start = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            up = stripped.upper()
            if up.startswith("FRAMES:"):
                continue
            if up.startswith("FRAME TIME:"):
                try:
                    frame_time = float(stripped.split(":", 1)[1].strip())
                except ValueError:
                    frame_time = 1.0 / 30.0
                continue
            data_start = i
            break

        num_dofs = skeleton.num_dofs
        frames: list[JointStateFrame] = []
        for idx, raw in enumerate(lines[data_start:]):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                values = [float(v) for v in stripped.split()]
            except ValueError as e:
                raise ValueError(
                    f"BVH motion line {idx + data_start} has non-numeric data: {stripped!r}"
                ) from e
            if len(values) < num_dofs:
                values.extend([0.0] * (num_dofs - len(values)))
            elif len(values) > num_dofs:
                values = values[:num_dofs]
            q = [float(np.deg2rad(v)) for v in values]
            frames.append(
                JointStateFrame(
                    timestamp=idx * frame_time,
                    q=q,
                    qdot=None,
                    qddot=None,
                    frame_index=idx,
                )
            )
        if not frames:
            raise ValueError("BVH file has MOTION section with no frames")
        return frames
