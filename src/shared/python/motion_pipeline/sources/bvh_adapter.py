"""
BVH (BioVision Hierarchy) adapter for motion capture pipeline.

Part of issue #4563. Handles BVH files from Move.ai, Rokoko, and other sources.

BVH files contain:
- HIERARCHY section: joint hierarchy with Euler rotations
- MOTION section: frame data (positions + rotations)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Union

import numpy as np

from ..contracts import (
    Calibration,
    JointDef,
    JointLimit,
    JointStateFrame,
    JointTrajectory,
    Keypoint,
    KeypointFrame,
    KeypointSequence,
    SkeletonRig,
)
from .base import MocapSourceAdapter, SourceMetadata


class BVHAdapter(MocapSourceAdapter):
    """
    Adapter for BVH (BioVision Hierarchy) files.
    
    BVH is a joint hierarchy format with Euler rotations.
    Common sources: Move.ai, Rokoko, Blender exports.
    
    Coordinate system: Y-up, Z-forward (typically)
    Rotation order: XYZ (typically, but varies by source)
    """
    
    format_name = "bvh"
    supported_extensions = [".bvh"]
    
    def __init__(self, rotation_order: str = "XYZ", up_axis: str = "+Y"):
        """
        Initialize BVH adapter.
        
        Args:
            rotation_order: Euler rotation order (XYZ, XZY, YXZ, YZX, ZXY, ZYX)
            up_axis: Up axis (+Y, +Z, +X, etc.)
        """
        self.rotation_order = rotation_order
        self.up_axis = up_axis
    
    @classmethod
    def supports(cls, path: Union[str, Path]) -> bool:
        """Check if file is a BVH file."""
        path = cls._pathify_static(path)
        
        # Check extension
        if path.suffix.lower() != ".bvh":
            return False
        
        # Check for BVH header
        try:
            with open(path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip().upper()
                return first_line == "HIERARCHY"
        except (OSError, UnicodeDecodeError):
            return False
    
    @staticmethod
    def _pathify_static(path: Union[str, Path]) -> Path:
        """Static version of _pathify."""
        if isinstance(path, str):
            return Path(path)
        return path
    
    def load(
        self,
        path: Union[str, Path],
        calibration: Calibration | None = None,
    ) -> KeypointSequence | JointTrajectory:
        """
        Load a BVH file into CIR types.
        
        BVH files are joint hierarchy + motion data.
        We return a JointTrajectory with the skeleton rig.
        
        Args:
            path: Path to BVH file
            calibration: Optional calibration data
        
        Returns:
            JointTrajectory with skeleton and joint angles
        
        Raises:
            ValueError: If file cannot be parsed
            FileNotFoundError: If file does not exist
        """
        path = self._pathify(path)
        self._check_exists(path)
        
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Parse BVH
        hierarchy_section, motion_section = self._parse_sections(content)
        
        # Parse hierarchy to get skeleton
        skeleton = self._parse_hierarchy(hierarchy_section)
        
        # Parse motion data
        frames = self._parse_motion(motion_section, skeleton)
        
        # Build joint trajectory
        trajectory = JointTrajectory(
            id=f"bvh-{path.stem}",
            skeleton=skeleton,
            frames=frames,
            metadata={
                "source_file": str(path),
                "rotation_order": self.rotation_order,
                "up_axis": self.up_axis,
            }
        )
        
        return trajectory
    
    def metadata(self, path: Union[str, Path]) -> SourceMetadata:
        """
        Extract metadata from a BVH file.
        
        Args:
            path: Path to BVH file
        
        Returns:
            SourceMetadata with FPS, frame count, etc.
        """
        path = self._pathify(path)
        self._check_exists(path)
        
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Extract frame count and frame time
        frame_time_match = re.search(r"FRAME\s+TIME:\s*([\d.]+)", content, re.IGNORECASE)
        frames_match = re.search(r"FRAMES:\s*(\d+)", content, re.IGNORECASE)
        
        frame_time = float(frame_time_match.group(1)) if frame_time_match else 1.0 / 30.0
        frame_count = int(frames_match.group(1)) if frames_match else 0
        fps = 1.0 / frame_time if frame_time > 0 else 30.0
        
        return SourceMetadata(
            format_name=self.format_name,
            fps=fps,
            frame_count=frame_count,
            units="degrees",  # BVH uses degrees for rotations
            schema_name="BVH",
            duration=frame_count * frame_time,
            extra={
                "rotation_order": self.rotation_order,
                "up_axis": self.up_axis,
                "frame_time": frame_time,
            }
        )
    
    def _parse_sections(self, content: str) -> tuple[str, str]:
        """Parse BVH content into hierarchy and motion sections."""
        # Find MOTION section
        motion_match = re.search(r"\nMOTION\s*\n", content, re.IGNORECASE)
        
        if not motion_match:
            raise ValueError("BVH file missing MOTION section")
        
        hierarchy = content[:motion_match.start()].strip()
        motion = content[motion_match.end():].strip()
        
        return hierarchy, motion
    
    def _parse_hierarchy(self, hierarchy: str) -> SkeletonRig:
        """Parse BVH hierarchy section into SkeletonRig."""
        joints: dict[str, JointDef] = {}
        root_joint: str | None = None
        
        lines = hierarchy.split("\n")
        stack: list[str] = []  # Parent joint stack
        
        for line in lines:
            line = line.strip()
            
            # Root joint
            root_match = re.match(r"ROOT\s+(\S+)", line, re.IGNORECASE)
            if root_match:
                name = root_match.group(1)
                root_joint = name
                joints[name] = JointDef(name=name, parent=None)
                stack = [name]
                continue
            
            # Joint
            joint_match = re.match(r"JOINT\s+(\S+)", line, re.IGNORECASE)
            if joint_match:
                name = joint_match.group(1)
                parent = stack[-1] if stack else None
                joints[name] = JointDef(name=name, parent=parent)
                if parent and parent in joints:
                    joints[parent].children.append(name)
                stack.append(name)
                continue
            
            # End site (leaf joint)
            end_match = re.match(r"END\s+SITE\s+(\S*)", line, re.IGNORECASE)
            if end_match:
                name = end_match.group(1) or f"end_{stack[-1]}"
                parent = stack[-1] if stack else None
                joints[name] = JointDef(name=name, parent=parent)
                if parent and parent in joints:
                    joints[parent].children.append(name)
                continue
            
            # Opening brace - push to stack
            if line == "{":
                continue
            
            # Closing brace - pop from stack
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
            metadata={"rotation_order": self.rotation_order}
        )
    
    def _parse_motion(
        self,
        motion: str,
        skeleton: SkeletonRig
    ) -> list[JointStateFrame]:
        """Parse BVH motion section into joint frames."""
        lines = motion.split("\n")
        
        # Parse frame count and frame time
        frame_time = 1.0 / 30.0  # Default
        for line in lines[:2]:
            if line.upper().startswith("FRAME TIME:"):
                frame_time = float(line.split(":")[1].strip())
        
        # Find frame data (skip header lines)
        frame_data_start = 0
        for i, line in enumerate(lines):
            if line.strip() and not line.upper().startswith("FRAMES") and not line.upper().startswith("FRAME TIME"):
                frame_data_start = i
                break
        
        frame_lines = lines[frame_data_start:]
        
        # Calculate DOFs from skeleton
        num_dofs = skeleton.num_dofs
        
        frames: list[JointStateFrame] = []
        for frame_idx, line in enumerate(frame_lines):
            values = [float(v) for v in line.split()]
            
            if len(values) != num_dofs:
                # Pad or truncate to match skeleton DOFs
                if len(values) < num_dofs:
                    values.extend([0.0] * (num_dofs - len(values)))
                else:
                    values = values[:num_dofs]
            
            # Convert degrees to radians
            q = [np.deg2rad(v) for v in values]
            
            frames.append(JointStateFrame(
                timestamp=frame_idx * frame_time,
                q=q,
                qdot=None,
                qddot=None,
                frame_index=frame_idx
            ))
        
        return frames


# Register adapter
from .base import AdapterRegistryEntry

BVH_REGISTRY_ENTRY = AdapterRegistryEntry(
    adapter_class=BVHAdapter,
    priority=50,
    enabled=True
)