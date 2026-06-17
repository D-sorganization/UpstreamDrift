# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""Motion retargeting between different embodiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _squared_euclidean_error(
    current: NDArray[np.floating], target: NDArray[np.floating]
) -> float:
    """Return squared Euclidean error without allocating squared temporaries."""
    diff = current - target
    return float(np.vdot(diff, diff))


def _validate_finite_array(
    value: NDArray[np.floating],
    *,
    name: str,
    shape: tuple[int | None, ...],
) -> NDArray[np.floating]:
    """Validate an array boundary contract and return a float ndarray view."""
    if value is None:
        raise ValueError(f"{name} must be provided")
    array = np.asarray(value, dtype=float)
    if array.ndim != len(shape):
        raise ValueError(f"{name} must have {len(shape)} dimensions")
    for axis, expected in enumerate(shape):
        if expected is not None and array.shape[axis] != expected:
            raise ValueError(
                f"{name}.shape[{axis}] must be {expected}; got {array.shape[axis]}",
            )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _validate_joint_angles(
    joint_angles: NDArray[np.floating],
    skeleton: SkeletonConfig,
    *,
    name: str = "joint_angles",
) -> NDArray[np.floating]:
    return _validate_finite_array(
        joint_angles,
        name=name,
        shape=(skeleton.n_joints,),
    )


def _validate_motion_matrix(
    motion: NDArray[np.floating],
    skeleton: SkeletonConfig,
    *,
    name: str = "source_motion",
) -> NDArray[np.floating]:
    return _validate_finite_array(
        motion,
        name=name,
        shape=(None, skeleton.n_joints),
    )


def _validate_position_mapping(
    positions: dict[str, NDArray[np.floating]],
    *,
    name: str,
) -> None:
    if positions is None:
        raise ValueError(f"{name} must be provided")
    for joint_name, position in positions.items():
        _validate_finite_array(position, name=f"{name}[{joint_name!r}]", shape=(3,))


def _build_end_effector_chain_indices(
    skeleton: SkeletonConfig,
) -> dict[str, tuple[int, ...]]:
    """Build root-to-end-effector FK chains once using joint indices."""
    chains: dict[str, tuple[int, ...]] = {}
    for ee_name in skeleton.end_effectors:
        idx = skeleton.get_joint_index(ee_name)
        chain: list[int] = []
        while idx >= 0:
            chain.append(idx)
            idx = skeleton.parent_indices[idx]
        chains[ee_name] = tuple(reversed(chain))
    return chains


@dataclass
class SkeletonConfig:
    """Skeleton configuration for motion retargeting.

    Describes the kinematic structure of a skeleton for
    motion transfer between different embodiments.

    Attributes:
        name: Skeleton name/identifier.
        joint_names: List of joint names.
        parent_indices: Parent index for each joint (-1 for root).
        joint_offsets: T-pose offsets from parent (n_joints, 3).
        joint_axes: Rotation axes for each joint (n_joints, 3).
        joint_limits: Joint angle limits (n_joints, 2) as [min, max].
        semantic_labels: Mapping of semantic names to joint names.
        end_effectors: Names of end-effector joints.
    """

    name: str
    joint_names: list[str]
    parent_indices: list[int]
    joint_offsets: NDArray[np.floating]
    joint_axes: NDArray[np.floating] | None = None
    joint_limits: NDArray[np.floating] | None = None
    semantic_labels: dict[str, str] = field(default_factory=dict)
    end_effectors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate skeleton configuration."""
        n_joints = len(self.joint_names)

        if len(self.parent_indices) != n_joints:
            raise ValueError(
                f"parent_indices length ({len(self.parent_indices)}) "
                f"must match joint_names ({n_joints})",
            )

        self.joint_offsets = _validate_finite_array(
            self.joint_offsets,
            name="joint_offsets",
            shape=(n_joints, 3),
        )

        for joint_idx, parent_idx in enumerate(self.parent_indices):
            if parent_idx >= n_joints or parent_idx < -1:
                raise ValueError(
                    "parent_indices entries must be -1 or valid joint indices; "
                    f"got parent_indices[{joint_idx}]={parent_idx}",
                )
            if parent_idx == joint_idx:
                raise ValueError("parent_indices entries cannot reference themselves")

        if self.joint_offsets.shape[0] != n_joints:
            raise ValueError(
                f"joint_offsets rows ({self.joint_offsets.shape[0]}) "
                f"must match joint_names ({n_joints})",
            )

        if self.joint_axes is None:
            # Default to z-axis rotation
            self.joint_axes = np.tile(np.array([0.0, 0.0, 1.0]), (n_joints, 1))
        else:
            self.joint_axes = _validate_finite_array(
                self.joint_axes,
                name="joint_axes",
                shape=(n_joints, 3),
            )

        if self.joint_limits is None:
            # Default to +/- pi
            self.joint_limits = np.array([[-np.pi, np.pi]] * n_joints)
        else:
            self.joint_limits = _validate_finite_array(
                self.joint_limits,
                name="joint_limits",
                shape=(n_joints, 2),
            )
        if np.any(self.joint_limits[:, 0] > self.joint_limits[:, 1]):
            raise ValueError("joint_limits lower bounds must be <= upper bounds")

    @property
    def n_joints(self) -> int:
        """Number of joints in skeleton."""
        return len(self.joint_names)

    def get_joint_index(self, name: str) -> int:
        """Get joint index by name.

        Args:
            name: Joint name.

        Returns:
            Joint index.

        Raises:
            ValueError: If joint not found.
        """
        try:
            return self.joint_names.index(name)
        except ValueError:
            raise ValueError(f"Joint '{name}' not found in skeleton") from None

    def get_semantic_joint(self, semantic_name: str) -> str | None:
        """Get joint name from semantic label.

        Args:
            semantic_name: Semantic label (e.g., "left_shoulder").

        Returns:
            Joint name or None if not mapped.
        """
        return self.semantic_labels.get(semantic_name)

    def get_kinematic_chain(self, end_joint: str) -> list[str]:
        """Get kinematic chain from root to end joint.

        Args:
            end_joint: Name of end joint.

        Returns:
            List of joint names from root to end.
        """
        if end_joint is None:
            raise ValueError("end_joint must be provided")
        chain: list[str] = []
        idx = self.get_joint_index(end_joint)

        while idx >= 0:
            chain.insert(0, self.joint_names[idx])
            idx = self.parent_indices[idx]

        return chain

    @classmethod
    def create_humanoid(cls) -> SkeletonConfig:
        """Create a standard humanoid skeleton configuration.

        Returns:
            Humanoid skeleton config.
        """
        joint_names, parent_indices = _humanoid_joint_names_and_parents()
        joint_offsets = _humanoid_joint_offsets()
        semantic_labels = _humanoid_semantic_labels()
        end_effectors = ["head", "left_hand", "right_hand", "left_foot", "right_foot"]

        return cls(
            name="humanoid",
            joint_names=joint_names,
            parent_indices=parent_indices,
            joint_offsets=joint_offsets,
            semantic_labels=semantic_labels,
            end_effectors=end_effectors,
        )


def _humanoid_joint_names_and_parents() -> tuple[list[str], list[int]]:
    joint_names = [
        "pelvis",
        "spine_1",
        "spine_2",
        "spine_3",
        "neck",
        "head",
        "left_hip",
        "left_knee",
        "left_ankle",
        "left_foot",
        "right_hip",
        "right_knee",
        "right_ankle",
        "right_foot",
        "left_shoulder",
        "left_elbow",
        "left_wrist",
        "left_hand",
        "right_shoulder",
        "right_elbow",
        "right_wrist",
        "right_hand",
    ]

    parent_indices = [
        -1,
        0,
        1,
        2,
        3,
        4,  # Spine chain
        0,
        6,
        7,
        8,  # Left leg
        0,
        10,
        11,
        12,  # Right leg
        3,
        14,
        15,
        16,  # Left arm
        3,
        18,
        19,
        20,  # Right arm
    ]
    return joint_names, parent_indices


def _humanoid_joint_offsets() -> NDArray[np.floating]:
    return np.array(
        [
            [0, 0, 0],  # pelvis (root)
            [0, 0, 0.1],  # spine_1
            [0, 0, 0.1],  # spine_2
            [0, 0, 0.1],  # spine_3
            [0, 0, 0.1],  # neck
            [0, 0, 0.1],  # head
            [0.1, 0, 0],  # left_hip
            [0, 0, -0.4],  # left_knee
            [0, 0, -0.4],  # left_ankle
            [0, 0.1, 0],  # left_foot
            [-0.1, 0, 0],  # right_hip
            [0, 0, -0.4],  # right_knee
            [0, 0, -0.4],  # right_ankle
            [0, 0.1, 0],  # right_foot
            [0.15, 0, 0],  # left_shoulder
            [0.3, 0, 0],  # left_elbow
            [0.25, 0, 0],  # left_wrist
            [0.1, 0, 0],  # left_hand
            [-0.15, 0, 0],  # right_shoulder
            [-0.3, 0, 0],  # right_elbow
            [-0.25, 0, 0],  # right_wrist
            [-0.1, 0, 0],  # right_hand
        ],
    )


def _humanoid_semantic_labels() -> dict[str, str]:
    return {
        "pelvis": "pelvis",
        "spine": "spine_2",
        "chest": "spine_3",
        "neck": "neck",
        "head": "head",
        "left_hip": "left_hip",
        "left_knee": "left_knee",
        "left_ankle": "left_ankle",
        "left_foot": "left_foot",
        "right_hip": "right_hip",
        "right_knee": "right_knee",
        "right_ankle": "right_ankle",
        "right_foot": "right_foot",
        "left_shoulder": "left_shoulder",
        "left_elbow": "left_elbow",
        "left_wrist": "left_wrist",
        "left_hand": "left_hand",
        "right_shoulder": "right_shoulder",
        "right_elbow": "right_elbow",
        "right_wrist": "right_wrist",
        "right_hand": "right_hand",
    }


class MotionRetargeter:
    """Retarget motion between different skeleton types.

    Supports multiple retargeting methods:
    - Direct joint mapping (same topology)
    - IK-based retargeting (different topologies)
    - Optimization-based retargeting

    Attributes:
        source_skeleton: Source skeleton configuration.
        target_skeleton: Target skeleton configuration.
    """

    def __init__(
        self,
        source_skeleton: SkeletonConfig,
        target_skeleton: SkeletonConfig,
    ) -> None:
        """Initialize motion retargeter.

        Args:
            source_skeleton: Source skeleton configuration.
            target_skeleton: Target skeleton configuration.
        """
        if source_skeleton is None:
            raise ValueError("source_skeleton must be provided")
        if target_skeleton is None:
            raise ValueError("target_skeleton must be provided")
        self.source = source_skeleton
        self.target = target_skeleton
        self._joint_mapping = self._compute_joint_mapping()
        self._scale_factors = self._compute_scale_factors()
        self._end_effector_chain_indices = {
            id(self.source): _build_end_effector_chain_indices(self.source),
            id(self.target): _build_end_effector_chain_indices(self.target),
        }

    def _end_effector_chains_for(
        self,
        skeleton: SkeletonConfig,
    ) -> dict[str, tuple[int, ...]]:
        cache_key = id(skeleton)
        chains = self._end_effector_chain_indices.get(cache_key)
        if chains is None:
            chains = _build_end_effector_chain_indices(skeleton)
            self._end_effector_chain_indices[cache_key] = chains
        return chains

    def _compute_joint_mapping(self) -> dict[str, str]:
        """Compute mapping between source and target joints.

        Uses semantic labels to establish correspondence.

        Returns:
            Dictionary mapping source joints to target joints.
        """
        mapping = {}

        for semantic_name, source_joint in self.source.semantic_labels.items():
            target_joint = self.target.get_semantic_joint(semantic_name)
            if target_joint is not None:
                mapping[source_joint] = target_joint

        return mapping

    def _compute_scale_factors(self) -> dict[str, float]:
        """Compute scale factors for bone lengths.

        Returns:
            Dictionary of scale factors per joint.
        """
        scales = {}

        for source_joint, target_joint in self._joint_mapping.items():
            source_idx = self.source.get_joint_index(source_joint)
            target_idx = self.target.get_joint_index(target_joint)

            source_len = np.linalg.norm(self.source.joint_offsets[source_idx])
            target_len = np.linalg.norm(self.target.joint_offsets[target_idx])

            if source_len > 1e-6:
                scales[target_joint] = float(target_len / source_len)
            else:
                scales[target_joint] = 1.0

        return scales

    def retarget(
        self,
        source_motion: NDArray[np.floating],
        method: str = "direct",
    ) -> NDArray[np.floating]:
        """Retarget motion to target skeleton.

        Args:
            source_motion: Source motion data (T, n_source_joints).
            method: Retargeting method ("direct", "optimization", "ik").

        Returns:
            Retargeted motion for target skeleton.
        """
        if method == "direct":
            return self._retarget_direct(source_motion)
        if method == "optimization":
            return self._retarget_optimization(source_motion)
        if method == "ik":
            return self._retarget_ik(source_motion)
        raise ValueError(f"Unknown retargeting method: {method}")

    def _retarget_direct(
        self,
        source_motion: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Direct joint-angle mapping.

        Args:
            source_motion: Source joint angles (T, n_source).

        Returns:
            Target joint angles (T, n_target).
        """
        source_motion = _validate_motion_matrix(source_motion, self.source)
        n_frames = source_motion.shape[0]
        target_motion = np.zeros((n_frames, self.target.n_joints))

        for source_joint, target_joint in self._joint_mapping.items():
            source_idx = self.source.get_joint_index(source_joint)
            target_idx = self.target.get_joint_index(target_joint)

            # Direct copy with potential scaling
            target_motion[:, target_idx] = source_motion[:, source_idx]

        # Apply joint limits
        if self.target.joint_limits is not None:
            for j in range(self.target.n_joints):
                lower, upper = self.target.joint_limits[j]
                target_motion[:, j] = np.clip(target_motion[:, j], lower, upper)

        _validate_motion_matrix(target_motion, self.target, name="target_motion")
        return target_motion

    def _retarget_optimization(
        self,
        source_motion: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Optimization-based retargeting.

        Optimizes to match end-effector positions and orientations.

        Args:
            source_motion: Source motion data.

        Returns:
            Optimized target motion.
        """
        source_motion = _validate_motion_matrix(source_motion, self.source)
        n_frames = source_motion.shape[0]
        target_motion = np.zeros((n_frames, self.target.n_joints))

        # Initialize with direct mapping
        initial_guess = self._retarget_direct(source_motion)

        # Optimize each frame
        for t in range(n_frames):
            source_frame = source_motion[t]

            # Compute source end-effector positions
            source_ee_positions = self._compute_end_effector_positions(
                source_frame,
                self.source,
            )

            # Optimize target angles to match end-effector positions
            target_frame = self._optimize_frame(
                initial_guess[t],
                source_ee_positions,
            )
            target_motion[t] = target_frame

        _validate_motion_matrix(target_motion, self.target, name="target_motion")
        return target_motion

    def _compute_end_effector_positions(
        self,
        joint_angles: NDArray[np.floating],
        skeleton: SkeletonConfig,
    ) -> dict[str, NDArray[np.floating]]:
        """Compute end-effector positions via forward kinematics.

        Args:
            joint_angles: Joint angles.
            skeleton: Skeleton configuration.

        Returns:
            Dictionary of end-effector positions.
        """
        joint_angles = _validate_joint_angles(joint_angles, skeleton)
        positions: dict[str, NDArray[np.floating]] = {}

        # Compute forward kinematics for each end-effector
        for ee_name, chain_indices in self._end_effector_chains_for(skeleton).items():
            position = np.zeros(3, dtype=float)

            for idx in chain_indices:
                offset = skeleton.joint_offsets[idx]
                angle = joint_angles[idx]

                # Simplified: assume z-axis rotation
                c, s = np.cos(angle), np.sin(angle)
                x, y, z = position
                position[0] = c * x - s * y + offset[0]
                position[1] = s * x + c * y + offset[1]
                position[2] = z + offset[2]

            positions[ee_name] = position

        _validate_position_mapping(positions, name="positions")
        return positions

    def _compute_end_effector_error(
        self,
        current_positions: dict[str, NDArray[np.floating]],
        target_positions: dict[str, NDArray[np.floating]],
    ) -> float:
        """Return the squared end-effector objective for configured targets."""
        total_error = 0.0
        for ee_name in self.target.end_effectors:
            if ee_name in target_positions:
                target_position = target_positions.get(ee_name, np.zeros(3))
                current_position = current_positions.get(ee_name, np.zeros(3))
                total_error += _squared_euclidean_error(
                    current_position, target_position
                )
        return total_error

    def _optimize_frame(
        self,
        initial_angles: NDArray[np.floating],
        target_ee_positions: dict[str, NDArray[np.floating]],
        max_iterations: int = 50,
    ) -> NDArray[np.floating]:
        """Optimize joint angles for a single frame.

        Args:
            initial_angles: Initial guess for joint angles.
            target_ee_positions: Target end-effector positions.
            max_iterations: Maximum optimization iterations.

        Returns:
            Optimized joint angles.
        """
        initial_angles = _validate_joint_angles(
            initial_angles, self.target, name="initial_angles"
        )
        _validate_position_mapping(target_ee_positions, name="target_ee_positions")
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        angles = initial_angles.copy()
        step_size = 0.01
        gradient = np.empty_like(angles)
        angles_plus = angles.copy()

        for _ in range(max_iterations):
            # Compute current end-effector positions
            current_ee = self._compute_end_effector_positions(angles, self.target)

            # Compute error
            total_error = self._compute_end_effector_error(
                current_ee, target_ee_positions
            )

            if total_error < 1e-6:
                break

            # Gradient descent step (numerical gradient)
            gradient.fill(0.0)
            eps = 1e-4
            angles_plus[:] = angles

            for j in range(len(angles)):
                angles_plus[j] += eps
                ee_plus = self._compute_end_effector_positions(angles_plus, self.target)

                error_plus = self._compute_end_effector_error(
                    ee_plus, target_ee_positions
                )

                gradient[j] = (error_plus - total_error) / eps
                angles_plus[j] = angles[j]

            angles = angles - step_size * gradient

            # Apply joint limits
            if self.target.joint_limits is not None:
                for j in range(len(angles)):
                    lower, upper = self.target.joint_limits[j]
                    angles[j] = np.clip(angles[j], lower, upper)

        _validate_joint_angles(angles, self.target, name="optimized_angles")
        return angles

    def _retarget_ik(
        self,
        source_motion: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """IK-based retargeting.

        Uses inverse kinematics to match end-effector poses.

        Args:
            source_motion: Source motion data.

        Returns:
            Retargeted motion using IK.
        """
        # For now, use optimization-based approach
        # Full IK would integrate with physics engine IK solver
        return self._retarget_optimization(source_motion)

    def retarget_from_mocap(
        self,
        marker_positions: NDArray[np.floating],
        marker_names: list[str],
        marker_to_joint_mapping: dict[str, str] | None = None,
    ) -> NDArray[np.floating]:
        """Retarget from motion capture marker data.

        Args:
            marker_positions: Marker positions (T, n_markers, 3).
            marker_names: Names of markers.
            marker_to_joint_mapping: Mapping of markers to joints.

        Returns:
            Retargeted joint angles.
        """
        if marker_names is None:
            raise ValueError("marker_names must be provided")
        marker_positions = _validate_finite_array(
            marker_positions,
            name="marker_positions",
            shape=(None, len(marker_names), 3),
        )
        n_frames = marker_positions.shape[0]
        target_motion = np.zeros((n_frames, self.target.n_joints))

        # Default marker to joint mapping based on common naming
        if marker_to_joint_mapping is None:
            marker_to_joint_mapping = self._infer_marker_mapping(marker_names)

        name_to_idx: dict[str, int] = {}
        for marker_idx, marker_name in enumerate(marker_names):
            name_to_idx.setdefault(marker_name, marker_idx)
        mapped_marker_indices: list[tuple[int, str]] = []
        for marker_name, joint_name in marker_to_joint_mapping.items():
            marker_lookup_idx = name_to_idx.get(marker_name)
            if marker_lookup_idx is not None:
                mapped_marker_indices.append((marker_lookup_idx, joint_name))

        for t in range(n_frames):
            # Extract joint positions from markers
            joint_positions = {}
            for marker_idx, joint_name in mapped_marker_indices:
                joint_positions[joint_name] = marker_positions[t, marker_idx]

            # Convert positions to joint angles via IK
            target_motion[t] = self._positions_to_angles(joint_positions)

        _validate_motion_matrix(target_motion, self.target, name="target_motion")
        return target_motion

    def _infer_marker_mapping(
        self,
        marker_names: list[str],
    ) -> dict[str, str]:
        """Infer marker to joint mapping from marker names.

        Args:
            marker_names: List of marker names.

        Returns:
            Mapping dictionary.
        """
        if marker_names is None:
            raise ValueError("marker_names must be provided")
        mapping = {}
        common_mappings = {
            "LSHO": "left_shoulder",
            "RSHO": "right_shoulder",
            "LELB": "left_elbow",
            "RELB": "right_elbow",
            "LWRI": "left_wrist",
            "RWRI": "right_wrist",
            "LHIP": "left_hip",
            "RHIP": "right_hip",
            "LKNE": "left_knee",
            "RKNE": "right_knee",
            "LANK": "left_ankle",
            "RANK": "right_ankle",
        }

        for marker_name in marker_names:
            upper_name = marker_name.upper()
            if upper_name in common_mappings:
                semantic = common_mappings[upper_name]
                joint = self.target.get_semantic_joint(semantic)
                if joint:
                    mapping[marker_name] = joint

        return mapping

    def _positions_to_angles(
        self,
        joint_positions: dict[str, NDArray[np.floating]],
    ) -> NDArray[np.floating]:
        """Convert joint positions to joint angles via IK.

        Args:
            joint_positions: Dictionary of joint positions.

        Returns:
            Joint angles.
        """
        # Start with zero angles
        _validate_position_mapping(joint_positions, name="joint_positions")
        angles = np.zeros(self.target.n_joints)

        # For each kinematic chain ending at a positioned joint,
        # solve IK to find joint angles
        for joint_name, target_pos in joint_positions.items():
            if joint_name not in self.target.joint_names:
                continue

            # Get kinematic chain
            chain = self.target.get_kinematic_chain(joint_name)

            # Simple analytical IK for 2-link chains
            if len(chain) >= 2:
                # Compute angle using law of cosines
                parent_idx = self.target.get_joint_index(chain[-2])
                joint_idx = self.target.get_joint_index(chain[-1])

                parent_offset = np.linalg.norm(self.target.joint_offsets[parent_idx])
                joint_offset = np.linalg.norm(self.target.joint_offsets[joint_idx])

                # Distance to target
                dist = np.linalg.norm(target_pos)

                if parent_offset + joint_offset > 0:
                    # Law of cosines for elbow angle
                    cos_angle = (parent_offset**2 + joint_offset**2 - dist**2) / (
                        2 * parent_offset * joint_offset + 1e-6
                    )
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angles[parent_idx] = np.arccos(cos_angle)

        return angles

    def get_joint_mapping(self) -> dict[str, str]:
        """Get the computed joint mapping.

        Returns:
            Dictionary mapping source joints to target joints.
        """
        return self._joint_mapping.copy()

    def visualize_mapping(self) -> str:
        """Generate a text visualization of the joint mapping.

        Returns:
            Multi-line string showing the mapping.
        """
        lines = [
            f"Motion Retargeting: {self.source.name} -> {self.target.name}",
            "=" * 50,
            "",
            "Joint Mapping:",
        ]

        for source, target in sorted(self._joint_mapping.items()):
            scale = self._scale_factors.get(target, 1.0)
            lines.append(f"  {source:20s} -> {target:20s} (scale: {scale:.2f})")

        lines.append("")
        lines.append(f"Mapped joints: {len(self._joint_mapping)}")
        lines.append(f"Source joints: {self.source.n_joints}")
        lines.append(f"Target joints: {self.target.n_joints}")

        return "\n".join(lines)
