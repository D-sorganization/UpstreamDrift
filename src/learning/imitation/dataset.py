"""Demonstration dataset for imitation learning."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class Demonstration:
    """A single demonstration trajectory.

    Attributes:
        timestamps: Time values for each frame (T,).
        joint_positions: Joint positions over time (T, n_q).
        joint_velocities: Joint velocities over time (T, n_v).
        actions: Control actions if available (T, n_u).
        end_effector_poses: End-effector poses if available (T, 7).
        contact_states: Contact states per timestep.
        task_id: Identifier for the task being demonstrated.
        success: Whether demonstration completed successfully.
        source: Source of demonstration (teleoperation, mocap, etc.).
        metadata: Additional metadata.
    """

    timestamps: NDArray[np.floating]
    joint_positions: NDArray[np.floating]
    joint_velocities: NDArray[np.floating]
    actions: NDArray[np.floating] | None = None
    end_effector_poses: NDArray[np.floating] | None = None
    contact_states: list[list[dict[str, Any]]] | None = None
    task_id: str | None = None
    success: bool = True
    source: str = "teleoperation"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate demonstration data."""
        if len(self.timestamps) != len(self.joint_positions):
            raise ValueError(
                f"Timestamps ({len(self.timestamps)}) and joint_positions "
                f"({len(self.joint_positions)}) must have same length"
            )
        if len(self.timestamps) != len(self.joint_velocities):
            raise ValueError(
                f"Timestamps ({len(self.timestamps)}) and joint_velocities "
                f"({len(self.joint_velocities)}) must have same length"
            )

    @property
    def duration(self) -> float:
        """Total duration of demonstration in seconds."""
        return float(self.timestamps[-1] - self.timestamps[0])

    @property
    def n_frames(self) -> int:
        """Number of frames in demonstration."""
        return len(self.timestamps)

    @property
    def n_joints(self) -> int:
        """Number of joints."""
        return self.joint_positions.shape[1]

    def get_frame(self, idx: int) -> dict[str, Any]:
        """Get a single frame from the demonstration.

        Args:
            idx: Frame index.

        Returns:
            Dictionary with frame data.
        """
        frame = {
            "timestamp": self.timestamps[idx],
            "joint_positions": self.joint_positions[idx],
            "joint_velocities": self.joint_velocities[idx],
        }
        if self.actions is not None:
            frame["action"] = self.actions[idx]
        if self.end_effector_poses is not None:
            frame["ee_pose"] = self.end_effector_poses[idx]
        return frame

    def subsample(self, factor: int) -> "Demonstration":
        """Subsample demonstration by a factor.

        Args:
            factor: Subsampling factor (keep every nth frame).

        Returns:
            Subsampled demonstration.
        """
        indices = np.arange(0, len(self.timestamps), factor)
        return Demonstration(
            timestamps=self.timestamps[indices],
            joint_positions=self.joint_positions[indices],
            joint_velocities=self.joint_velocities[indices],
            actions=self.actions[indices] if self.actions is not None else None,
            end_effector_poses=(
                self.end_effector_poses[indices]
                if self.end_effector_poses is not None
                else None
            ),
            contact_states=(
                [self.contact_states[i] for i in indices]
                if self.contact_states is not None
                else None
            ),
            task_id=self.task_id,
            success=self.success,
            source=self.source,
            metadata=self.metadata.copy(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = {
            "timestamps": self.timestamps.tolist(),
            "joint_positions": self.joint_positions.tolist(),
            "joint_velocities": self.joint_velocities.tolist(),
            "task_id": self.task_id,
            "success": self.success,
            "source": self.source,
            "metadata": self.metadata,
        }
        if self.actions is not None:
            data["actions"] = self.actions.tolist()
        if self.end_effector_poses is not None:
            data["end_effector_poses"] = self.end_effector_poses.tolist()
        if self.contact_states is not None:
            data["contact_states"] = self.contact_states
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Demonstration":
        """Create demonstration from dictionary.

        Args:
            data: Dictionary with demonstration data.

        Returns:
            Demonstration instance.
        """
        return cls(
            timestamps=np.array(data["timestamps"]),
            joint_positions=np.array(data["joint_positions"]),
            joint_velocities=np.array(data["joint_velocities"]),
            actions=np.array(data["actions"]) if "actions" in data else None,
            end_effector_poses=(
                np.array(data["end_effector_poses"])
                if "end_effector_poses" in data
                else None
            ),
            contact_states=data.get("contact_states"),
            task_id=data.get("task_id"),
            success=data.get("success", True),
            source=data.get("source", "unknown"),
            metadata=data.get("metadata", {}),
        )


class DemonstrationDataset:
    """Dataset of demonstration trajectories.

    Provides functionality for:
    - Loading and saving demonstrations
    - Converting to training data (state, action, next_state)
    - Data augmentation
    - Filtering and sampling

    Attributes:
        demonstrations: List of demonstrations.
    """

    def __init__(
        self,
        demonstrations: list[Demonstration] | None = None,
    ) -> None:
        """Initialize demonstration dataset.

        Args:
            demonstrations: Initial list of demonstrations.
        """
        self.demonstrations = demonstrations or []

    def __len__(self) -> int:
        """Return number of demonstrations."""
        return len(self.demonstrations)

    def __getitem__(self, idx: int) -> Demonstration:
        """Get demonstration by index."""
        return self.demonstrations[idx]

    def __iter__(self):
        """Iterate over demonstrations."""
        return iter(self.demonstrations)

    def add(self, demo: Demonstration) -> None:
        """Add demonstration to dataset.

        Args:
            demo: Demonstration to add.
        """
        self.demonstrations.append(demo)

    def extend(self, demos: list[Demonstration]) -> None:
        """Add multiple demonstrations.

        Args:
            demos: List of demonstrations to add.
        """
        self.demonstrations.extend(demos)

    def filter_successful(self) -> "DemonstrationDataset":
        """Return dataset with only successful demonstrations.

        Returns:
            Filtered dataset.
        """
        successful = [d for d in self.demonstrations if d.success]
        return DemonstrationDataset(successful)

    def filter_by_task(self, task_id: str) -> "DemonstrationDataset":
        """Return dataset with only demonstrations for a specific task.

        Args:
            task_id: Task identifier to filter by.

        Returns:
            Filtered dataset.
        """
        filtered = [d for d in self.demonstrations if d.task_id == task_id]
        return DemonstrationDataset(filtered)

    @property
    def total_frames(self) -> int:
        """Total number of frames across all demonstrations."""
        return sum(d.n_frames for d in self.demonstrations)

    @property
    def total_transitions(self) -> int:
        """Total number of state transitions (frames - 1 per demo)."""
        return sum(max(0, d.n_frames - 1) for d in self.demonstrations)

    def to_transitions(
        self,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """Convert dataset to (states, actions, next_states) for training.

        Returns:
            Tuple of (states, actions, next_states) arrays.
        """
        states = []
        actions = []
        next_states = []

        for demo in self.demonstrations:
            if demo.actions is None:
                continue

            n = demo.n_frames
            for i in range(n - 1):
                state = np.concatenate([
                    demo.joint_positions[i],
                    demo.joint_velocities[i],
                ])
                next_state = np.concatenate([
                    demo.joint_positions[i + 1],
                    demo.joint_velocities[i + 1],
                ])
                states.append(state)
                actions.append(demo.actions[i])
                next_states.append(next_state)

        return (
            np.array(states),
            np.array(actions),
            np.array(next_states),
        )

    def to_state_action_pairs(
        self,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Convert to (states, actions) pairs for behavior cloning.

        Returns:
            Tuple of (states, actions) arrays.
        """
        states, actions, _ = self.to_transitions()
        return states, actions

    def augment(
        self,
        noise_std: float = 0.01,
        num_augmentations: int = 5,
        rng: np.random.Generator | None = None,
    ) -> "DemonstrationDataset":
        """Augment demonstrations with noise.

        Args:
            noise_std: Standard deviation of Gaussian noise.
            num_augmentations: Number of augmented copies per demo.
            rng: Random number generator.

        Returns:
            Augmented dataset.
        """
        if rng is None:
            rng = np.random.default_rng()

        augmented = []
        for demo in self.demonstrations:
            # Keep original
            augmented.append(demo)

            # Add noisy copies
            for _ in range(num_augmentations):
                noisy_positions = demo.joint_positions + rng.normal(
                    0, noise_std, demo.joint_positions.shape
                )
                noisy_velocities = demo.joint_velocities + rng.normal(
                    0, noise_std, demo.joint_velocities.shape
                )
                noisy_demo = Demonstration(
                    timestamps=demo.timestamps.copy(),
                    joint_positions=noisy_positions,
                    joint_velocities=noisy_velocities,
                    actions=demo.actions.copy() if demo.actions is not None else None,
                    end_effector_poses=(
                        demo.end_effector_poses.copy()
                        if demo.end_effector_poses is not None
                        else None
                    ),
                    contact_states=demo.contact_states,
                    task_id=demo.task_id,
                    success=demo.success,
                    source=f"{demo.source}_augmented",
                    metadata={**demo.metadata, "augmented": True},
                )
                augmented.append(noisy_demo)

        return DemonstrationDataset(augmented)

    def save(self, path: str | Path) -> None:
        """Save dataset to disk.

        Args:
            path: Path to save file (JSON format).
        """
        path = Path(path)
        data = {
            "version": "1.0",
            "n_demonstrations": len(self.demonstrations),
            "demonstrations": [d.to_dict() for d in self.demonstrations],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "DemonstrationDataset":
        """Load dataset from disk.

        Args:
            path: Path to load file.

        Returns:
            Loaded dataset.
        """
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)

        demonstrations = [
            Demonstration.from_dict(d) for d in data["demonstrations"]
        ]
        return cls(demonstrations)

    def sample(
        self,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> "DemonstrationDataset":
        """Random sample of demonstrations.

        Args:
            n: Number of demonstrations to sample.
            rng: Random number generator.

        Returns:
            Sampled dataset.
        """
        if rng is None:
            rng = np.random.default_rng()

        n = min(n, len(self.demonstrations))
        indices = rng.choice(len(self.demonstrations), size=n, replace=False)
        sampled = [self.demonstrations[i] for i in indices]
        return DemonstrationDataset(sampled)

    def get_statistics(self) -> dict[str, Any]:
        """Compute dataset statistics.

        Returns:
            Dictionary with statistics.
        """
        if not self.demonstrations:
            return {"n_demonstrations": 0}

        all_positions = np.concatenate(
            [d.joint_positions for d in self.demonstrations]
        )
        all_velocities = np.concatenate(
            [d.joint_velocities for d in self.demonstrations]
        )

        return {
            "n_demonstrations": len(self.demonstrations),
            "total_frames": self.total_frames,
            "total_transitions": self.total_transitions,
            "success_rate": sum(d.success for d in self.demonstrations) / len(self),
            "mean_duration": np.mean([d.duration for d in self.demonstrations]),
            "position_mean": all_positions.mean(axis=0).tolist(),
            "position_std": all_positions.std(axis=0).tolist(),
            "velocity_mean": all_velocities.mean(axis=0).tolist(),
            "velocity_std": all_velocities.std(axis=0).tolist(),
        }
