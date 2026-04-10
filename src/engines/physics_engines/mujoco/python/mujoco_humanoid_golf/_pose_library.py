from __future__ import annotations

import json
import time
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class StoredPose:
    name: str
    qpos: np.ndarray
    qvel: np.ndarray | None = None
    timestamp: float = 0.0
    description: str = ""


class PoseLibraryMixin:
    pose_library: dict[str, StoredPose]

    def save_pose(self, name: str, description: str = "") -> StoredPose:
        if not (name is not None):
            raise ValueError("name must be provided")
        pose = StoredPose(
            name=name,
            qpos=self.data.qpos.copy(),
            qvel=self.data.qvel.copy(),
            timestamp=time.time(),
            description=description,
        )

        self.pose_library[name] = pose
        return pose

    def load_pose(self, name: str, apply_velocities: bool = False) -> bool:
        if not (name is not None):
            raise ValueError("name must be provided")
        if name not in self.pose_library:
            return False

        pose = self.pose_library[name]
        self.data.qpos[:] = pose.qpos.copy()

        if apply_velocities and pose.qvel is not None:
            self.data.qvel[:] = pose.qvel.copy()
        else:
            self.data.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self.data)
        return True

    def delete_pose(self, name: str) -> bool:
        if not (name is not None):
            raise ValueError("name must be provided")
        if name in self.pose_library:
            del self.pose_library[name]
            return True
        return False

    def interpolate_poses(
        self,
        pose_name_a: str,
        pose_name_b: str,
        alpha: float,
    ) -> bool:
        if not (pose_name_a is not None):
            raise ValueError("pose_name_a must be provided")
        if pose_name_a not in self.pose_library or pose_name_b not in self.pose_library:
            return False

        pose_a = self.pose_library[pose_name_a]
        pose_b = self.pose_library[pose_name_b]

        alpha = np.clip(alpha, 0.0, 1.0)
        interpolated_qpos = (1 - alpha) * pose_a.qpos + alpha * pose_b.qpos

        self.data.qpos[:] = interpolated_qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        return True

    def export_pose_library(self, filepath: str) -> None:
        if not (filepath is not None):
            raise ValueError("filepath must be provided")
        data = {}
        for name, pose in self.pose_library.items():
            data[name] = {
                "qpos": pose.qpos.tolist(),
                "qvel": pose.qvel.tolist() if pose.qvel is not None else None,
                "timestamp": pose.timestamp,
                "description": pose.description,
            }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def import_pose_library(self, filepath: str) -> int:
        try:
            with open(filepath) as f:
                data = json.load(f)

            count = 0
            for name, pose_data in data.items():
                pose = StoredPose(
                    name=name,
                    qpos=np.array(pose_data["qpos"]),
                    qvel=np.array(pose_data["qvel"]) if pose_data.get("qvel") else None,
                    timestamp=pose_data.get("timestamp", 0.0),
                    description=pose_data.get("description", ""),
                )
                self.pose_library[name] = pose
                count += 1

            return count
        except (FileNotFoundError, PermissionError, OSError):
            return 0

    def list_poses(self) -> list[str]:
        return list(self.pose_library.keys())
