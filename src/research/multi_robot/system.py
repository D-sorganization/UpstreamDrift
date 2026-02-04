"""Multi-robot system management."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.engines.protocols import PhysicsEngineProtocol


class TaskStatus(Enum):
    """Status of a task."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskType(Enum):
    """Type of robot task."""

    MOVE_TO = "move_to"
    PICK = "pick"
    PLACE = "place"
    MANIPULATE = "manipulate"
    INSPECT = "inspect"
    WAIT = "wait"


@dataclass
class Task:
    """A task to be executed by a robot.

    Attributes:
        task_id: Unique task identifier.
        task_type: Type of task.
        target_position: Target position for movement tasks.
        target_object: Target object for manipulation tasks.
        priority: Task priority (higher = more important).
        status: Current task status.
        assigned_robot: Robot assigned to this task.
        dependencies: Task IDs that must complete first.
        estimated_duration: Estimated time to complete.
    """

    task_id: str
    task_type: TaskType
    target_position: NDArray[np.floating] | None = None
    target_object: str | None = None
    priority: int = 0
    status: TaskStatus = TaskStatus.PENDING
    assigned_robot: str | None = None
    dependencies: list[str] = field(default_factory=list)
    estimated_duration: float = 0.0

    def is_ready(self, completed_tasks: set[str]) -> bool:
        """Check if task dependencies are satisfied.

        Args:
            completed_tasks: Set of completed task IDs.

        Returns:
            True if all dependencies are met.
        """
        return all(dep in completed_tasks for dep in self.dependencies)


class TaskCoordinator:
    """Coordinates task allocation among robots.

    Handles task scheduling, assignment, and monitoring
    for multi-robot systems.

    Attributes:
        tasks: Dictionary of tasks by ID.
        completed_tasks: Set of completed task IDs.
    """

    def __init__(self) -> None:
        """Initialize task coordinator."""
        self._tasks: dict[str, Task] = {}
        self._completed_tasks: set[str] = set()
        self._robot_tasks: dict[str, str] = {}  # robot_id -> task_id

    def add_task(self, task: Task) -> None:
        """Add a task to the queue.

        Args:
            task: Task to add.
        """
        self._tasks[task.task_id] = task

    def remove_task(self, task_id: str) -> bool:
        """Remove a task from the queue.

        Args:
            task_id: Task identifier.

        Returns:
            True if task was found and removed.
        """
        if task_id in self._tasks:
            del self._tasks[task_id]
            return True
        return False

    def get_ready_tasks(self) -> list[Task]:
        """Get tasks that are ready to be assigned.

        Returns:
            List of ready tasks sorted by priority.
        """
        ready = [
            task
            for task in self._tasks.values()
            if task.status == TaskStatus.PENDING
            and task.is_ready(self._completed_tasks)
        ]
        return sorted(ready, key=lambda t: -t.priority)

    def assign_task(self, task_id: str, robot_id: str) -> bool:
        """Assign a task to a robot.

        Args:
            task_id: Task identifier.
            robot_id: Robot identifier.

        Returns:
            True if assignment was successful.
        """
        if task_id not in self._tasks:
            return False

        task = self._tasks[task_id]
        if task.status != TaskStatus.PENDING:
            return False

        task.assigned_robot = robot_id
        task.status = TaskStatus.ASSIGNED
        self._robot_tasks[robot_id] = task_id
        return True

    def start_task(self, task_id: str) -> bool:
        """Mark a task as in progress.

        Args:
            task_id: Task identifier.

        Returns:
            True if status was updated.
        """
        if task_id not in self._tasks:
            return False

        task = self._tasks[task_id]
        if task.status != TaskStatus.ASSIGNED:
            return False

        task.status = TaskStatus.IN_PROGRESS
        return True

    def complete_task(self, task_id: str) -> bool:
        """Mark a task as completed.

        Args:
            task_id: Task identifier.

        Returns:
            True if status was updated.
        """
        if task_id not in self._tasks:
            return False

        task = self._tasks[task_id]
        task.status = TaskStatus.COMPLETED
        self._completed_tasks.add(task_id)

        if task.assigned_robot and task.assigned_robot in self._robot_tasks:
            del self._robot_tasks[task.assigned_robot]

        return True

    def fail_task(self, task_id: str) -> bool:
        """Mark a task as failed.

        Args:
            task_id: Task identifier.

        Returns:
            True if status was updated.
        """
        if task_id not in self._tasks:
            return False

        task = self._tasks[task_id]
        task.status = TaskStatus.FAILED

        if task.assigned_robot and task.assigned_robot in self._robot_tasks:
            del self._robot_tasks[task.assigned_robot]

        return True

    def get_robot_task(self, robot_id: str) -> Task | None:
        """Get the task assigned to a robot.

        Args:
            robot_id: Robot identifier.

        Returns:
            Assigned task or None.
        """
        task_id = self._robot_tasks.get(robot_id)
        if task_id:
            return self._tasks.get(task_id)
        return None

    def get_available_robots(self, all_robots: list[str]) -> list[str]:
        """Get robots not currently assigned to tasks.

        Args:
            all_robots: List of all robot IDs.

        Returns:
            List of available robot IDs.
        """
        return [r for r in all_robots if r not in self._robot_tasks]


class MultiRobotSystem:
    """Manage multiple robots in a shared environment.

    Provides unified interface for controlling and monitoring
    multiple robots, including collision checking between robots.

    Attributes:
        robots: Dictionary of robot engines by ID.
    """

    def __init__(self) -> None:
        """Initialize multi-robot system."""
        self._robots: dict[str, PhysicsEngineProtocol] = {}
        self._robot_poses: dict[str, NDArray[np.floating]] = {}
        self._coordinator = TaskCoordinator()

    @property
    def robots(self) -> dict[str, PhysicsEngineProtocol]:
        """Get robot dictionary."""
        return self._robots

    @property
    def n_robots(self) -> int:
        """Number of robots in the system."""
        return len(self._robots)

    def add_robot(
        self,
        robot_id: str,
        engine: PhysicsEngineProtocol,
        base_pose: NDArray[np.floating],
    ) -> None:
        """Add a robot to the system.

        Args:
            robot_id: Unique robot identifier.
            engine: Physics engine for this robot.
            base_pose: Initial base pose (7D: xyz + quaternion).
        """
        self._robots[robot_id] = engine
        self._robot_poses[robot_id] = base_pose.copy()

    def remove_robot(self, robot_id: str) -> bool:
        """Remove a robot from the system.

        Args:
            robot_id: Robot identifier.

        Returns:
            True if robot was found and removed.
        """
        if robot_id in self._robots:
            del self._robots[robot_id]
            del self._robot_poses[robot_id]
            return True
        return False

    def get_robot(self, robot_id: str) -> PhysicsEngineProtocol | None:
        """Get a robot by ID.

        Args:
            robot_id: Robot identifier.

        Returns:
            Robot engine or None.
        """
        return self._robots.get(robot_id)

    def get_robot_pose(self, robot_id: str) -> NDArray[np.floating] | None:
        """Get a robot's base pose.

        Args:
            robot_id: Robot identifier.

        Returns:
            Base pose or None.
        """
        return self._robot_poses.get(robot_id)

    def set_robot_pose(
        self,
        robot_id: str,
        pose: NDArray[np.floating],
    ) -> None:
        """Set a robot's base pose.

        Args:
            robot_id: Robot identifier.
            pose: New base pose.
        """
        if robot_id in self._robot_poses:
            self._robot_poses[robot_id] = pose.copy()

    def step_all(self, dt: float) -> None:
        """Step all robots synchronously.

        Args:
            dt: Timestep.
        """
        for engine in self._robots.values():
            if hasattr(engine, "step"):
                engine.step(dt)

    def check_inter_robot_collision(
        self,
        safety_distance: float = 0.2,
    ) -> list[tuple[str, str]]:
        """Check for collisions between robots.

        Uses simple bounding sphere collision check.

        Args:
            safety_distance: Minimum distance between robots.

        Returns:
            List of colliding robot pairs.
        """
        collisions = []
        robot_ids = list(self._robots.keys())

        for i, id1 in enumerate(robot_ids):
            for id2 in robot_ids[i + 1 :]:
                pos1 = self._robot_poses[id1][:3]
                pos2 = self._robot_poses[id2][:3]
                distance = np.linalg.norm(pos1 - pos2)

                if distance < safety_distance:
                    collisions.append((id1, id2))

        return collisions

    def allocate_tasks(
        self,
        tasks: list[Task],
    ) -> dict[str, list[Task]]:
        """Allocate tasks to robots.

        Uses greedy allocation based on distance and priority.

        Args:
            tasks: List of tasks to allocate.

        Returns:
            Dictionary mapping robot IDs to assigned tasks.
        """
        allocation: dict[str, list[Task]] = {robot_id: [] for robot_id in self._robots}

        # Add tasks to coordinator
        for task in tasks:
            self._coordinator.add_task(task)

        # Allocate ready tasks to available robots
        while True:
            ready_tasks = self._coordinator.get_ready_tasks()
            available_robots = self._coordinator.get_available_robots(
                list(self._robots.keys())
            )

            if not ready_tasks or not available_robots:
                break

            # Assign highest priority task to closest robot
            task = ready_tasks[0]

            best_robot = None
            best_distance = float("inf")

            for robot_id in available_robots:
                if task.target_position is not None:
                    robot_pos = self._robot_poses[robot_id][:3]
                    distance = np.linalg.norm(task.target_position - robot_pos)
                    if distance < best_distance:
                        best_distance = distance
                        best_robot = robot_id
                else:
                    best_robot = robot_id
                    break

            if best_robot:
                self._coordinator.assign_task(task.task_id, best_robot)
                allocation[best_robot].append(task)

        return allocation

    def get_system_state(self) -> dict[str, Any]:
        """Get state of the entire system.

        Returns:
            Dictionary with system state.
        """
        state = {
            "n_robots": self.n_robots,
            "robot_states": {},
        }

        for robot_id, engine in self._robots.items():
            robot_state = {
                "base_pose": self._robot_poses[robot_id].tolist(),
            }

            if hasattr(engine, "get_joint_positions"):
                robot_state["joint_positions"] = engine.get_joint_positions().tolist()

            if hasattr(engine, "get_joint_velocities"):
                robot_state["joint_velocities"] = engine.get_joint_velocities().tolist()

            state["robot_states"][robot_id] = robot_state

        return state
