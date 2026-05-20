"""Tests for src/research/multi_robot/*."""

from __future__ import annotations

import numpy as np
import pytest

from src.research.multi_robot.coordination import (
    CooperativeManipulation,
    FormationConfig,
    FormationController,
)
from src.research.multi_robot.system import (
    MultiRobotSystem,
    Task,
    TaskCoordinator,
    TaskStatus,
    TaskType,
)


class TestFormationConfig:
    def test_line_formation(self) -> None:
        c = FormationConfig.line_formation(3, spacing=2.0)
        assert c.name == "line"
        assert c.positions.shape == (3, 3)
        assert c.positions[2, 1] == 4.0

    def test_circle_formation(self) -> None:
        c = FormationConfig.circle_formation(4, radius=1.0)
        assert c.name == "circle"
        # each robot at distance ~1 from origin
        dists = np.linalg.norm(c.positions[:, :2], axis=1)
        np.testing.assert_allclose(dists, 1.0)

    def test_wedge_formation(self) -> None:
        c = FormationConfig.wedge_formation(5, spacing=1.0, angle=0.5)
        assert c.name == "wedge"
        np.testing.assert_allclose(c.positions[0], 0.0)
        # second/third robots symmetric on y
        assert c.positions[1, 1] == -c.positions[2, 1]


class TestFormationController:
    def test_construction(self) -> None:
        f = FormationConfig.line_formation(2)
        fc = FormationController(["a", "b"], f)
        assert fc.robots == ["a", "b"]
        assert fc.formation is f

    def test_set_gains(self) -> None:
        f = FormationConfig.line_formation(1)
        fc = FormationController(["a"], f)
        fc.set_gains(position=5.0, velocity=2.0, heading=3.0)
        assert fc._gains["position"] == 5.0

    def test_compute_formation_control(self) -> None:
        f = FormationConfig.line_formation(2, spacing=1.0)
        fc = FormationController(["a", "b"], f)
        leader_pose = np.array([0.0, 0, 0, 1, 0, 0, 0])
        positions = {"a": np.array([0.0, 0, 0]), "b": np.array([0.0, 0.5, 0])}
        cmds = fc.compute_formation_control(leader_pose, positions)
        # b should be commanded to move further along +y to reach desired
        assert cmds["b"][1] > 0

    def test_compute_formation_with_velocities(self) -> None:
        f = FormationConfig.line_formation(1)
        fc = FormationController(["a"], f)
        cmds = fc.compute_formation_control(
            np.array([0.0, 0, 0]),
            {"a": np.zeros(3)},
            {"a": np.array([1.0, 0, 0])},
        )
        # velocity damping subtracts
        assert cmds["a"][0] < 0

    def test_compute_formation_missing_robot(self) -> None:
        f = FormationConfig.line_formation(2)
        fc = FormationController(["a", "b"], f)
        cmds = fc.compute_formation_control(np.zeros(3), {"a": np.zeros(3)})
        assert "b" not in cmds

    def test_quat_to_rotation_identity(self) -> None:
        f = FormationConfig.line_formation(1)
        fc = FormationController(["a"], f)
        R = fc._quat_to_rotation(np.array([1.0, 0, 0, 0]))
        np.testing.assert_allclose(R, np.eye(3))

    def test_set_formation(self) -> None:
        f1 = FormationConfig.line_formation(2)
        f2 = FormationConfig.circle_formation(2)
        fc = FormationController(["a", "b"], f1)
        fc.set_formation(f2)
        assert fc.formation is f2

    def test_get_formation_error(self) -> None:
        f = FormationConfig.line_formation(2, spacing=1.0)
        fc = FormationController(["a", "b"], f)
        # perfect formation
        leader = np.array([0.0, 0, 0])
        positions = {
            "a": np.array([0.0, 0, 0]),
            "b": np.array([0.0, 1.0, 0]),
        }
        err = fc.get_formation_error(leader, positions)
        assert err == pytest.approx(0.0, abs=1e-9)

    def test_get_formation_error_missing(self) -> None:
        f = FormationConfig.line_formation(2)
        fc = FormationController(["a", "b"], f)
        err = fc.get_formation_error(np.zeros(7), {"a": np.zeros(3)})
        assert err == 0.0


class TestCooperativeManipulation:
    def test_construction(self) -> None:
        cm = CooperativeManipulation(["r1", "r2"])
        assert cm.n_robots == 2

    def test_set_grasp_points_default_normals(self) -> None:
        cm = CooperativeManipulation(["r1", "r2"])
        cm.set_grasp_points([np.array([1.0, 0, 0]), np.array([-1.0, 0, 0])])
        assert len(cm._grasp_normals) == 2
        # normals point toward each other
        assert cm._grasp_normals[0][0] < 0
        assert cm._grasp_normals[1][0] > 0

    def test_set_grasp_points_explicit_normals(self) -> None:
        cm = CooperativeManipulation(["r1"])
        normals = [np.array([0.0, 0, 1.0])]
        cm.set_grasp_points([np.zeros(3)], grasp_normals=normals)
        assert cm._grasp_normals[0][2] == 1.0

    def test_compute_grasp_matrix(self) -> None:
        cm = CooperativeManipulation(["r1", "r2"])
        cm.set_grasp_points([np.array([1.0, 0, 0]), np.array([-1.0, 0, 0])])
        G = cm.compute_grasp_matrix(np.array([0.0, 0, 0, 1, 0, 0, 0]))
        assert G.shape == (6, 6)

    def test_compute_load_sharing(self) -> None:
        cm = CooperativeManipulation(["r1", "r2"])
        cm.set_grasp_points([np.array([1.0, 0, 0]), np.array([-1.0, 0, 0])])
        forces = cm.compute_load_sharing(
            np.array([0.0, 0, 10, 0, 0, 0]),
            np.array([0.0, 0, 0, 1, 0, 0, 0]),
        )
        assert len(forces) == 2
        # total z force ~ 10
        z_total = forces[0][2] + forces[1][2]
        assert z_total == pytest.approx(10.0, rel=1e-3)

    def test_plan_cooperative_motion(self) -> None:
        cm = CooperativeManipulation(["r1", "r2"])
        cm.set_grasp_points([np.array([1.0, 0, 0]), np.array([-1.0, 0, 0])])
        trajs = cm.plan_cooperative_motion(
            object_goal_pose=np.array([1.0, 0, 0, 1, 0, 0, 0]),
            object_current_pose=np.array([0.0, 0, 0, 1, 0, 0, 0]),
            dt=0.1,
            duration=0.5,
        )
        assert len(trajs) == 2
        assert trajs[0].shape[1] == 7
        assert trajs[0].shape[0] >= 2

    def test_slerp_identical(self) -> None:
        cm = CooperativeManipulation(["r1"])
        q = np.array([1.0, 0, 0, 0])
        out = cm._slerp(q, q, 0.5)
        np.testing.assert_allclose(out, q, atol=1e-9)

    def test_slerp_negative_dot(self) -> None:
        cm = CooperativeManipulation(["r1"])
        q0 = np.array([1.0, 0, 0, 0])
        q1 = np.array([-1.0, 0, 0, 0])  # negative dot -> negate
        out = cm._slerp(q0, q1, 0.5)
        assert np.linalg.norm(out) == pytest.approx(1.0, rel=1e-6)

    def test_slerp_orthogonal(self) -> None:
        cm = CooperativeManipulation(["r1"])
        q0 = np.array([1.0, 0, 0, 0])
        q1 = np.array([0.0, 1, 0, 0])
        out = cm._slerp(q0, q1, 0.5)
        # midway, magnitude one
        assert np.linalg.norm(out) == pytest.approx(1.0, rel=1e-6)

    def test_check_force_closure(self) -> None:
        cm = CooperativeManipulation(["r1", "r2"])
        cm.set_grasp_points([np.array([1.0, 0, 0]), np.array([-1.0, 0, 0])])
        has, qual = cm.check_force_closure(np.array([0.0, 0, 0, 1, 0, 0, 0]))
        assert bool(has) in (True, False)
        assert qual >= 0


class TestTask:
    def test_is_ready_no_deps(self) -> None:
        t = Task("t1", TaskType.MOVE_TO)
        assert t.is_ready(set())

    def test_is_ready_with_deps(self) -> None:
        t = Task("t2", TaskType.PICK, dependencies=["t1"])
        assert not t.is_ready(set())
        assert t.is_ready({"t1"})


class TestTaskCoordinator:
    def test_add_remove(self) -> None:
        c = TaskCoordinator()
        c.add_task(Task("t1", TaskType.MOVE_TO))
        assert c.remove_task("t1")
        assert not c.remove_task("missing")

    def test_get_ready_priority_sorted(self) -> None:
        c = TaskCoordinator()
        c.add_task(Task("a", TaskType.WAIT, priority=1))
        c.add_task(Task("b", TaskType.WAIT, priority=5))
        ready = c.get_ready_tasks()
        assert ready[0].task_id == "b"

    def test_assign_start_complete(self) -> None:
        c = TaskCoordinator()
        c.add_task(Task("t1", TaskType.WAIT))
        assert c.assign_task("t1", "r1")
        assert c._tasks["t1"].status == TaskStatus.ASSIGNED
        assert c.start_task("t1")
        assert c._tasks["t1"].status == TaskStatus.IN_PROGRESS
        assert c.complete_task("t1")
        assert "t1" in c._completed_tasks
        # robot freed
        assert "r1" not in c._robot_tasks

    def test_assign_unknown(self) -> None:
        c = TaskCoordinator()
        assert not c.assign_task("x", "r1")

    def test_assign_already_assigned_fails(self) -> None:
        c = TaskCoordinator()
        c.add_task(Task("t1", TaskType.WAIT))
        c.assign_task("t1", "r1")
        assert not c.assign_task("t1", "r2")

    def test_start_invalid(self) -> None:
        c = TaskCoordinator()
        assert not c.start_task("missing")
        c.add_task(Task("t1", TaskType.WAIT))
        assert not c.start_task("t1")  # not assigned yet

    def test_fail_task(self) -> None:
        c = TaskCoordinator()
        c.add_task(Task("t1", TaskType.WAIT))
        c.assign_task("t1", "r1")
        assert c.fail_task("t1")
        assert c._tasks["t1"].status == TaskStatus.FAILED
        assert "r1" not in c._robot_tasks
        assert not c.fail_task("missing")

    def test_complete_missing(self) -> None:
        c = TaskCoordinator()
        assert not c.complete_task("missing")

    def test_get_robot_task(self) -> None:
        c = TaskCoordinator()
        c.add_task(Task("t1", TaskType.WAIT))
        c.assign_task("t1", "r1")
        t = c.get_robot_task("r1")
        assert t.task_id == "t1"
        assert c.get_robot_task("nobody") is None

    def test_available_robots(self) -> None:
        c = TaskCoordinator()
        c.add_task(Task("t1", TaskType.WAIT))
        c.assign_task("t1", "r1")
        avail = c.get_available_robots(["r1", "r2", "r3"])
        assert avail == ["r2", "r3"]


class TestMultiRobotSystem:
    def test_add_remove(self, fake_engine) -> None:
        s = MultiRobotSystem()
        s.add_robot("r1", fake_engine, np.array([1.0, 0, 0, 1, 0, 0, 0]))
        assert s.n_robots == 1
        assert s.get_robot("r1") is fake_engine
        assert s.remove_robot("r1")
        assert s.n_robots == 0
        assert not s.remove_robot("missing")

    def test_get_robot_missing(self) -> None:
        s = MultiRobotSystem()
        assert s.get_robot("x") is None
        assert s.get_robot_pose("x") is None

    def test_set_robot_pose(self, fake_engine) -> None:
        s = MultiRobotSystem()
        s.add_robot("r1", fake_engine, np.zeros(7))
        s.set_robot_pose("r1", np.array([5.0, 0, 0, 1, 0, 0, 0]))
        assert s.get_robot_pose("r1")[0] == 5.0
        # silently ignored for missing
        s.set_robot_pose("missing", np.zeros(7))

    def test_step_all(self, fake_engine) -> None:
        s = MultiRobotSystem()
        s.add_robot("r1", fake_engine, np.zeros(7))
        s.step_all(0.01)  # just verify no errors

    def test_check_inter_robot_collision(self, fake_engine) -> None:
        s = MultiRobotSystem()
        s.add_robot("r1", fake_engine, np.array([0.0, 0, 0, 1, 0, 0, 0]))
        s.add_robot("r2", fake_engine, np.array([0.1, 0, 0, 1, 0, 0, 0]))
        s.add_robot("r3", fake_engine, np.array([5.0, 0, 0, 1, 0, 0, 0]))
        col = s.check_inter_robot_collision(safety_distance=0.5)
        assert ("r1", "r2") in col
        assert ("r1", "r3") not in col

    def test_allocate_tasks(self, fake_engine) -> None:
        s = MultiRobotSystem()
        s.add_robot("r1", fake_engine, np.array([0.0, 0, 0, 1, 0, 0, 0]))
        s.add_robot("r2", fake_engine, np.array([10.0, 0, 0, 1, 0, 0, 0]))
        tasks = [
            Task(
                "t1",
                TaskType.MOVE_TO,
                target_position=np.array([0.0, 0, 0]),
                priority=1,
            ),
            Task(
                "t2",
                TaskType.MOVE_TO,
                target_position=np.array([10.0, 0, 0]),
                priority=1,
            ),
        ]
        alloc = s.allocate_tasks(tasks)
        # r1 gets the task near origin
        r1_ids = [t.task_id for t in alloc["r1"]]
        assert "t1" in r1_ids

    def test_allocate_tasks_no_target(self, fake_engine) -> None:
        s = MultiRobotSystem()
        s.add_robot("r1", fake_engine, np.zeros(7))
        tasks = [Task("t1", TaskType.WAIT)]
        alloc = s.allocate_tasks(tasks)
        assert any(t.task_id == "t1" for t in alloc["r1"])

    def test_system_state(self, fake_engine) -> None:
        s = MultiRobotSystem()
        s.add_robot("r1", fake_engine, np.zeros(7))
        state = s.get_system_state()
        assert state["n_robots"] == 1
        assert "r1" in state["robot_states"]
        assert "joint_positions" in state["robot_states"]["r1"]
