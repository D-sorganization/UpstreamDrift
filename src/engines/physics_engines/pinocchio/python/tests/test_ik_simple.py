"""Simple verification test for PinkSolver."""

import numpy as np
import pytest

from src.shared.python.logging_config import get_logger, setup_logging

pin = pytest.importorskip("pinocchio")

try:
    from dtack.ik.pink_solver import PinkSolver
    from dtack.ik.tasks import create_frame_task, create_posture_task
except ImportError:
    pytest.skip("dtack.ik dependencies missing", allow_module_level=True)


logger = get_logger(__name__)


def test_ik_convergence() -> None:
    """Test that IK converges for a simple manipulator."""
    # 1. Build sample model
    model = pin.buildSampleModelManipulator()
    data = model.createData()
    # Mock visual/collision models as they aren't needed for pure IK math
    visual_model = pin.GeometryModel()
    collision_model = pin.GeometryModel()

    solver = PinkSolver(model, data, visual_model, collision_model)

    # 2. Setup initial state
    q_init = pin.neutral(model)

    # 3. Setup Target
    # Let's try to reach a point slightly away from neutral
    end_effector_frame = model.getFrameId("effector_body")

    # Target pose: Translate +0.1 in X
    pin.forwardKinematics(model, data, q_init)
    pin.updateFramePlacements(model, data)
    start_pose = data.oMf[end_effector_frame]

    target_pose = start_pose.copy()
    target_pose.translation[0] += 0.05  # Move 5cm

    # 4. Create Tasks
    task_frame = create_frame_task("effector_body")
    task_frame.set_target(target_pose)

    task_posture = create_posture_task(cost=1e-3, q_ref=q_init)

    tasks = [task_frame, task_posture]

    # 5. Loop
    dt = 1e-2
    q = q_init.copy()

    for i in range(500):
        q = solver.solve(q, tasks, dt)

        # Check error
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        current_pose = data.oMf[end_effector_frame]
        error = np.linalg.norm(current_pose.translation - target_pose.translation)

        if i % 10 == 0:
            logger.info(f"Step {i}: Error = {error:.4f}")

        if error < 1e-3:
            return

    final_error = np.linalg.norm(current_pose.translation - target_pose.translation)

    assert final_error < 1e-3, "IK did not converge"


if __name__ == "__main__":
    setup_logging()
    test_ik_convergence()
