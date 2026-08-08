"""The pipeline's default IK backend must actually solve (epic #8390, A1/#8391).

Regression guard: the orchestrator previously defaulted to
``ik_backend="mujoco"``, a backend whose ``solve_frame`` unconditionally
raises ``NotImplementedError`` (#7046). The default configuration of the
IK stage must always point at a working solver.
"""

from __future__ import annotations

from src.shared.python.motion_pipeline.contracts import (
    Marker,
    MarkerFrame,
    MarkerTrajectory,
)
from src.shared.python.motion_pipeline.ik.base import make_ik_solver
from src.shared.python.motion_pipeline.orchestrator import (
    AdapterOverride,
    PipelineConfig,
)

from ._local_fixtures import make_3dof_phantom_rig


def test_default_ik_backend_is_geometric() -> None:
    cfg = PipelineConfig(adapter=AdapterOverride(format="c3d"))
    assert cfg.ik_backend == "geometric"


def test_default_ik_backend_solves_without_not_implemented() -> None:
    """A default-configured pipeline must produce joint angles, not raise."""
    cfg = PipelineConfig(adapter=AdapterOverride(format="c3d"))
    solver = make_ik_solver(cfg.ik_backend)

    rig = make_3dof_phantom_rig()
    # Markers named after rig joints so the geometric solver has targets.
    frames = [
        MarkerFrame(
            timestamp=i / 100.0,
            frame_index=i,
            markers={"link1": Marker(name="link1", x=0.4, y=0.1 * i, z=0.1)},
        )
        for i in range(2)
    ]
    traj = MarkerTrajectory(id="default_backend_traj", frames=frames)

    result = solver.solve(traj, rig)
    assert result is not None
    assert len(result.frames) == 2
    assert all(len(frame.q) == rig.num_dofs for frame in result.frames)
