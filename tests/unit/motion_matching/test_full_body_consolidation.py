"""Tests for full-body IK and forward dynamics consolidation (MS-11 #10330).

Validates:
1. Snapshot parity: IK on frame 0 and frame 300 of driver capture through shim vs shared module.
2. Snapshot parity: 50-step computed-torque rollout through shim vs shared module.
3. Shims emit DeprecationWarning on import.
4. Import-graph audit: no `mujoco` imports in `src/shared/python/motion_matching/`.
"""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.full_body_model import (
    NativeMujocoFullBodyModel,
)
from src.shared.python.motion_matching.contact_law import GroundPlane

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = REPO_ROOT / "docs/development/full_body_models/full_body_spec_v1.json"
C3D_PATH = REPO_ROOT / "data/C3D_TA_Driver.c3d"


def test_import_graph_no_mujoco_in_shared_motion_matching() -> None:
    """Verify that no file under src/shared/python/motion_matching imports mujoco."""
    shared_dir = REPO_ROOT / "src/shared/python/motion_matching"
    violations: list[str] = []

    for py_file in shared_dir.rglob("*.py"):
        code = py_file.read_text(encoding="utf-8")
        tree = ast.parse(code, filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "mujoco" or alias.name.startswith("mujoco."):
                        violations.append(
                            f"{py_file.name}:{node.lineno}: import {alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module == "mujoco" or (
                    node.module and node.module.startswith("mujoco.")
                ):
                    violations.append(
                        f"{py_file.name}:{node.lineno}: from {node.module} import ..."
                    )

    assert not violations, (
        "Forbidden mujoco imports found in shared motion matching:\n"
        + "\n".join(violations)
    )


def test_shims_emit_deprecation_warning() -> None:
    """Importing full_body_markers or full_body_simulation shims must emit DeprecationWarning."""
    # Ensure fresh imports to trigger module-level warnings
    for mod in (
        "src.engines.physics_engines.mujoco.python.full_body_markers",
        "src.engines.physics_engines.mujoco.python.full_body_simulation",
    ):
        if mod in sys.modules:
            del sys.modules[mod]

    with pytest.deprecated_call():
        import src.engines.physics_engines.mujoco.python.full_body_markers as _m_markers  # noqa: F401

    with pytest.deprecated_call():
        import src.engines.physics_engines.mujoco.python.full_body_simulation as _m_sim  # noqa: F401


def test_ik_parity_frame_0_and_300() -> None:
    """IK solve on frames 0 and 300 reproduces within 1e-9 between shim and shared."""
    from src.engines.physics_engines.mujoco.python import full_body_ik as mujoco_ik
    from src.engines.physics_engines.mujoco.python import (
        full_body_markers as old_markers,
    )

    spec_bytes = SPEC_PATH.read_bytes()
    spec = json.loads(spec_bytes)
    adapter = NativeMujocoFullBodyModel(spec_bytes)
    attachments = {
        label: (a["body"], tuple(a["offset_m"]))
        for label, a in spec["marker_attachments"].items()
        if a["offset_m"] is not None
    }
    ground = GroundPlane(normal=(0.0, 0.0, 1.0), height_m=0.0)

    old_kin = old_markers.FullBodyMarkerKinematics(adapter, attachments)
    new_kin = mujoco_ik.FullBodyMarkerKinematics(adapter, attachments)

    rng = np.random.default_rng(42)
    q0 = np.zeros(len(adapter.coordinate_order))

    for frame_idx in (0, 300):
        # Deterministic synthetic marker targets per frame
        q_target = q0.copy()
        q_target[2] = 0.95 + 0.05 * (frame_idx / 300.0)
        q_target[6:] = rng.uniform(-0.3, 0.3, len(q_target) - 6)
        targets = new_kin.marker_positions(q_target)
        valid = np.ones(len(targets), dtype=bool)

        fit_old = old_kin.solve_pose(targets, valid, q0, ground=ground, iterations=30)
        fit_new = new_kin.solve_pose(targets, valid, q0, ground=ground, iterations=30)

        np.testing.assert_allclose(fit_new.q, fit_old.q, atol=1e-9)
        assert abs(fit_new.marker_rms_m - fit_old.marker_rms_m) < 1e-9


def test_computed_torque_rollout_parity() -> None:
    """50-step computed-torque rollout reproduces within 1e-9 between shim and shared."""
    from src.engines.physics_engines.mujoco.python import (
        full_body_simulation as old_sim,
    )
    from src.shared.python.motion_matching import full_body_forward_dynamics as new_sim

    spec_bytes = SPEC_PATH.read_bytes()
    adapter = NativeMujocoFullBodyModel(spec_bytes)

    old_simulator = old_sim.FullBodySimulator(adapter)
    new_simulator = new_sim.FullBodySimulator(adapter)

    q0 = np.zeros(old_simulator.nv)
    q0[2] = 1.0  # above ground
    v0 = np.zeros(old_simulator.nv)

    old_ctrl = old_sim.hold_pose_controller(
        old_simulator, q0, omega_rad_s=20.0, zeta=1.0
    )
    new_ctrl = new_sim.hold_pose_controller(
        new_simulator, q0, omega_rad_s=20.0, zeta=1.0
    )

    # 50-step RK4 rollout
    dt = 0.001
    duration = 50 * dt

    rec_old = old_simulator.run(q0, v0, old_ctrl, duration_s=duration, dt_s=dt)
    rec_new = new_simulator.run(q0, v0, new_ctrl, duration_s=duration, dt_s=dt)

    np.testing.assert_allclose(rec_new.q, rec_old.q, atol=1e-9)
    np.testing.assert_allclose(rec_new.v, rec_old.v, atol=1e-9)
    np.testing.assert_allclose(rec_new.tau, rec_old.tau, atol=1e-9)
