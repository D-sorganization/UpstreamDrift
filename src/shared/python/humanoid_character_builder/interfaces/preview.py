"""Simulation and preview helpers for humanoid models."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def run_simulation(urdf_xml: str, duration: float = 1.0) -> bool:
    """Run a short simulation to verify physics stability."""
    try:
        import mujoco
    except ImportError:
        logger.warning("MuJoCo not installed. Simulation skipped.")
        return False

    try:
        model = mujoco.MjModel.from_xml_string(urdf_xml)
        data = mujoco.MjData(model)

        steps = int(duration / model.opt.timestep)
        for _ in range(steps):
            mujoco.mj_step(model, data)
            if np.isnan(data.qpos).any() or np.isnan(data.qvel).any():
                logger.error("Simulation instability detected (NaN values).")
                return False
        return True

    except (RuntimeError, ValueError, OSError) as e:
        logger.error(f"Simulation failed: {e}")
        return False


def run_preview(urdf_xml: str, animate: bool = False) -> None:
    """Open visual preview of the character."""
    try:
        import mujoco
        import mujoco.viewer
    except ImportError:
        logger.warning("MuJoCo viewer not available.")
        return

    try:
        model = mujoco.MjModel.from_xml_string(urdf_xml)
        data = mujoco.MjData(model)

        with mujoco.viewer.launch_passive(model, data) as viewer:
            import time

            start = time.time()
            while viewer.is_running():
                step_start = time.time()

                if animate and model.nu > 0:
                    t = time.time() - start
                    data.ctrl[:] = 0.5 * np.sin(t * 2.0)

                mujoco.mj_step(model, data)
                viewer.sync()

                dt = model.opt.timestep
                elapsed = time.time() - step_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)

    except (RuntimeError, ValueError, OSError) as e:
        logger.error(f"Preview failed: {e}")
