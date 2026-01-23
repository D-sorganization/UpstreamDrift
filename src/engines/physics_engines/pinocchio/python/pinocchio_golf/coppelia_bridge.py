#!/usr/bin/env python3
"""
CoppeliaSim ↔ Pinocchio ↔ Meshcat sync layer.
Continuously pulls joint states from CoppeliaSim and displays the corresponding
configuration in Pinocchio's Meshcat visualizer.
"""

import logging
import time

import numpy as np
import numpy.typing as npt
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
from zmqRemoteApi import RemoteAPIClient

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# === CONFIG ===========================================================

URDF_PATH = "/path/to/your/robot.urdf"
MESH_DIR = "/path/to/meshes"  # can be "" if meshes are in URDF paths

# joint name mapping: Coppelia joint handle -> Pinocchio joint index
# Simplest case: same names in both worlds; you can adapt this as needed.
COPPELIA_JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    # ...
]


# === SETUP PINOCCHIO + MESHCAT =======================================

logger.info("Loading Pinocchio model...")
model, collision_model, visual_model = pin.buildModelsFromUrdf(URDF_PATH, MESH_DIR)
data = model.createData()

viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer(open=True)
viz.loadViewerModel()
q = pin.neutral(model)
viz.display(q)

logger.info("Model loaded: nq=%s, nv=%s", model.nq, model.nv)


# Map CoppeliaSim joint names to handles and to Pinocchio indices.
def build_joint_mapping(sim: object, model: object) -> tuple[list[int], list[int]]:
    """Map CoppeliaSim joint names to handles and Pinocchio indices.

    Args:
        sim: CoppeliaSim simulation object
        model: Pinocchio model object

    Returns:
        Tuple of (handles, pin_indices):
            handles: list of joint handles in same order as Pinocchio velocity indices.
            pin_indices: list of indices into q/nv to update.
    """
    handles: list[int] = []
    pin_indices: list[int] = []

    for name in COPPELIA_JOINT_NAMES:
        h = sim.getObject(name)  # type: ignore[attr-defined]
        if h == -1:
            msg = f"Coppelia joint '{name}' not found"
            raise RuntimeError(msg)

        # For simplicity, assume every joint corresponds to one DoF
        # and that order of COPPELIA_JOINT_NAMES matches pinocchio's joint order
        # (excluding the universe joint). You can make this more robust as needed.
        joint_id = model.getJointId(name)  # type: ignore[attr-defined]
        if joint_id == 0:
            msg = f"Pinocchio joint '{name}' not found"
            raise RuntimeError(msg)

        # index in q for this joint's first DoF (for revolute: one DoF)
        idx_q = model.idx_qs[joint_id]  # type: ignore[attr-defined]  # starting index in q
        handles.append(h)
        pin_indices.append(idx_q)

    return handles, pin_indices


# === SETUP COPPELIASIM ===============================================

logger.info("Connecting to CoppeliaSim...")
client = RemoteAPIClient()
sim = client.getObject("sim")

sim.stopSimulation()
time.sleep(0.5)
sim.startSimulation()

handles, pin_indices = build_joint_mapping(sim, model)
logger.info("Joint mapping established.")


# === MAIN LOOP =======================================================


def read_joint_positions(sim: object, handles: list[int]) -> npt.NDArray[np.float64]:
    """Read joint positions from CoppeliaSim.

    Args:
        sim: CoppeliaSim simulation object
        handles: List of joint handles

    Returns:
        Array of joint positions in radians
    """
    q_joint: list[float] = []
    for h in handles:
        # For revolute joints, position is angle (rad)
        pos = sim.getJointPosition(h)  # type: ignore[attr-defined]
        q_joint.append(pos)
    return np.array(q_joint)


try:
    while True:
        if sim.getSimulationState() != sim.simulation_stopped:
            joint_positions = read_joint_positions(sim, handles)

            # Build full q; assume base is fixed and q = [q0; joint_positions]
            q = pin.neutral(model)
            for local_idx, q_idx in enumerate(pin_indices):
                q[q_idx] = joint_positions[local_idx]

            viz.display(q)

        time.sleep(0.02)  # ~50 Hz update

except KeyboardInterrupt:
    logger.info("Stopping simulation...")
    sim.stopSimulation()
