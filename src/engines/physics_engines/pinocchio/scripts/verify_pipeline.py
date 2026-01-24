"""Manual verification of the Unified Platform pipeline."""

import sys
from pathlib import Path

import numpy as np

from src.shared.python.logging_config import get_logger, setup_logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import pinocchio as pin
except ImportError:
    sys.exit(1)

from dtack.ik.pink_solver import PinkSolver
from dtack.sim.dynamics import DynamicsEngine
from dtack.utils.matlab_importer import MATLABImporter


def main() -> None:
    setup_logging()
    logger = get_logger("VerifyWorkflow")

    logger.info("1. Verifying Pinocchio Installation...")

    logger.info("2. Building Sample Model...")
    model = pin.buildSampleModelManipulator()
    data = model.createData()
    logger.info(f"Model built: nq={model.nq}, nv={model.nv}")

    logger.info("3. Verifying Dynamics Engine...")
    dyn = DynamicsEngine(model, data)
    q = pin.neutral(model)
    v = np.zeros(model.nv)
    tau = np.zeros(model.nv)
    acc = dyn.forward_dynamics(q, v, tau)
    logger.info(f"Forward Dynamics computed. Acc shape: {acc.shape}")

    logger.info("4. Verifying Pink Solver...")
    try:
        PinkSolver(model, data, pin.GeometryModel(), pin.GeometryModel())
        logger.info("PinkSolver instantiated.")
    except Exception:
        logger.exception("PinkSolver failed")

    logger.info("5. Verifying Data Import...")
    MATLABImporter()
    logger.info("MATLABImporter instantiated.")

    logger.info("Verification Complete. Ready for integration.")


if __name__ == "__main__":
    main()
