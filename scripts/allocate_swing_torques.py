"""CLI utility to allocate contact forces and actuator torques across multi-physics engines.

Usage:
    python scripts/allocate_swing_torques.py \
        --engine mujoco \
        --candidate evidence/matched/driver_full_pinocchio/candidate.npz \
        --objective minimum_effort \
        --out evidence/matched/driver_full_mujoco_torques.npz
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from src.shared.python.motion_matching.contact_force_allocator import (
    AllocationObjective,
)
from src.shared.python.motion_matching.multi_engine_torque_allocator import (
    EngineType,
    MultiEngineTorqueAllocator,
    create_engine_force_adapter,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("allocate_swing_torques")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Allocate dynamic actuator torques and contact forces across physics engines."
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="mujoco",
        choices=[e.value for e in EngineType if e != EngineType.PINOCCHIO],
        help="Target physics engine (mujoco, drake, opensim, simscape)",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="Path to candidate trajectory .npz file (containing q, v, time_s)",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("docs/development/full_body_models/full_body_spec_v1.json"),
        help="Path to full body specification JSON (for MuJoCo)",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="minimum_effort",
        choices=["minimum_effort", "minimum_trail_arm"],
        help="Torque allocation objective",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame evaluation stride (default: 1, full resolution)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path for allocated torque trajectory .npz",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    candidate_path = Path(args.candidate)
    if not candidate_path.is_file():
        logger.error("Candidate file not found: %s", candidate_path)
        return 1

    logger.info("Loading candidate trajectory from %s...", candidate_path)
    data = np.load(candidate_path, allow_pickle=True)
    time_s = np.asarray(data["time_s"], dtype=np.float64)
    q = np.asarray(data["q"], dtype=np.float64)
    v = np.asarray(data["v"], dtype=np.float64)

    # Compute numerical acceleration if not stored
    if "a" in data:
        a = np.asarray(data["a"], dtype=np.float64)
    else:
        dt = np.gradient(time_s)
        dt = np.where(dt <= 0, 1e-3, dt)
        a = np.gradient(v, axis=0) / dt[:, None]

    n_frames = len(time_s)
    logger.info(
        "Trajectory loaded: %d frames (%.3f s)", n_frames, time_s[-1] - time_s[0]
    )

    logger.info("Creating force adapter for engine: %s...", args.engine)
    adapter = create_engine_force_adapter(
        engine=args.engine,
        spec_path=args.spec if Path(args.spec).is_file() else None,
        nv=q.shape[1],
    )

    allocator = MultiEngineTorqueAllocator(adapter=adapter)
    objective = (
        AllocationObjective.MINIMUM_EFFORT
        if args.objective == "minimum_effort"
        else AllocationObjective.MINIMUM_TRAIL_ARM
    )

    # Trail arm coordinate indices if minimizing trail arm
    trail_indices = (
        list(range(18, 27))
        if objective == AllocationObjective.MINIMUM_TRAIL_ARM
        else None
    )

    logger.info("Solving dynamic force allocation (objective=%s)...", objective.value)
    result = allocator.allocate_trajectory(
        time_s=time_s,
        q_traj=q,
        v_traj=v,
        a_traj=a,
        objective=objective,
        trail_arm_indices=trail_indices,
        stride=args.stride,
    )

    logger.info("Allocation completed:")
    logger.info("  Success: %s", result.success)
    logger.info(
        "  Max dynamic equilibrium residual: %.4e N*m", result.max_equilibrium_residual
    )
    logger.info("  Max root slack residual: %.4e N", result.max_root_residual)
    if result.parity_residuals:
        logger.info(
            "  Mean acceleration parity error: %.4e m/s^2",
            float(np.mean(result.parity_residuals)),
        )
        logger.info(
            "  Max acceleration parity error: %.4e m/s^2",
            float(np.max(result.parity_residuals)),
        )

    out_path = args.out
    if out_path is None:
        out_path = (
            candidate_path.parent / f"allocated_{args.engine}_{args.objective}.npz"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        time_s=result.time_s,
        tau_actuated=result.tau_actuated,
        f_ground=result.f_ground,
        lambda_grip=result.lambda_grip,
        delta_tau_root=result.delta_tau_root,
        metrics=json.dumps(result.as_dict()),
    )
    logger.info("Saved allocated torques and contact forces to: %s", out_path)

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
