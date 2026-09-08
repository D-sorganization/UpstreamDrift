"""Generate the swing-backend parity table (epic #9762, Phase 2.2).

Runs every available trajectory-optimization backend on one benchmark
golfer and reports terminal clubhead speed, wall time, iterations and --
the column that motivates the epic -- the maximum dynamics defect, i.e. how
far each solution is from satisfying the swing ODE between its own nodes.

    python -m benchmarks.bioptim_parity --nodes 8 --out docs/estimation/bioptim_parity.md

Backends whose optional stack is absent are reported as unavailable rather
than skipped silently.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.optimization._swing_kinematics import (
    JOINTS,
    generate_initial_guess,
)
from src.shared.python.optimization._swing_models import (
    ClubModel,
    GolferModel,
    OptimizationConfig,
)
from src.shared.python.optimization.model_provider import swing_joint_limits


@dataclass
class Row:
    """One backend's parity numbers."""

    backend: str
    available: bool
    converged: bool | None = None
    clubhead_speed_m_s: float | None = None
    wall_time_s: float | None = None
    iterations: int | None = None
    max_position_defect_rad: float | None = None
    max_velocity_defect_rad_s: float | None = None
    note: str = ""


def _torque_limits(golfer: GolferModel) -> dict[str, float]:
    return {
        "hip_rotation": golfer.max_hip_torque,
        "trunk_rotation": golfer.max_trunk_torque,
        "shoulder_horizontal": golfer.max_shoulder_torque,
        "shoulder_vertical": golfer.max_shoulder_torque,
        "elbow_flexion": golfer.max_elbow_torque,
        "wrist_cock": golfer.max_wrist_torque,
        "wrist_rotation": golfer.max_wrist_torque,
    }


def _terminal_speed(
    golfer: GolferModel, club: ClubModel, x: np.ndarray, n_nodes: int
) -> float:
    from src.shared.python.optimization.ocp.symbolic_model import SymbolicSwingModel

    n = len(JOINTS)
    q = x[: n * n_nodes].reshape(n, n_nodes)
    v = x[n * n_nodes :].reshape(n, n_nodes)
    model = SymbolicSwingModel(golfer, club)
    velocity = model.clubhead_velocity(q[:, -1], v[:, -1], np.zeros(0))
    return float(np.linalg.norm(np.asarray(velocity)))


def _defects(
    golfer: GolferModel,
    club: ClubModel,
    config: OptimizationConfig,
    x: np.ndarray,
    torques: np.ndarray | None,
) -> tuple[float, float]:
    from src.shared.python.optimization.casadi_backend import dynamics_defect

    report = dynamics_defect(golfer, club, config, x, torques=torques)
    return report.max_position_defect, report.max_velocity_defect


def run(config: OptimizationConfig, golfer: GolferModel, club: ClubModel) -> list[Row]:
    """Run every available backend and return the parity rows."""
    joint_limits = swing_joint_limits(golfer)
    limits = _torque_limits(golfer)
    x0 = generate_initial_guess(golfer, config, joint_limits)
    rows: list[Row] = []

    initial_defect = None
    from src.shared.python.optimization.casadi_backend import casadi_available

    if casadi_available():
        initial_defect = _defects(golfer, club, config, x0, None)
        rows.append(
            Row(
                backend="initial guess (kinematic)",
                available=True,
                converged=None,
                clubhead_speed_m_s=_terminal_speed(golfer, club, x0, config.n_nodes),
                max_position_defect_rad=initial_defect[0],
                max_velocity_defect_rad_s=initial_defect[1],
                note="the smooth sine warm start every backend receives",
            )
        )

    def timed(name: str, note: str, call: Any) -> None:
        started = time.perf_counter()
        result = call()
        wall = time.perf_counter() - started
        defect = _defects(golfer, club, config, result.x, result.torques)
        rows.append(
            Row(
                backend=name,
                available=True,
                converged=bool(result.success),
                clubhead_speed_m_s=_terminal_speed(
                    golfer, club, result.x, config.n_nodes
                ),
                wall_time_s=wall,
                iterations=int(result.iterations),
                max_position_defect_rad=defect[0],
                max_velocity_defect_rad_s=defect[1],
                note=note,
            )
        )

    # scipy flagship
    from src.shared.python.optimization.swing_optimizer import SwingOptimizer

    started = time.perf_counter()
    scipy_result = SwingOptimizer(golfer, club, config).optimize()
    scipy_wall = time.perf_counter() - started
    rows.append(
        Row(
            backend="scipy SLSQP",
            available=True,
            converged=bool(scipy_result.success),
            clubhead_speed_m_s=float(scipy_result.predicted_clubhead_speed),
            wall_time_s=scipy_wall,
            iterations=int(scipy_result.iterations),
            note="flagship path; its own kinematic model, no ODE to violate",
        )
    )

    if casadi_available():
        from src.shared.python.optimization.casadi_backend import solve_swing_casadi

        timed(
            "casadi finite-difference",
            "legacy #9756 path: kinematic fit with a torque check",
            lambda: solve_swing_casadi(golfer, club, config, limits, joint_limits, x0),
        )
        timed(
            "casadi multiple shooting",
            "RK4 shooting constraints, torque decision variables",
            lambda: solve_swing_casadi(
                golfer,
                club,
                config,
                limits,
                joint_limits,
                x0,
                transcription="multiple_shooting",
            ),
        )
    else:
        rows.append(
            Row(
                backend="casadi finite-difference",
                available=False,
                note="pip install 'upstream-drift[optimal-control]'",
            )
        )
        rows.append(
            Row(
                backend="casadi multiple shooting",
                available=False,
                note="pip install 'upstream-drift[optimal-control]'",
            )
        )

    from src.shared.python.optimization.crocoddyl_backend import crocoddyl_available

    if crocoddyl_available():
        from src.shared.python.optimization.backend_registry import get_backend

        spec = get_backend("crocoddyl")
        assert spec is not None and spec.solve is not None
        timed(
            "crocoddyl FDDP",
            "DDP; targets an impact speed rather than maximising",
            lambda: spec.solve(golfer, club, config, limits, joint_limits, x0),
        )
    else:
        rows.append(
            Row(
                backend="crocoddyl FDDP",
                available=False,
                note="conda install -c conda-forge crocoddyl pinocchio",
            )
        )

    from src.shared.python.optimization.ocp._compat import bioptim_available

    if bioptim_available():
        from src.shared.python.optimization.ocp.swing_ocp import solve_max_speed_ocp

        for ode in ("rk4", "collocation"):
            timed(
                f"bioptim {ode}",
                "target-speed objective (convex); dynamics enforced by construction",
                lambda ode=ode: solve_max_speed_ocp(
                    golfer, club, config, limits, joint_limits, x0, ode=ode
                ).result,
            )
    else:
        rows.append(
            Row(
                backend="bioptim rk4",
                available=False,
                note="pip install 'upstream-drift[bioptim]'",
            )
        )
        rows.append(
            Row(
                backend="bioptim collocation",
                available=False,
                note="pip install 'upstream-drift[bioptim]'",
            )
        )

    return rows


def _cell(value: Any) -> str:
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e5):
            return f"{value:.1e}"
        return f"{value:.3g}"
    return str(value)


def to_markdown(
    rows: list[Row], config: OptimizationConfig, golfer: GolferModel
) -> str:
    """Render the parity table."""
    header = (
        "| Backend | Converged | Clubhead speed [m/s] | Wall [s] | Iterations "
        "| Max position defect [rad] | Max velocity defect [rad/s] | Notes |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )
    lines = []
    for row in rows:
        if not row.available:
            lines.append(
                f"| {row.backend} | _not installed_ | -- | -- | -- | -- | -- | {row.note} |"
            )
            continue
        lines.append(
            f"| {row.backend} | {_cell(row.converged)} | {_cell(row.clubhead_speed_m_s)} | {_cell(row.wall_time_s)} | {_cell(row.iterations)} | {_cell(row.max_position_defect_rad)} | {_cell(row.max_velocity_defect_rad_s)} | {row.note} |"
        )
    return header + "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=8)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    golfer, club = GolferModel(), ClubModel()
    config = OptimizationConfig(
        n_nodes=args.nodes,
        swing_duration=args.duration,
        max_iterations=args.max_iterations,
    )
    rows = run(config, golfer, club)
    table = to_markdown(rows, config, golfer)
    print(table)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table, encoding="utf-8")
    if args.json:
        args.json.write_text(
            json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
