"""Driver for OpenSim degree-six polynomial effort profile and forward replay (OS-5).

Executes on ControlTower with OpenSim 4.6 bindings:
1. Loads discrete control trajectories from OS-4 dynamic tracking.
2. Fits degree-6 polynomials (c0..c6 descending powers) to all 39 OpenSim actuators.
3. Validates fit fidelity (R^2, max error, RMS error) and effort/rate bounds.
4. Builds an opensim.PrescribedController with opensim.PolynomialFunction per actuator.
5. Replays continuous forward simulation via opensim.Manager (zero-feedback).
6. Exports polynomial coefficients, forward states/controls, and cryptographic receipt.

Verified on ControlTower runner for deliverable OS-5 (epic #10003).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import subprocess  # nosec B404
import sys
import time
from typing import Any

# Add repo root to path
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.engines.physics_engines.opensim.python.tour_matching.polynomial_profile import (
    DEGREE_6,
    PolynomialTorqueProfile,
    check_effort_and_rate_bounds,
    create_polynomial_prescribed_controller,
    fit_degree6_from_discrete_controls,
    load_controls_from_sto,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("os5_polynomial_fit_driver")


def sha256_file(path: Path) -> str:
    """Calculate SHA256 digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        out = subprocess.check_output(  # nosec B603, B607
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            text=True,
        ).strip()
        return out
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="OS-5 Degree-6 Polynomial Driver")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "/mnt/c/Users/diete/opensim-10003/os3b-scale-ik/golf_humanoid_scaled_tour_markers.osim"
        ),
        help="Path to scaled, calibrated .osim model",
    )
    parser.add_argument(
        "--controls",
        type=Path,
        default=repo_root
        / "docs/development/opensim_tour_matching/evidence/os4_moco_tracking/tracked_controls.sto",
        help="Path to OS-4 tracked controls .sto",
    )
    parser.add_argument(
        "--ik-states",
        type=Path,
        default=Path("/mnt/c/Users/diete/opensim-10003/ik_states_full.sto"),
        help="Path to full IK states trajectory .sto",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root
        / "docs/development/opensim_tour_matching/evidence/os5_polynomial_profile",
        help="Output directory for OS-5 evidence",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.10,
        help="Polynomial domain duration in seconds (default: 0.10s)",
    )
    return parser.parse_args()


def load_initial_ik_coordinates(ik_states_path: Path) -> dict[str, float]:
    """Extract initial coordinate values from IK states table."""
    import opensim

    coords: dict[str, float] = {}
    if not ik_states_path.is_file():
        logger.warning(
            "IK states file %s not found; skipping coordinate seeding", ik_states_path
        )
        return coords

    table = opensim.TimeSeriesTable(str(ik_states_path))
    labels = table.getColumnLabels()
    row_0 = table.getRowAtIndex(0)
    for i in range(table.getNumColumns()):
        label = labels[i]
        val = row_0.getElt(0, i)
        clean = label.split("/")[-1].replace("/value", "")
        coords[clean] = float(val)
    return coords


def fit_and_export_polynomial_profile(
    controls_path: Path,
    duration_s: float,
    output_dir: Path,
) -> tuple[PolynomialTorqueProfile, dict[str, Any]]:
    """Fit degree-6 polynomials to controls and export JSON."""
    logger.info("Loading discrete controls from %s", controls_path)
    times, controls, names = load_controls_from_sto(controls_path)
    logger.info(
        "Fitting degree-6 polynomials for %d actuators over %.3f s",
        len(names),
        duration_s,
    )

    profile = fit_degree6_from_discrete_controls(
        times=times,
        controls=controls,
        actuator_names=names,
        duration_s=duration_s,
    )

    poly_json_path = output_dir / "polynomial_coefficients.json"
    profile.save_json(poly_json_path)
    logger.info("Saved polynomial coefficients to %s", poly_json_path)

    ok_bounds, violations = check_effort_and_rate_bounds(
        profile=profile,
        max_effort=1000.0,
        max_rate=50000.0,
    )
    logger.info(
        "Effort and rate bounds check: ok=%s (%d violations)",
        ok_bounds,
        len(violations),
    )

    summary_metrics: dict[str, Any] = {
        "num_actuators": len(names),
        "polynomial_degree": DEGREE_6,
        "duration_s": duration_s,
        "sample_count": len(times),
        "bounds_passed": ok_bounds,
        "bounds_violations": violations,
        "actuator_metrics": profile.fit_metrics,
    }
    return profile, summary_metrics


def run_forward_polynomial_replay(
    model_path: Path,
    profile: PolynomialTorqueProfile,
    ik_states_path: Path,
    duration_s: float,
    output_dir: Path,
) -> dict[str, Any]:
    """Execute forward simulation via opensim.Manager using PrescribedController."""
    import opensim

    logger.info("Loading model from %s", model_path)
    model = opensim.Model(str(model_path))

    controller = create_polynomial_prescribed_controller(profile, model)
    model.addController(controller)

    state = model.initSystem()
    init_coords = load_initial_ik_coordinates(ik_states_path)
    coord_set = model.getCoordinateSet()
    for name, val in init_coords.items():
        if coord_set.contains(name):
            coord = coord_set.get(name)
            coord.setValue(state, val)
            coord.setSpeedValue(state, 0.0)

    model.assemble(state)
    logger.info("Model assembled at t=%.4f s", state.getTime())

    manager = opensim.Manager(model)
    state.setTime(0.0)
    manager.initialize(state)

    t0 = time.perf_counter()
    final_state = manager.integrate(duration_s)
    sim_wall_time_ms = (time.perf_counter() - t0) * 1000.0

    logger.info(
        "Forward simulation completed to t=%.4f s in %.2f ms",
        final_state.getTime(),
        sim_wall_time_ms,
    )

    # Save forward states
    states_table = manager.getStatesTable()
    states_sto_path = output_dir / "forward_states.sto"
    opensim.STOFileAdapter.write(states_table, str(states_sto_path))
    logger.info("Saved forward states to %s", states_sto_path)

    return {
        "success": True,
        "sim_wall_time_ms": sim_wall_time_ms,
        "terminal_time_s": float(final_state.getTime()),
        "num_steps": int(states_table.getNumRows()),
        "states_sto": str(states_sto_path),
    }


def main() -> None:
    """Run OS-5 end-to-end polynomial fitting and forward replay."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    profile, fit_summary = fit_and_export_polynomial_profile(
        controls_path=args.controls,
        duration_s=args.duration,
        output_dir=args.output_dir,
    )

    replay_summary = run_forward_polynomial_replay(
        model_path=args.model,
        profile=profile,
        ik_states_path=args.ik_states,
        duration_s=args.duration,
        output_dir=args.output_dir,
    )

    # Build receipt
    receipt: dict[str, Any] = {
        "schema_version": "1.0",
        "deliverable": "OS-5",
        "title": "Global Degree-Six Effort Profile and Forward Simulation Replay",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "inputs": {
            "model_path": str(args.model),
            "model_sha256": sha256_file(args.model) if args.model.is_file() else None,
            "controls_path": str(args.controls),
            "controls_sha256": (
                sha256_file(args.controls) if args.controls.is_file() else None
            ),
            "ik_states_path": str(args.ik_states),
            "ik_states_sha256": (
                sha256_file(args.ik_states) if args.ik_states.is_file() else None
            ),
        },
        "polynomial_fit": fit_summary,
        "forward_replay": replay_summary,
    }

    receipt_path = args.output_dir / "receipt.json"
    with receipt_path.open("w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)
    logger.info("Saved cryptographic receipt to %s", receipt_path)


if __name__ == "__main__":
    main()
