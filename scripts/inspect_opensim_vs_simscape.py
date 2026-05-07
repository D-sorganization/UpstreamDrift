"""Debug helper for the OpenSim ↔ Simscape equivalence test (issue #4131).

Prints a per-pose comparison table of grip and clubhead positions /
orientations between the OpenSim engine and the Simscape ground-truth
fixture used by ``tests/test_opensim_simscape_equivalence.py``.

Run from repo root::

    python3 scripts/inspect_opensim_vs_simscape.py

Optional arguments:
    --fixture PATH   override the .npz path
    --pose NAME      restrict to a single pose (address|top_of_backswing|impact)
    --verbose        print full per-axis residuals

The script never asserts; it is purely diagnostic. Use the equivalence
test for pass/fail signal in CI.
"""

from __future__ import annotations

import argparse
import importlib
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "opensim_simscape_equivalence"
    / "simscape_reference.npz"
)
POSE_NAMES = ("address", "top_of_backswing", "impact")


def _load_fixture(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        sys.stderr.write(
            f"fixture missing: {path}\n"
            f"  run: python3 -m tests.fixtures.opensim_simscape_equivalence."
            f"build_fixture --trial <trial_001_*.csv>\n"
        )
        raise SystemExit(2)
    with np.load(path, allow_pickle=False) as data:
        return {key: np.array(data[key]) for key in data.files}


def _quat_geodesic_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = float(np.clip(abs(np.dot(q1, q2)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(dot))


def _try_import_simulate() -> object | None:
    try:
        module = importlib.import_module(
            "src.engines.physics_engines.opensim.python.motion_matching.simulate"
        )
    except ModuleNotFoundError:
        return None
    return getattr(module, "simulate_with_coefficients", None)


def _try_import_coord_map() -> object | None:
    try:
        return importlib.import_module(
            "src.engines.physics_engines.opensim.python.motion_matching.coord_map"
        )
    except ModuleNotFoundError:
        return None


def _print_simscape_only(fixture: dict[str, np.ndarray], pose: str) -> None:
    grip = fixture[f"{pose}_grip"]
    clubhead = fixture[f"{pose}_clubhead"]
    quat = fixture[f"{pose}_grip_quat"]
    print(f"  Simscape grip       (m): {grip}")
    print(f"  Simscape clubhead   (m): {clubhead}")
    print(f"  Simscape grip_quat     : {quat}")


def _print_pose_comparison(
    pose: str,
    fixture: dict[str, np.ndarray],
    coord_map: object,
    simulate_fn: object,
    *,
    verbose: bool,
) -> None:
    print(f"\n=== pose: {pose} ===")

    if simulate_fn is None or coord_map is None:
        _print_simscape_only(fixture, pose)
        if simulate_fn is None:
            print("  [skipped] simulate_with_coefficients unavailable (#4120)")
        if coord_map is None:
            print("  [skipped] coord_map unavailable (#4114 / #4167)")
        return

    n_joints = len(coord_map.OPENSIM_COORD_ORDER)  # type: ignore[attr-defined]
    theta = np.zeros(n_joints * 7, dtype=np.float64)

    q_key = f"{pose}_q_simscape"
    if q_key in fixture and fixture[q_key].shape == (25,):
        q_opensim = coord_map.from_simscape(fixture[q_key])  # type: ignore[attr-defined]
    else:
        q_opensim = np.array(
            coord_map.OPENSIM_NEUTRAL_POSE,  # type: ignore[attr-defined]
            dtype=np.float64,
        )
    initial_pose = {
        "q_opensim": q_opensim,
        "qdot_opensim": np.zeros_like(q_opensim),
    }

    try:
        sim_out = simulate_fn(theta=theta, initial_pose=initial_pose)  # type: ignore[misc]
    except Exception as exc:  # noqa: BLE001 — debug script
        print(f"  simulate_with_coefficients raised: {exc}")
        _print_simscape_only(fixture, pose)
        return

    grip_yup = np.asarray(sim_out.grip, dtype=np.float64)[0]
    clubhead_yup = np.asarray(sim_out.clubhead, dtype=np.float64)[0]
    grip_quat = np.asarray(sim_out.grip_quat, dtype=np.float64)[0]

    grip_zup = coord_map.frame_y_up_to_z_up(grip_yup)  # type: ignore[attr-defined]
    clubhead_zup = coord_map.frame_y_up_to_z_up(clubhead_yup)  # type: ignore[attr-defined]

    grip_truth = fixture[f"{pose}_grip"]
    clubhead_truth = fixture[f"{pose}_clubhead"]
    grip_quat_truth = fixture[f"{pose}_grip_quat"]

    grip_err_mm = float(np.sqrt(np.mean((grip_zup - grip_truth) ** 2))) * 1000.0
    ch_err_mm = float(np.sqrt(np.mean((clubhead_zup - clubhead_truth) ** 2))) * 1000.0
    quat_err_deg = _quat_geodesic_deg(grip_quat, grip_quat_truth)

    print(f"  OpenSim grip      (Z-up, m): {grip_zup}")
    print(f"  Simscape grip     (Z-up, m): {grip_truth}")
    print(f"  Δgrip RMSE                : {grip_err_mm:.3f} mm")
    print(f"  OpenSim clubhead  (Z-up, m): {clubhead_zup}")
    print(f"  Simscape clubhead (Z-up, m): {clubhead_truth}")
    print(f"  Δclubhead RMSE            : {ch_err_mm:.3f} mm")
    print(f"  Δgrip orientation         : {quat_err_deg:.3f} deg")
    if verbose:
        print(f"  OpenSim grip_quat       : {grip_quat}")
        print(f"  Simscape grip_quat      : {grip_quat_truth}")
        print(f"  per-axis grip residual    (m): {grip_zup - grip_truth}")
        print(f"  per-axis clubhead residual(m): {clubhead_zup - clubhead_truth}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--pose", choices=POSE_NAMES, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    fixture = _load_fixture(args.fixture)
    print(f"fixture: {args.fixture}")
    print(f"  trial : {fixture.get('trial_path', np.array(['?']))}")
    print(f"  N     : {int(fixture['n_samples'])}")
    print(
        f"  impact_idx={int(fixture['impact_idx'])} top_idx={int(fixture['top_idx'])}"
    )

    coord_map = _try_import_coord_map()
    simulate_fn = _try_import_simulate()

    poses = [args.pose] if args.pose else list(POSE_NAMES)
    for pose in poses:
        _print_pose_comparison(
            pose,
            fixture,
            coord_map,
            simulate_fn,
            verbose=args.verbose,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
