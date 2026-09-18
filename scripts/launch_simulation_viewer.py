"""CLI to launch candidate simulations across 3D viewers.

Usage:
    python scripts/launch_simulation_viewer.py --candidate evidence/matched/driver_full_pinocchio/candidate.npz --viewer gepetto
    python scripts/launch_simulation_viewer.py --candidate evidence/matched/driver_full_pinocchio/candidate.npz --viewer mujoco
    python scripts/launch_simulation_viewer.py --candidate evidence/matched/driver_full_pinocchio/candidate.npz --viewer pyvista
    python scripts/launch_simulation_viewer.py --candidate evidence/matched/driver_full_pinocchio/candidate.npz --viewer meshcat
    python scripts/launch_simulation_viewer.py --candidate evidence/matched/driver_full_pinocchio/candidate.npz --viewer matplotlib
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.shared.python.motion_matching.visualization.simulation_viewer import (
    SimulationData,
    SimulationViewer,
    ViewerBackend,
    launch_viewer,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch motion matching simulation candidate in an interactive 3D viewer."
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="Path to candidate.npz archive containing full swing trajectory.",
    )
    parser.add_argument(
        "--viewer",
        type=str,
        default="pyvista",
        choices=[v.value for v in ViewerBackend],
        help="Target visualizer backend (default: pyvista).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=60.0,
        help="Playback frame rate in frames per second (default: 60.0).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Playback frame stride step (default: 1).",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop playback continuously.",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=None,
        help="Optional path to full body model specification JSON for native model loading.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check whether the requested viewer backend is available without launching.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.check_only:
        avail = SimulationViewer.is_backend_available(args.viewer)
        print(f"Viewer '{args.viewer}' available: {avail}")
        return 0 if avail else 1

    if not args.candidate.is_file():
        print(f"Error: Candidate file '{args.candidate}' not found.", file=sys.stderr)
        return 1

    print(f"Loading candidate trajectory from {args.candidate}...")
    sim_data = SimulationData.from_npz(args.candidate)
    print(
        f"Loaded {len(sim_data.time_s)} frames across {sim_data.q.shape[1]} coordinates."
    )

    try:
        launch_viewer(
            viewer=args.viewer,
            data=sim_data,
            loop=args.loop,
            stride=args.stride,
            fps=args.fps,
            spec_path=args.spec,
        )
        return 0
    except (
        RuntimeError,
        ValueError,
        FileNotFoundError,
        OSError,
        ConnectionError,
    ) as exc:
        print(f"Failed to launch viewer '{args.viewer}': {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
