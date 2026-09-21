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

from src.shared.python.model_generation.export.model_bundle import load_model_bundle
from src.shared.python.motion_matching.native_viewers import (
    VALID_VIEW_MODES,
    ViewerLaunchConfig,
    ViewerUnavailableError,
    get_backend_adapter,
    get_supported_backends,
    open_in_native_viewer,
)
from src.shared.python.motion_matching.visualization.simulation_viewer import (
    SimulationData,
    SimulationViewer,
    ViewerBackend,
    launch_viewer,
)


def is_backend_available(viewer_name: str) -> bool:
    """Check whether a viewer backend is available in the current environment."""
    v_clean = viewer_name.lower()
    if v_clean in get_supported_backends():
        try:
            return get_backend_adapter(v_clean).is_available()
        except (RuntimeError, ValueError, OSError, ImportError):
            return False
    try:
        return SimulationViewer.is_backend_available(v_clean)
    except (RuntimeError, ValueError, OSError, ImportError):
        return False


def build_parser() -> argparse.ArgumentParser:
    all_choices = list(
        dict.fromkeys([v.value for v in ViewerBackend] + get_supported_backends())
    )
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
        choices=all_choices,
        help="Target visualizer backend (default: pyvista).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=60.0,
        help="Playback frame rate in frames per second (default: 60.0).",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (default: 1.0).",
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
        "--view-mode",
        type=str,
        default="fitted",
        choices=list(VALID_VIEW_MODES),
        help="Viewing camera/scene mode: static, fitted, or native (default: fitted).",
    )
    parser.add_argument(
        "--model-bundle",
        type=Path,
        default=None,
        help="Optional path to model bundle archive (.zip) or directory.",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=None,
        help="Optional path to standalone URDF file.",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=None,
        help="Optional path to full body model specification JSON for native model loading.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=None,
        help="Optional path to export static HTML visualization.",
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
        avail = is_backend_available(args.viewer)
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

    if args.model_bundle is not None and args.model_bundle.exists():
        print(f"Loading model bundle from {args.model_bundle}...")
        bundle = load_model_bundle(args.model_bundle)
        print(
            f"Loaded model bundle with {len(bundle.manifest.coordinate_order)} coordinates and {len(bundle.mesh_assets)} mesh assets."
        )

    launch_cfg = ViewerLaunchConfig(
        speed=args.speed,
        loop=args.loop,
        stride=args.stride,
        fps=args.fps,
        view_mode=args.view_mode,
        model_bundle_path=args.model_bundle,
        urdf_path=args.urdf,
        output_html=args.output_html,
    )

    try:
        if args.viewer in get_supported_backends():
            open_in_native_viewer(sim_data, args.viewer, config=launch_cfg)
        else:
            launch_viewer(
                viewer=args.viewer,
                data=sim_data,
                loop=args.loop,
                stride=args.stride,
                fps=args.fps * args.speed,
                spec_path=args.spec,
            )
        return 0
    except (
        ViewerUnavailableError,
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
