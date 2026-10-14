"""Regenerate the committed back-squat same-input bundle (LIFT-5, #11745).

Usage (headless, from the repository root)::

    MUJOCO_GL=egl python3 scripts/generate_lift_bundle.py \
        --out docs/development/lifting/evidence/back_squat/back_squat_bundle.npz

The Movement Optimizer is consumed from the vendored Tools tree
(``vendor/ud-tools/src``); the source commit and Tools pin are recorded in
the bundle provenance.  The window is centred on the squat bottom (lowest
bar sample) so the bundle covers the descent-to-ascent reversal.

``--geometry back_bar`` (#12347) carries the bar on the torso shelf with the
parity standard's grip width and a medio-lateral axis, and writes the derived
spec into the bundle.  The committed bundle is still ``legacy`` (the golf club
grip): regenerating it with ``back_bar`` changes the inputs of the bar-path,
kinetics and barbell evidence (#12340, #12341, #12345), so that happens
together with their regeneration, not before.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOLS_SRC = ROOT / "vendor" / "ud-tools" / "src"
if str(TOOLS_SRC) not in sys.path:
    sys.path.append(str(TOOLS_SRC))

from src.shared.python.lifting.back_bar import (  # noqa: E402
    BACK_SQUAT,
    BackBarGeometry,
)
from src.shared.python.lifting.bundles import (  # noqa: E402
    BACK_SQUAT_BUNDLE,
    LIFT_SPEC,
    SEED_RECORD,
    build_lift_bundle,
    lift_to_full_body,
    load_seed_pose,
    solve_sagittal_squat,
)

logger = logging.getLogger("generate_lift_bundle")


def _git(*args: str) -> str:
    out = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", "-C", str(ROOT), *args],  # noqa: S607
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return out.stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=BACK_SQUAT_BUNDLE)
    parser.add_argument("--window", type=float, default=0.2, help="seconds")
    parser.add_argument("--body-mass", type=float, default=80.0)
    parser.add_argument("--height", type=float, default=1.78)
    parser.add_argument("--bar-mass", type=float, default=100.0)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--n-eval", type=int, default=151)
    parser.add_argument(
        "--geometry",
        choices=("legacy", "back_bar"),
        default="legacy",
        help="legacy: golf club grip (committed bundle); back_bar: #12347 geometry",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    solution = solve_sagittal_squat(
        body_mass_kg=args.body_mass,
        height_m=args.height,
        bar_mass_kg=args.bar_mass,
        duration_s=args.duration,
        n_eval=args.n_eval,
    )
    spec = LIFT_SPEC.read_bytes()
    geometry = (
        BackBarGeometry.from_standard(BACK_SQUAT, height_m=args.height)
        if args.geometry == "back_bar"
        else None
    )
    lift = lift_to_full_body(spec, solution, load_seed_pose(), geometry=geometry)
    if lift.back_bar is not None:
        spec = lift.back_bar.spec_bytes
    for label, value in lift.max_residual_by_landmark().items():
        logger.info("IK max residual %-10s %.5f m", label, value)
    logger.info("IK max closure %.3e m", lift.closure_m.max())
    logger.info("IK min foot-sphere height %.3e m", lift.lowest_sphere_m.min())

    bottom = float(solution.time_s[int(np.argmin(solution.bar_m[:, 1]))])
    wanted = bottom - args.window / 2
    start = float(solution.time_s[int(np.argmin(np.abs(solution.time_s - wanted)))])
    tools_pin = _git("ls-tree", "HEAD", "vendor/ud-tools").split()[2]
    bundle = build_lift_bundle(
        spec,
        solution,
        lift,
        lift_name="back_squat",
        window_start_s=start,
        window_s=args.window,
        provenance={
            "source_commit": _git("rev-parse", "HEAD"),
            "tools_pin": tools_pin,
            "seed": {
                "path": str(SEED_RECORD.relative_to(ROOT)),
                "sha256": _sha256(SEED_RECORD),
                "row": 0,
            },
            "spec_path": str(LIFT_SPEC.relative_to(ROOT)),
            "generator": "scripts/generate_lift_bundle.py",
        },
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    bundle.save(args.out)
    logger.info(
        "wrote %s: %d steps from %.3f s, %d bytes, max pose drift %.2e m",
        args.out,
        bundle.steps,
        start,
        args.out.stat().st_size,
        bundle.provenance["max_pose_drift_m"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
