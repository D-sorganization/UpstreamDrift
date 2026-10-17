"""Regenerate a committed lift same-input bundle (LIFT-5, #11745).

Usage (headless, from the repository root)::

    MUJOCO_GL=egl python3 scripts/generate_lift_bundle.py --lift squat
    MUJOCO_GL=egl python3 scripts/generate_lift_bundle.py --lift deadlift
    MUJOCO_GL=egl python3 scripts/generate_lift_bundle.py --lift clean_and_jerk
    MUJOCO_GL=egl python3 scripts/generate_lift_bundle.py --lift bench_press

Each lift's optimizer exercise, bar model, duration and window event come
from its recipe (``src/shared/python/lifting/lift_recipes.py``); the bundle is
written to the recipe's evidence path unless ``--out`` is given.  The Movement
Optimizer is consumed from the vendored Tools tree (``vendor/ud-tools/src``);
the source commit and Tools pin are recorded in the bundle provenance.  The
squat window is centred on the bottom (lowest bar sample); the pulls'
windows on the knee pass (bar at knee height), with the bar held in the hands
at the standard's grip width (:func:`lift_hands_to_full_body`).  The bench
press lies supine on a bench contact body and is centred on the chest touch
(:func:`~src.shared.python.lifting.bench_press.lift_bench_to_full_body`); its
bench record and limitations go into the provenance.

``--geometry back_bar`` (#12347) carries the bar on the torso shelf with the
parity standard's grip width and a medio-lateral axis, and writes the derived
spec into the bundle.  The committed bundle is still ``legacy`` (the golf club
grip): regenerating it with ``back_bar`` changes the inputs of the bar-path,
kinetics and barbell evidence (#12340, #12341, #12345), so that happens
together with their regeneration, not before.

Since #13093 the back-bar squat IK balances the optimizer's bar load over
mid-foot with ``--load-balance`` (opt-in), and the bundle's spec then carries
that bar as a rigid solid on the hands (:func:`lift_bundle_spec`).  The
committed squat bundle is not balanced: tracking the balanced IK with the bar
load loses the feet within the window (OSL-15 measurements in
``docs/research/lift_models/lift_models.tex``).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOLS_SRC = ROOT / "vendor" / "ud-tools" / "src"
if str(TOOLS_SRC) not in sys.path:
    # Ahead of site-packages so the pinned Tools tree resolves the retired
    # shared roots (sidekick utils) imported by the same-input pipeline.
    sys.path.insert(1, str(TOOLS_SRC))

from src.shared.python.lifting.back_bar import (  # noqa: E402
    BACK_SQUAT,
    BackBarGeometry,
)
from src.shared.python.lifting.barbell_bushing import Barbell  # noqa: E402
from src.shared.python.lifting.bench_press import (  # noqa: E402
    BenchLift,
    lift_bench_to_full_body,
)
from src.shared.python.lifting.bundles import (  # noqa: E402
    LIFT_SPEC,
    build_lift_bundle,
    lift_bundle_spec,
    lift_hands_to_full_body,
    lift_to_full_body,
    load_seed_pose,
    solve_sagittal_lift,
)
from src.shared.python.lifting.loaded_balance import LoadBalance  # noqa: E402
from src.shared.python.lifting.lift_recipes import (  # noqa: E402
    LiftRecipe,
    bundle_lifts,
    recipe,
    window_start_time,
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
    parser.add_argument("--lift", choices=bundle_lifts(), default="squat")
    parser.add_argument("--out", type=Path, default=None, help="default: recipe")
    parser.add_argument("--window", type=float, default=0.2, help="seconds")
    parser.add_argument("--body-mass", type=float, default=80.0)
    parser.add_argument("--height", type=float, default=1.78)
    parser.add_argument(
        "--bar-mass", type=float, default=None, help="default: the recipe's"
    )
    parser.add_argument(
        "--duration", type=float, default=None, help="default: the recipe's"
    )
    parser.add_argument("--n-eval", type=int, default=151)
    parser.add_argument(
        "--geometry",
        choices=("legacy", "back_bar"),
        default="back_bar",
        help="squat only. back_bar: #12347 geometry (committed); legacy: golf grip",
    )
    parser.add_argument(
        "--load-balance",
        action="store_true",
        help="back_bar only: balance the bar load over mid-foot (#13093, opt-in)",
    )
    return parser


def _full_body(args: argparse.Namespace, lift: LiftRecipe, solution: Any) -> Any:
    """Full-body IK of *solution*: back bar (squat), hands or bench press.

    The back bar is balanced with the optimizer's bar load over mid-foot
    (#13093) when ``--load-balance`` is given.
    """
    spec = LIFT_SPEC.read_bytes()
    if lift.supine:
        return lift_bench_to_full_body(spec, solution, load_seed_pose())
    if lift.hand_held:
        return lift_hands_to_full_body(spec, solution, load_seed_pose(), lift.lift)
    if args.geometry != "back_bar":
        return lift_to_full_body(spec, solution, load_seed_pose())
    geometry = BackBarGeometry.from_standard(BACK_SQUAT, height_m=args.height)
    load = (
        LoadBalance(Barbell(float(solution.source["bar_mass_kg"])))
        if args.load_balance
        else None
    )
    return lift_to_full_body(
        spec, solution, load_seed_pose(), geometry=geometry, load=load
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    lift_recipe = recipe(args.lift)
    out = lift_recipe.bundle_path if args.out is None else args.out
    solution = solve_sagittal_lift(
        lift_recipe.exercise,
        bar_convention=lift_recipe.bar_convention,
        body_mass_kg=args.body_mass,
        height_m=args.height,
        bar_mass_kg=args.bar_mass or lift_recipe.bar_mass_kg,
        duration_s=args.duration or lift_recipe.duration_s,
        n_eval=args.n_eval,
    )
    lift = _full_body(args, lift_recipe, solution)
    extra: dict[str, Any] = {}
    if isinstance(lift, BenchLift):
        extra = {"bench": dict(lift.bench), "limitations": lift.limitations}
        lift = lift.lift
    spec = lift_bundle_spec(lift) or LIFT_SPEC.read_bytes()
    for label, value in lift.max_residual_by_landmark().items():
        logger.info("IK max residual %-10s %.5f m", label, value)
    logger.info("IK max closure %.3e m", lift.closure_m.max())
    logger.info("IK min foot-sphere height %.3e m", lift.lowest_sphere_m.min())
    if lift.balance is not None:
        logger.info(
            "IK bar-load balance: %.4f m -> %.2e m in %d corrections",
            lift.balance["unbalanced_max_offset_m"],
            lift.balance["max_offset_m"],
            lift.balance["iterations"],
        )

    start = window_start_time(solution, lift_recipe.window_event, args.window)
    tools_pin = _git("ls-tree", "HEAD", "vendor/ud-tools").split()[2]
    bundle = build_lift_bundle(
        spec,
        solution,
        lift,
        lift_name=lift_recipe.bundle_name,
        window_start_s=start,
        window_s=args.window,
        provenance={
            "source_commit": _git("rev-parse", "HEAD"),
            "tools_pin": tools_pin,
            "seed": {
                "source": "address_seed_deg of the lift-body spec, root raised "
                "(lift_body.lift_seed_pose)",
                "path": str(LIFT_SPEC.relative_to(ROOT)),
                "sha256": _sha256(LIFT_SPEC),
            },
            "spec_path": str(LIFT_SPEC.relative_to(ROOT)),
            "generator": "scripts/generate_lift_bundle.py",
            "recipe": {
                "lift": lift_recipe.lift,
                "window_event": lift_recipe.window_event,
                "phase": lift_recipe.phase,
                "rationale": lift_recipe.rationale,
            },
            **extra,
        },
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    bundle.save(out)
    logger.info(
        "wrote %s: %d steps from %.3f s, %d bytes, max pose drift %.2e m",
        out,
        bundle.steps,
        start,
        out.stat().st_size,
        bundle.provenance["max_pose_drift_m"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
