"""Write offline grip-loss prediction reports for committed swing evidence (#15128)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shared.python.grip_contact.grip_loss_config import (
    GripBreakawayThresholds,
    GripLossPhysicsConfig,
)
from src.shared.python.grip_contact.grip_loss_evaluator import (
    GripLossEvaluator,
    serialise_grip_loss_report,
)
from src.shared.python.grip_contact.grip_margin import DEFAULT_NORMAL_FLOOR_N

EVIDENCE = ROOT / "docs/development/full_body_models/evidence/grip_kinetics"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--club", choices=("driver", "iron7"), action="append", required=True
    )
    ap.add_argument("--friction", type=float, required=True)
    ap.add_argument("--slip-enter", type=float, required=True)
    ap.add_argument("--slip-exit", type=float, required=True)
    ap.add_argument("--release-enter", type=float, required=True)
    ap.add_argument("--min-slip-dwell-s", type=float, required=True)
    ap.add_argument("--min-release-dwell-s", type=float, required=True)
    ap.add_argument("--squeeze-scale", type=float, default=1.0)
    ap.add_argument("--normal-floor-n", type=float, default=DEFAULT_NORMAL_FLOOR_N)
    ap.add_argument("--out-dir", type=Path, default=EVIDENCE)
    ap.add_argument(
        "--sensitivity-friction",
        type=float,
        nargs="*",
        default=(),
        help="optional friction scale factors for sensitivity rows",
    )
    ap.add_argument(
        "--sensitivity-squeeze",
        type=float,
        nargs="*",
        default=(),
        help="optional squeeze scale factors for sensitivity rows",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    thresholds = GripBreakawayThresholds(
        slip_enter_margin=args.slip_enter,
        slip_exit_margin=args.slip_exit,
        release_enter_margin=args.release_enter,
        min_slip_dwell_s=args.min_slip_dwell_s,
        min_release_dwell_s=args.min_release_dwell_s,
    )
    physics = GripLossPhysicsConfig(
        friction=args.friction,
        normal_floor_n=args.normal_floor_n,
        thresholds=thresholds,
        squeeze_scale=args.squeeze_scale,
    )
    evaluator = GripLossEvaluator(physics)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for club in args.club:
        if args.sensitivity_friction and args.sensitivity_squeeze:
            from src.shared.python.grip_contact.grip_loss_loads import (
                resolve_grip_load_provider,
            )

            loads = resolve_grip_load_provider("contact_run").load(
                club, evidence_root=out_dir
            )
            report = evaluator.evaluate_with_sensitivity(
                loads,
                args.sensitivity_friction,
                args.sensitivity_squeeze,
            )
        else:
            report = evaluator.evaluate_club(club, evidence_root=out_dir)
        payload = serialise_grip_loss_report(report)
        path = out_dir / f"grip_loss_prediction_{club}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
