"""Command-line interface for contraction and Floquet analysis."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np

from src.tools.contraction.verifier import (
    ContractionVerifier,
    linear_system_floquet_multipliers,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Contraction and Floquet tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    measure = subparsers.add_parser("measure", help="Estimate contraction rate.")
    measure.add_argument("--decay-rate", type=float, default=1.0)
    measure.add_argument("--trials", type=int, default=16)
    measure.add_argument("--perturbation-scale", type=float, default=1e-3)
    measure.add_argument("--horizon", type=float, default=1.0)
    measure.add_argument("--steps", type=int, default=100)

    floquet = subparsers.add_parser("floquet", help="Compute diagonal LTI multipliers.")
    floquet.add_argument("--rates", type=float, nargs="+", required=True)
    floquet.add_argument("--period", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload: dict[str, Any]
    if args.command == "measure":
        verifier = ContractionVerifier(
            decay_rate=args.decay_rate,
            horizon=args.horizon,
            n_steps=args.steps,
        )
        payload = verifier.verify(
            n_trials=args.trials,
            perturbation_scale=args.perturbation_scale,
        ).to_dict()
    else:
        system_matrix = -np.diag(np.asarray(args.rates, dtype=np.float64))
        multipliers = linear_system_floquet_multipliers(system_matrix, args.period)
        payload = {
            "period": args.period,
            "multipliers": [
                {"real": float(value.real), "imag": float(value.imag)}
                for value in multipliers
            ],
        }
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
