"""Optional Pinocchio ABA timing benchmark CLI."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AbaTimingResult:
    """Timing result for the Pinocchio Articulated Body Algorithm."""

    available: bool
    iterations: int
    mean_seconds: float | None
    min_seconds: float | None
    max_seconds: float | None
    reason: str | None = None

    def to_dict(self) -> dict[str, bool | float | int | str | None]:
        return {
            "available": self.available,
            "iterations": self.iterations,
            "mean_seconds": self.mean_seconds,
            "min_seconds": self.min_seconds,
            "max_seconds": self.max_seconds,
            "reason": self.reason,
        }


def run_aba_timing_benchmark(
    iterations: int = 100,
    pinocchio_module: Any | None = ...,
) -> AbaTimingResult:
    """Benchmark Pinocchio ABA when the optional backend is importable."""
    if iterations <= 0:
        return _unavailable("iterations must be positive")

    pin = _load_pinocchio() if pinocchio_module is ... else pinocchio_module
    if pin is None:
        return _unavailable("Pinocchio Python bindings are not installed")

    model = pin.buildSampleModelManipulator()
    data = model.createData()
    q = pin.neutral(model)
    v = np.zeros(model.nv)
    tau = np.zeros(model.nv)

    timings: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        pin.aba(model, data, q, v, tau)
        timings.append(time.perf_counter() - start)

    values = np.asarray(timings, dtype=np.float64)
    return AbaTimingResult(
        available=True,
        iterations=iterations,
        mean_seconds=float(np.mean(values)),
        min_seconds=float(np.min(values)),
        max_seconds=float(np.max(values)),
    )


def _load_pinocchio() -> Any | None:
    try:
        import pinocchio as pin
    except ImportError:
        return None
    return pin


def _unavailable(reason: str) -> AbaTimingResult:
    return AbaTimingResult(
        available=False,
        iterations=0,
        mean_seconds=None,
        min_seconds=None,
        max_seconds=None,
        reason=reason,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Pinocchio ABA timing.")
    parser.add_argument("--iterations", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_aba_timing_benchmark(iterations=args.iterations)
    sys.stdout.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")
    return 0 if result.available else 2


if __name__ == "__main__":
    raise SystemExit(main())
