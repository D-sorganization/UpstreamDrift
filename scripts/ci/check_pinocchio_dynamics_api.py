#!/usr/bin/env python3
"""Fail unless ``import pinocchio`` resolves to the robotics dynamics API."""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Any

REQUIRED_DYNAMICS_API: tuple[str, ...] = (
    "Model",
    "JointModelFreeFlyer",
    "SE3",
    "Inertia",
    "crba",
    "rnea",
    "computeCoriolisMatrix",
)


def missing_dynamics_api(pinocchio_module: Any) -> tuple[str, ...]:
    """Return required Pinocchio robotics symbols absent from a module."""
    return tuple(
        name for name in REQUIRED_DYNAMICS_API if not hasattr(pinocchio_module, name)
    )


def check_pinocchio_dynamics_api(module_name: str = "pinocchio") -> int:
    """Return 0 only when the imported module exposes required dynamics APIs."""
    try:
        pinocchio = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - diagnostic command
        print(f"Pinocchio import failed: {exc}", file=sys.stderr)
        return 1

    missing = missing_dynamics_api(pinocchio)
    if missing:
        module_file = getattr(pinocchio, "__file__", "<unknown>")
        module_version = getattr(pinocchio, "__version__", "<unknown>")
        print(
            "Pinocchio import resolved to a module without required dynamics APIs: "
            f"missing={list(missing)}, file={module_file}, version={module_version}",
            file=sys.stderr,
        )
        return 1

    print(
        "OK: Pinocchio dynamics API available "
        f"(version={getattr(pinocchio, '__version__', '<unknown>')})."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify that import pinocchio exposes robotics dynamics APIs."
    )
    parser.add_argument("--module", default="pinocchio")
    args = parser.parse_args(argv)
    return check_pinocchio_dynamics_api(args.module)


if __name__ == "__main__":
    raise SystemExit(main())
