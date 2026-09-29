#!/usr/bin/env python3
"""Compatibility entry point for the canonical UpstreamDrift launcher."""

from __future__ import annotations

import warnings

from launch_upstream_drift import parse_arguments, route_launch
from src.api._version import warn_if_unsupported_platform


def main() -> None:
    """Run the canonical launcher while preserving the legacy module path."""
    warnings.warn(
        "launch_golf_suite.py is deprecated; use launch_upstream_drift.py or "
        "the upstream-drift console script instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    warn_if_unsupported_platform()
    route_launch(parse_arguments())


if __name__ == "__main__":
    main()
