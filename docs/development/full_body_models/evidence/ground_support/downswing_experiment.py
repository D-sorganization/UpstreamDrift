"""Downswing tracking experiments on a finished ground-support run (compatibility wrapper).

DEPRECATED (Issue #10520): This entry point in docs/ is deprecated.
Use ``python -m src.shared.python.motion_matching.execution.downswing`` instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shared.python.motion_matching.execution.downswing import (  # noqa: E402
    build_parser,
    condition,
    main as _packaged_main,
    run_downswing_experiment,
)

__all__ = [
    "build_parser",
    "condition",
    "main",
    "run_downswing_experiment",
]


def main(argv: Sequence[str] | None = None) -> int:
    warnings.warn(
        "docs/development/full_body_models/evidence/ground_support/downswing_experiment.py is deprecated. "
        "Use 'python -m src.shared.python.motion_matching.execution.downswing' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _packaged_main(argv)


if __name__ == "__main__":
    sys.exit(main())
