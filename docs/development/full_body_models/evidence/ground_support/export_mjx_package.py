"""Export a finished ground-support run as a package for MJX (compatibility wrapper).

DEPRECATED (Issue #10520): This entry point in docs/ is deprecated.
Use ``python -m src.shared.python.motion_matching.execution.mjx_export`` instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shared.python.motion_matching.execution.mjx_export import (  # noqa: E402
    build_parser,
    export_mjx_package,
    main as _packaged_main,
    stiffen_weld,
)

__all__ = [
    "build_parser",
    "export_mjx_package",
    "main",
    "stiffen_weld",
]


def main(argv: Sequence[str] | None = None) -> int:
    warnings.warn(
        "docs/development/full_body_models/evidence/ground_support/export_mjx_package.py is deprecated. "
        "Use 'python -m src.shared.python.motion_matching.execution.mjx_export' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _packaged_main(argv)


if __name__ == "__main__":
    sys.exit(main())
