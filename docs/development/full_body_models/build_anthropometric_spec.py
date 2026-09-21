"""Build the anthropometric full-body candidate document (compatibility wrapper).

DEPRECATED (Issue #10520): This entry point in docs/ is deprecated.
Use ``python -m src.shared.python.motion_matching.execution.spec_builder`` instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shared.python.motion_matching.execution.spec_builder import (  # noqa: E402
    build_anthropometric_spec,
    build_parser,
    main as _packaged_main,
    marker_seeds,
)

__all__ = [
    "build_anthropometric_spec",
    "build_parser",
    "main",
    "marker_seeds",
]


def main(argv: Sequence[str] | None = None) -> int:
    warnings.warn(
        "docs/development/full_body_models/build_anthropometric_spec.py is deprecated. "
        "Use 'python -m src.shared.python.motion_matching.execution.spec_builder' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _packaged_main(argv)


if __name__ == "__main__":
    sys.exit(main())
