"""Ground-supported full-body pipeline runner (compatibility wrapper).

DEPRECATED (Issue #10520): This entry point in docs/ is deprecated.
Use ``python -m src.shared.python.motion_matching.execution.driver`` or
``src.shared.python.motion_matching.pipeline.cli`` instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shared.python.motion_matching.pipeline import (  # noqa: E402
    BUILD_RECEIPT,
    CANDIDATE,
    CAPTURES,
    SPEC,
    UPPER_SPEC,
)
from src.shared.python.motion_matching.pipeline.cli import (  # noqa: E402
    build_parser,
    run_pipeline,
)

HERE = Path(__file__).resolve().parent
C3D = CAPTURES["driver"]
OUT = HERE

__all__ = [
    "BUILD_RECEIPT",
    "C3D",
    "CANDIDATE",
    "CAPTURES",
    "HERE",
    "OUT",
    "SPEC",
    "UPPER_SPEC",
    "main",
]


def main(argv: Sequence[str] | None = None) -> None:
    """Entrypoint preserving defaults for ground support evidence run with deprecation warning."""
    warnings.warn(
        "docs/development/full_body_models/evidence/ground_support/run_ground_support.py is deprecated. "
        "Use 'python -m src.shared.python.motion_matching.execution.driver' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    parser = build_parser()
    parser.set_defaults(out=HERE)
    args = parser.parse_args(argv)
    run_pipeline(args)


if __name__ == "__main__":
    main()
