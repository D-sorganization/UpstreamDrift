"""Ground-supported full-body pipeline runner (GS-0 to GS-5).

Delegates directly to the engine-independent ``src.shared.python.motion_matching.pipeline.cli``
runner while preserving backward compatibility for existing evidence workflows.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
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


def main() -> None:
    """Entrypoint preserving defaults for ground support evidence run."""
    parser = build_parser()
    parser.set_defaults(out=HERE)
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
