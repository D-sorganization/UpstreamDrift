"""Packaged ground-supported full-body pipeline runner (GS-0 to GS-5, #10520).

Delegates directly to ``src.shared.python.motion_matching.pipeline.cli``.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.shared.python.motion_matching.pipeline.cli import (
    build_parser,
    run_pipeline,
)

__all__ = ["build_parser", "main", "run_pipeline"]


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for packaged ground support driver."""
    parser = build_parser()
    args = parser.parse_args(argv)
    run_pipeline(args)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
