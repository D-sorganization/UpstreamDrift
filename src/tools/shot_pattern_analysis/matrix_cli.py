"""Common CLI declarations for original and supplemental matched matrices."""

from __future__ import annotations

import argparse
from pathlib import Path


def add_matrix_arguments(parser: argparse.ArgumentParser) -> None:
    """Append shared arguments in their established public order.

    The caller owns parser creation and any preceding positional arguments.
    Argparse retains responsibility for conversion, choices, and diagnostics.
    """
    parser.add_argument("output", type=Path)
    parser.add_argument("--stage", choices=("pilot", "final"), default="pilot")
    parser.add_argument("--shots", type=int, default=32)
    parser.add_argument("--acceptance", type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--figures", choices=("none", "selected", "all"), default="selected"
    )
