"""Coverage tests for ``load_club_target`` and ``load_body_target`` dispatch."""

from __future__ import annotations

from pathlib import Path

import pytest
from src.shared.python.motion_matching.load_body_target import load_body_target
from src.shared.python.motion_matching.load_club_target import load_club_target


def test_load_club_target_unknown_format(tmp_path: Path) -> None:
    """Pin: unknown extension rejected with allowed-list message."""
    p = tmp_path / "x.bin"
    p.write_bytes(b"")
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_club_target(p)


def test_load_club_target_excel_requires_sheet(tmp_path: Path) -> None:
    """Pin: xlsx without ``sheet=`` rejected before opening the file."""
    p = tmp_path / "x.xlsx"
    p.write_bytes(b"")
    with pytest.raises(ValueError, match="sheet= is required"):
        load_club_target(p)


def test_load_club_target_dispatch_table() -> None:
    """Pin: the suffix sets are mutually exclusive and case-insensitive."""
    # Use a case-shifted suffix to exercise the lower() branch.
    with pytest.raises(ValueError, match="sheet="):
        load_club_target(Path("missing.XLSX"))


def test_load_body_target_unknown_format(tmp_path: Path) -> None:
    """Pin: unknown body-target extension rejected."""
    p = tmp_path / "x.bin"
    p.write_bytes(b"")
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_body_target(p)
