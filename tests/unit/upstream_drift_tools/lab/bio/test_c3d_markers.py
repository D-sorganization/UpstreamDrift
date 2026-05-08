"""Unit tests for ``validate_marker_positions`` heuristic + cp1252 safety.

Pins the fixes from PR #4582:

1. The min-position heuristic must NOT warn on swing-data minima around
   ``-1.97 m`` (negative values up to ~2 m are normal when the world
   origin is at the target).
2. It MUST warn on sub-millimetre POSITIVE minima (``< 1 mm``), which
   are the canonical sign of a missed mm-to-m conversion.
3. The warning message must encode under cp1252 — the Windows console
   default encoding — without raising ``UnicodeEncodeError``.
"""

from __future__ import annotations

import codecs
import logging

import numpy as np
import pytest
from src.shared.python.upstream_drift_tools.lab.bio._c3d_markers import (
    validate_marker_positions,
)

WARNING_LOGGER = "src.shared.python.upstream_drift_tools.lab.bio._c3d_markers"


def test_min_negative_one_point_nine_seven_does_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``min = -1.97 m`` (typical swing-data origin) must not trigger the warning."""
    coords = np.array(
        [
            [-1.97, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [1.2, 0.3, 0.7],
        ],
        dtype=float,
    )
    with caplog.at_level(logging.WARNING, logger=WARNING_LOGGER):
        validate_marker_positions(coords, source_units="m", target_units="m")
    suspect = [
        rec
        for rec in caplog.records
        if "Suspiciously small marker positions" in rec.getMessage()
    ]
    assert suspect == []


def test_min_sub_millimetre_positive_does_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``min = +0.0005 m`` (sub-mm, positive) MUST trigger the unit-error warning."""
    coords = np.array(
        [
            [0.0005, 0.001, 0.002],
            [0.0008, 0.0009, 0.0010],
        ],
        dtype=float,
    )
    with caplog.at_level(logging.WARNING, logger=WARNING_LOGGER):
        validate_marker_positions(coords, source_units="mm", target_units="m")
    suspect = [
        rec
        for rec in caplog.records
        if "Suspiciously small marker positions" in rec.getMessage()
    ]
    assert len(suspect) == 1


def test_warning_message_encodes_under_cp1252(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The emitted warning must round-trip through cp1252 (Windows console default)."""
    coords = np.array([[0.0005, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float)
    with caplog.at_level(logging.WARNING, logger=WARNING_LOGGER):
        validate_marker_positions(coords, source_units="mm", target_units="m")
    formatted = "\n".join(rec.getMessage() for rec in caplog.records)
    assert formatted, "expected at least one warning record"
    try:
        codecs.encode(formatted, "cp1252")
    except UnicodeEncodeError as exc:  # pragma: no cover - regression sentinel
        pytest.fail(f"warning message is not cp1252-safe: {exc}")
