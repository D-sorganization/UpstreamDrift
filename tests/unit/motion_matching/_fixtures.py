"""Shared fixtures for motion-matching unit tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from src.shared.python.motion_matching.club_target import (
    ClubTarget,
    SourceProvenance,
)


def make_provenance(filename: str = "synthetic.bin") -> SourceProvenance:
    """A SourceProvenance with a real sha256 of an empty payload."""
    return SourceProvenance(
        filename=filename,
        format="synthetic",
        subject_id="UNIT",
        trial_id="0",
        sha256=hashlib.sha256(b"").hexdigest(),
    )


def make_target(n: int = 301) -> ClubTarget:
    """A small but valid ClubTarget for use in unit tests."""
    time = np.linspace(0.0, 0.3, n)
    butt = np.column_stack(
        [
            0.5 * np.cos(2 * np.pi * time),
            0.5 * np.sin(2 * np.pi * time),
            np.zeros_like(time),
        ]
    )
    clubhead = butt + np.array([0.0, 0.0, 1.1])
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    return ClubTarget(
        time=time,
        butt=butt,
        clubhead=clubhead,
        club_quat=quat,
        impact_idx=n // 2,
        source=make_provenance(),
    )


def repo_root() -> Path:
    """Walk up from this file to the repository root."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Could not locate repo root")
