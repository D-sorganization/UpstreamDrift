"""Regression test for issue #9266.

ADR-0044 documented ``ShotResult``'s force field as ``forces_n_m``, but
``src/bunkershot3d/solvers/shot.py`` has always exposed it as ``forces_n``
(with ``torques_n_m`` for the torque field, which really is in N*m). The
nonexistent identifier sends implementers to the wrong API.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

_ADR_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "adr"
    / "0044-out-of-plane-fidelity-for-bunkershot3d.md"
)

pytestmark = pytest.mark.unit


def test_adr_0044_does_not_reference_nonexistent_forces_field() -> None:
    text = _ADR_PATH.read_text(encoding="utf-8")

    assert "forces_n_m" not in text
    assert "`forces_n`" in text


def test_adr_0044_field_reference_matches_shot_result() -> None:
    """The corrected identifier must actually exist on ShotResult."""
    from src.bunkershot3d.solvers.shot import ShotResult

    field_names = {f.name for f in dataclasses.fields(ShotResult)}

    assert "forces_n" in field_names
    assert "forces_n_m" not in field_names
