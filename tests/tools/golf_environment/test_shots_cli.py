"""Tests for the shots CLI helpers (epic #15150)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_cli_ball_shift_broadcasts_or_matches_roll_count() -> None:
    from src.tools.golf_environment import shots_cli

    assert shots_cli._per_shot(None, 3, "--ball-shift") == [0.0] * 3
    assert shots_cli._per_shot([0.01, 0.02], 2, "--ball-shift") == [0.01, 0.02]
    with pytest.raises(ValueError):
        shots_cli._per_shot([0.01, 0.02], 3, "--ball-shift")
