"""``save_initial_state`` / ``load_initial_state`` reject unknown engines."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.python.pose_interchange.canonical import (
    canonical_from_reference_setup,
)
from src.shared.python.pose_interchange.pose_io import (
    load_initial_state,
    save_initial_state,
)

pytestmark = pytest.mark.unit


def test_save_rejects_unknown_engine(tmp_path: Path) -> None:
    pose = canonical_from_reference_setup()
    with pytest.raises(ValueError, match="not supported"):
        save_initial_state(pose, "totally-not-an-engine", tmp_path / "out")


def test_load_rejects_unknown_engine(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not supported"):
        load_initial_state("totally-not-an-engine", tmp_path / "out")


def test_save_rejects_non_string_engine(tmp_path: Path) -> None:
    pose = canonical_from_reference_setup()
    with pytest.raises(TypeError, match="engine must be a string"):
        save_initial_state(pose, 42, tmp_path / "out")  # type: ignore[arg-type]


def test_save_rejects_non_canonical_pose(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="pose must be a CanonicalPose"):
        save_initial_state(
            "not_a_pose",  # type: ignore[arg-type]
            "simscape",
            tmp_path / "out",
        )
