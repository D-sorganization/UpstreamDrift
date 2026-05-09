"""Reference-pose library save / list / reload coverage."""

from __future__ import annotations

import pytest

from src.shared.python.pose_interchange.canonical import (
    CanonicalPose,
    canonical_from_reference_setup,
)
from src.shared.python.pose_interchange import pose_io as pose_io_mod
from src.shared.python.pose_interchange.pose_io import (
    list_saved_reference_poses,
    save_reference_pose,
)

pytestmark = pytest.mark.unit


@pytest.fixture()
def isolated_library(tmp_path, monkeypatch):
    """Redirect the on-disk library to a per-test tmpdir."""
    monkeypatch.setattr(pose_io_mod, "_REFERENCE_POSE_LIBRARY", tmp_path)
    return tmp_path


def test_save_then_list_then_reload(isolated_library) -> None:
    pose = canonical_from_reference_setup()
    name = "test_pose"
    written_path = save_reference_pose(pose, name)
    assert written_path.exists()
    assert written_path.name == f"{name}.json"

    listing = list_saved_reference_poses()
    assert name in listing

    reloaded = CanonicalPose.from_path(written_path)
    # Numpy arrays inside the dataclass make ``==`` ambiguous; compare
    # the canonical JSON forms (round-trip exact).
    assert reloaded.to_json() == pose.to_json()


def test_list_returns_empty_when_directory_absent(tmp_path, monkeypatch) -> None:
    missing = tmp_path / "does_not_exist"
    monkeypatch.setattr(pose_io_mod, "_REFERENCE_POSE_LIBRARY", missing)
    assert list_saved_reference_poses() == []


def test_save_rejects_path_separators(isolated_library) -> None:
    pose = canonical_from_reference_setup()
    with pytest.raises(ValueError, match="path separators"):
        save_reference_pose(pose, "../escape")


def test_save_rejects_empty_name(isolated_library) -> None:
    pose = canonical_from_reference_setup()
    with pytest.raises(ValueError, match="non-empty string"):
        save_reference_pose(pose, "")
