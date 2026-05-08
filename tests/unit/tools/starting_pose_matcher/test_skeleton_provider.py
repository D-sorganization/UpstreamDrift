"""Coverage tests for ``starting_pose_matcher.skeleton_provider``.

Test-only; no production code changes (issue #4673).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.tools.starting_pose_matcher.core import Skeleton, fallback_skeleton
from src.tools.starting_pose_matcher.skeleton_provider import (
    JsonSkeletonProvider,
    SkeletonProvider,
)


pytestmark = pytest.mark.unit


def test_skeleton_provider_is_abstract():
    with pytest.raises(TypeError):
        SkeletonProvider()  # type: ignore[abstract]


def test_json_provider_default_poses(tmp_path: Path):
    provider = JsonSkeletonProvider(json_dir=tmp_path)
    assert provider.list_poses() == ["TopofBackswing", "Impact"]


def test_json_provider_custom_poses(tmp_path: Path):
    provider = JsonSkeletonProvider(json_dir=tmp_path, poses=("A", "B", "C"))
    assert provider.list_poses() == ["A", "B", "C"]


def test_json_provider_get_skeleton_loads_existing_json(tmp_path: Path):
    blob = {
        "pose": "Impact",
        "joints": {"hip": [0.0, 0.0, 0.0]},
        "segments": [],
    }
    (tmp_path / "simscape_skeleton_Impact.json").write_text(json.dumps(blob))
    provider = JsonSkeletonProvider(json_dir=tmp_path)
    skel = provider.get_skeleton("Impact")
    assert isinstance(skel, Skeleton)
    assert skel.name == "Impact"
    assert "hip" in skel.joints


def test_json_provider_get_skeleton_falls_back_when_file_missing(tmp_path: Path):
    provider = JsonSkeletonProvider(json_dir=tmp_path)
    skel = provider.get_skeleton("Impact")
    # Falls back to FK-derived skeleton.
    assert isinstance(skel, Skeleton)
    fb = fallback_skeleton("Impact")
    assert set(skel.joints.keys()) == set(fb.joints.keys())


def test_json_provider_accepts_str_path(tmp_path: Path):
    provider = JsonSkeletonProvider(json_dir=str(tmp_path))
    skel = provider.get_skeleton("TopofBackswing")
    assert isinstance(skel, Skeleton)


class _ConcreteProvider(SkeletonProvider):
    def list_poses(self) -> list[str]:
        return ["only"]

    def get_skeleton(self, pose_name: str) -> Skeleton:
        return Skeleton(name=pose_name, joints={"hip": np.zeros(3)})


def test_skeleton_provider_subclass_works():
    p = _ConcreteProvider()
    assert p.list_poses() == ["only"]
    skel = p.get_skeleton("foo")
    assert skel.name == "foo"
