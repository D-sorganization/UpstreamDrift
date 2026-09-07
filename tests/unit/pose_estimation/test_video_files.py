"""The GUIs' video filter admits the rig's .mkv recordings (#9611)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.shared.python.pose_estimation.video_files import (
    VIDEO_SUFFIXES,
    is_video_file,
    video_dialog_filter,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
GUIS = (
    ROOT / "src/shared/python/pose_estimation/mediapipe_gui.py",
    ROOT / "src/shared/python/pose_estimation/openpose_gui.py",
)


def test_filter_lists_every_suffix_and_all_files() -> None:
    text = video_dialog_filter()
    for suffix in VIDEO_SUFFIXES:
        assert f"*{suffix}" in text
    assert "*.mkv" in text and text.endswith(";;All Files (*)")


def test_is_video_file_is_suffix_based_and_case_insensitive() -> None:
    assert is_video_file("take.MKV") and is_video_file(Path("a/b.mp4"))
    assert not is_video_file("notes.txt") and not is_video_file("clip")


@pytest.mark.parametrize("gui", GUIS, ids=lambda p: p.stem)
def test_guis_use_the_shared_filter_not_a_literal(gui: Path) -> None:
    source = gui.read_text(encoding="utf-8")
    assert "video_dialog_filter()" in source
    literals = [
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    assert not any("*.mp4" in s for s in literals), "hard-coded video filter"
