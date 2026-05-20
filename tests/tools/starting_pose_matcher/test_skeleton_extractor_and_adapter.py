"""Tests for skeleton_extractor.py, _embed_adapter.py, __init__.py, __main__.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.tools.starting_pose_matcher import core, skeleton_extractor


# ---------------------------------------------------------------------------
# JsonSkeletonExtractor
# ---------------------------------------------------------------------------


def test_json_extractor_list_poses_default():
    e = skeleton_extractor.JsonSkeletonExtractor("/no/such/dir")
    assert e.list_poses() == ["TopofBackswing", "Impact"]


def test_json_extractor_list_poses_custom():
    e = skeleton_extractor.JsonSkeletonExtractor("/x", poses=("A", "B"))
    assert e.list_poses() == ["A", "B"]


def test_json_extractor_loads_existing_file(tmp_path: Path):
    p = tmp_path / "simscape_skeleton_Impact.json"
    p.write_text(
        json.dumps({"pose": "Impact", "joints": {"hip": [0, 0, 0]}, "segments": []})
    )
    e = skeleton_extractor.JsonSkeletonExtractor(tmp_path)
    s = e.get_skeleton("Impact")
    assert s.name == "Impact"


def test_json_extractor_missing_file_uses_fallback(tmp_path: Path):
    e = skeleton_extractor.JsonSkeletonExtractor(tmp_path)
    s = e.get_skeleton("Impact")
    assert isinstance(s, core.Skeleton)
    assert s.joints  # FK fallback produces joints


def test_skeleton_extractor_is_abstract():
    with pytest.raises(TypeError):
        skeleton_extractor.SkeletonExtractor()  # type: ignore[abstract]


def test_skeleton_extractor_module_exports():
    assert "SkeletonExtractor" in skeleton_extractor.__all__
    assert "JsonSkeletonExtractor" in skeleton_extractor.__all__
    assert "fallback_skeleton" in skeleton_extractor.__all__


# ---------------------------------------------------------------------------
# Package __init__ re-exports
# ---------------------------------------------------------------------------


def test_package_reexports_core_names():
    import src.tools.starting_pose_matcher as pkg

    for name in [
        "CM_TO_M",
        "MocapEvents",
        "Skeleton",
        "RigidTransform",
        "SkeletonTrajectory",
        "PoseSlot",
        "load_skeleton",
        "load_simscape_trajectory_csv",
        "solve_shaft_rz_deg",
        "phase_display_label",
        "phase_key_from_label",
        "SESSION_SCHEMA_VERSION",
    ]:
        assert hasattr(pkg, name), name


# ---------------------------------------------------------------------------
# Embed adapter
# ---------------------------------------------------------------------------


def test_embed_adapter_tool_id_and_capabilities():
    from src.tools.starting_pose_matcher._embed_adapter import (
        _MotionMatchPreviewEmbedAdapter,
    )

    a = _MotionMatchPreviewEmbedAdapter()
    assert a.tool_id == "motion_target_preview"
    caps = a.embed_capabilities()
    assert caps.supports_embedded is True
    assert caps.prefers_dock is False
    assert caps.requires_separate_qapplication is False
    assert caps.min_size == (1024, 720)


def test_embed_adapter_cleanup_calls_widget_cleanup():
    from src.tools.starting_pose_matcher._embed_adapter import (
        _MotionMatchPreviewEmbedAdapter,
    )

    a = _MotionMatchPreviewEmbedAdapter()
    w = MagicMock()
    a._widgets.append(w)
    a.cleanup()
    w.cleanup.assert_called_once()
    assert a._widgets == []


def test_embed_adapter_cleanup_swallows_widget_errors():
    from src.tools.starting_pose_matcher._embed_adapter import (
        _MotionMatchPreviewEmbedAdapter,
    )

    a = _MotionMatchPreviewEmbedAdapter()
    w = MagicMock()
    w.cleanup.side_effect = RuntimeError("boom")
    a._widgets.append(w)
    a.cleanup()  # must not raise


def test_embed_adapter_is_dirty_returns_false():
    from src.tools.starting_pose_matcher._embed_adapter import (
        _MotionMatchPreviewEmbedAdapter,
    )

    assert _MotionMatchPreviewEmbedAdapter().is_dirty() is False


def test_embed_adapter_create_main_widget_lazy_imports():
    """``create_main_widget`` should defer importing the heavy gui_main_widget."""
    from src.tools.starting_pose_matcher._embed_adapter import (
        _MotionMatchPreviewEmbedAdapter,
    )

    a = _MotionMatchPreviewEmbedAdapter()

    fake_widget = MagicMock(name="fake-MainWidget")

    class FakeMW:
        def __init__(self, parent):
            self.parent = parent
            fake_widget.parent = parent

        def __new__(cls, parent):
            return fake_widget

    fake_module = MagicMock()
    fake_module.MainWidget = FakeMW
    with patch.dict(
        sys.modules,
        {"src.tools.starting_pose_matcher.gui_main_widget": fake_module},
    ):
        w = a.create_main_widget(parent="P")
    assert w is fake_widget
    assert a._widgets == [fake_widget]


# ---------------------------------------------------------------------------
# __main__ module
# ---------------------------------------------------------------------------


def test_main_returns_1_when_gui_import_fails():
    from src.tools.starting_pose_matcher import __main__ as m

    real_import = (
        __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__
    )

    def fake_import(name, *a, **kw):
        if name == "src.tools.starting_pose_matcher.gui":
            raise ImportError("no PyQt6")
        return real_import(name, *a, **kw)

    with patch("builtins.__import__", side_effect=fake_import):
        rc = m.main()
    assert rc == 1


def test_main_delegates_when_gui_importable():
    from src.tools.starting_pose_matcher import __main__ as m

    fake_gui = MagicMock()
    fake_gui.main.return_value = 0
    with patch.dict(sys.modules, {"src.tools.starting_pose_matcher.gui": fake_gui}):
        rc = m.main()
    assert rc == 0
    fake_gui.main.assert_called_once()


def test_get_dockable_ui_delegates():
    from src.tools.starting_pose_matcher import __main__ as m

    sentinel = object()
    fake_gui = MagicMock()
    fake_gui.get_dockable_ui.return_value = sentinel
    with patch.dict(sys.modules, {"src.tools.starting_pose_matcher.gui": fake_gui}):
        assert m.get_dockable_ui() is sentinel
