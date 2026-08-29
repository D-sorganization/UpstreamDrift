"""Regression coverage for Motion-Match Preview launcher containment (#8066)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.launchers.launcher_model_handlers import SpecialAppHandler


@dataclass(frozen=True)
class _MotionMatchPreviewModel:
    """Minimal tile configuration for the package entry-point launch path."""

    id: str = "motion_target_preview"
    name: str = "Motion-Match Preview"
    path: str = "src/tools/starting_pose_matcher/__main__.py"
    type: str = "special_app"
    source_root: str | None = None
    working_dir: str | None = None
    python_paths: tuple[str, ...] = ()


@pytest.mark.unit
def test_motion_match_preview_never_executes_package_main_in_launcher(
    tmp_path: Path,
) -> None:
    """Contain a missing optional C3D dependency instead of terminating the launcher.

    The historical dockable-UI probe executed ``__main__.py`` in-process.
    That ran the Qt CLI dispatcher, allowing the unavailable
    ``sidekick.lab.bio._c3d_marker_set`` dependency to terminate the launcher.
    Package entry points must be delegated to a child process rather than probed.
    """
    entrypoint = tmp_path / "src" / "tools" / "starting_pose_matcher" / "__main__.py"
    entrypoint.parent.mkdir(parents=True)
    sentinel = tmp_path / "launcher-was-terminated"
    entrypoint.write_text(
        "from pathlib import Path\n"
        f"Path({str(sentinel)!r}).touch()\n"
        "from sidekick.lab.bio._c3d_marker_set import MarkerSet\n"
        "raise SystemExit(1)\n",
        encoding="utf-8",
    )

    from src.shared.python.launcher_embed import EMBEDDABLE_TOOL_REGISTRY

    missing = object()
    previous = EMBEDDABLE_TOOL_REGISTRY.pop("motion_target_preview", missing)
    try:
        widget = SpecialAppHandler().get_dockable_ui(
            _MotionMatchPreviewModel(), tmp_path
        )

        assert widget is None
        assert not sentinel.exists()
    finally:
        EMBEDDABLE_TOOL_REGISTRY.pop("motion_target_preview", None)
        if previous is not missing:
            EMBEDDABLE_TOOL_REGISTRY["motion_target_preview"] = previous


@pytest.mark.unit
def test_motion_match_preview_uses_local_source_package_module(
    tmp_path: Path,
) -> None:
    """The child process must bypass the root-level ``tools`` package shadow."""
    entrypoint = tmp_path / "src" / "tools" / "starting_pose_matcher" / "__main__.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("# package entry point\n", encoding="utf-8")
    process_manager = MagicMock()
    process_manager.launch_module.return_value = MagicMock()

    assert SpecialAppHandler().launch(
        _MotionMatchPreviewModel(), tmp_path, process_manager
    )

    process_manager.launch_module.assert_called_once_with(
        name="Motion-Match Preview",
        module_name="src.tools.starting_pose_matcher",
        cwd=tmp_path.resolve(),
        extra_python_paths=(),
    )
    process_manager.launch_script.assert_not_called()
