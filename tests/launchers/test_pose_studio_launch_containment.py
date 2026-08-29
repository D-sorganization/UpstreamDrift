"""Regression coverage for Pose Studio launcher containment (#8070)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.launchers.launcher_model_handlers import SpecialAppHandler

pytestmark = pytest.mark.unit


@dataclass
class _PoseStudioModel:
    """Minimal manifest entry used by the special-app handler."""

    id: str = "pose_studio"
    name: str = "Pose Studio"
    path: str = "src/tools/pose_studio/__main__.py"
    type: str = "special_app"
    source_root: str | None = None
    working_dir: str | None = None
    python_paths: tuple[str, ...] = ()


def test_pose_studio_missing_marker_dependency_cannot_terminate_launcher(
    tmp_path: Path,
) -> None:
    """Never import a package CLI entry point while probing for a dockable UI.

    ``__main__.py`` dispatches via ``sys.exit(main())``.  If its optional C3D
    marker dependency is absent, importing it in the launcher process would
    therefore close the host rather than letting the child process report a
    recoverable launch failure.
    """
    entrypoint = tmp_path / "src" / "tools" / "pose_studio" / "__main__.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text(
        "from sidekick.lab.bio._c3d_marker_set import C3DMarkerSet\n"
        "raise SystemExit(1)\n",
        encoding="utf-8",
    )

    from src.shared.python.launcher_embed import EMBEDDABLE_TOOL_REGISTRY

    missing = object()
    previous = EMBEDDABLE_TOOL_REGISTRY.pop("pose_studio", missing)
    try:
        assert SpecialAppHandler().get_dockable_ui(_PoseStudioModel(), tmp_path) is None
    finally:
        EMBEDDABLE_TOOL_REGISTRY.pop("pose_studio", None)
        if previous is not missing:
            EMBEDDABLE_TOOL_REGISTRY["pose_studio"] = previous


def test_pose_studio_uses_a_child_package_module(
    tmp_path: Path,
) -> None:
    """Pose Studio starts out of process, preserving the native launcher."""
    entrypoint = tmp_path / "src" / "tools" / "pose_studio" / "__main__.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("# package entry point\n", encoding="utf-8")
    process_manager = MagicMock()
    process_manager.launch_module.return_value = MagicMock()

    assert SpecialAppHandler().launch(_PoseStudioModel(), tmp_path, process_manager)

    process_manager.launch_module.assert_called_once_with(
        name="Pose Studio",
        module_name="src.tools.pose_studio",
        cwd=tmp_path.resolve(),
        extra_python_paths=(),
    )
    process_manager.launch_script.assert_not_called()
