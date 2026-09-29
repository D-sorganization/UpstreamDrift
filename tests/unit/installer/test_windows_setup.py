from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

from installer.windows import setup_config
from installer.windows.packaging_profiles import get_packaging_profile


def _fake_setup_configuration(
    project_root: Path,
) -> setup_config.SetupProfileConfiguration:
    return setup_config.SetupProfileConfiguration(
        profile=get_packaging_profile(None),
        available_engines=(),
        build_exe_options={
            "include_files": [
                (str(project_root / "src" / "config"), "src/config"),
                (str(project_root / "src" / "shared" / "urdf"), "src/shared/urdf"),
                (
                    str(project_root / "src" / "launchers" / "assets"),
                    "src/launchers/assets",
                ),
            ]
        },
        bdist_msi_options={
            "install_icon": str(
                project_root / "src" / "launchers" / "assets" / "golf_robot_icon.ico"
            )
        },
        executables=(
            setup_config.ExecutableSpec(
                script=str(project_root / "launch_golf_suite.py"),
                base="Win32GUI",
                target_name="GolfModelingSuite.exe",
                icon=str(
                    project_root
                    / "src"
                    / "launchers"
                    / "assets"
                    / "golf_robot_icon.ico"
                ),
                shortcut_name="UpstreamDrift",
                shortcut_dir="DesktopFolder",
            ),
            setup_config.ExecutableSpec(
                script=str(project_root / "start_api_server.py"),
                base="Console",
                target_name="GolfAPI.exe",
                icon=str(
                    project_root
                    / "src"
                    / "launchers"
                    / "assets"
                    / "golf_robot_icon.ico"
                ),
            ),
        ),
    )


def _load_setup_module(monkeypatch) -> tuple[ModuleType, Any]:
    fake_cx_freeze = types.ModuleType("cx_Freeze")
    fake_cx_freeze.Executable = lambda *args, **kwargs: {
        "args": args,
        "kwargs": kwargs,
    }
    fake_cx_freeze.setup = MagicMock()
    monkeypatch.setitem(sys.modules, "cx_Freeze", fake_cx_freeze)
    monkeypatch.setattr(
        setup_config,
        "build_setup_configuration",
        lambda project_root, profile_name, **kwargs: _fake_setup_configuration(
            project_root
        ),
    )
    sys.modules.pop("installer.windows.setup", None)
    module = importlib.import_module("installer.windows.setup")
    return module, fake_cx_freeze


def test_setup_module_uses_real_src_layout(monkeypatch) -> None:
    module, _fake = _load_setup_module(monkeypatch)
    project_root = module.project_root
    config = setup_config.build_setup_configuration(
        project_root,
        None,
        importer=lambda name: object(),
    )

    include_files = config.build_exe_options["include_files"]
    assert (str(project_root / "src" / "config"), "src/config") in include_files
    assert (
        str(project_root / "src" / "shared" / "urdf"),
        "src/shared/urdf",
    ) in include_files
    assert (
        str(project_root / "src" / "launchers" / "assets"),
        "src/launchers/assets",
    ) in include_files
    assert config.bdist_msi_options["install_icon"] == str(
        project_root / "src" / "launchers" / "assets" / "golf_robot_icon.ico"
    )


def test_setup_main_builds_using_src_paths(monkeypatch) -> None:
    module, fake_cx_freeze = _load_setup_module(monkeypatch)
    fake_cx_freeze.setup.assert_called_once()

    kwargs = fake_cx_freeze.setup.call_args.kwargs
    build_exe_options = kwargs["options"]["build_exe"]
    executable_scripts = [item["kwargs"]["script"] for item in kwargs["executables"]]

    assert str(module.project_root / "launch_golf_suite.py") in executable_scripts
    assert str(module.project_root / "start_api_server.py") in executable_scripts
    assert (
        str(module.project_root / "src" / "config"),
        "src/config",
    ) in build_exe_options["include_files"]
    assert (
        str(module.project_root / "src" / "launchers" / "assets"),
        "src/launchers/assets",
    ) in build_exe_options["include_files"]
