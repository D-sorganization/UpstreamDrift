from __future__ import annotations

from pathlib import Path
from typing import Any, NoReturn

import pytest

from installer.windows.packaging_profiles import get_packaging_profile
from installer.windows.setup_config import (
    build_setup_configuration,
    detect_available_engines,
)


def test_detect_available_engines_core_skips_optional_imports() -> None:
    imported: list[str] = []

    def fake_import(name: str) -> Any:
        imported.append(name)
        if name == "mujoco":
            return object()
        raise ImportError

    engines = detect_available_engines(
        get_packaging_profile("core"),
        importer=fake_import,
    )

    assert engines == ("mujoco",)
    assert imported == ["mujoco"]


def test_detect_available_engines_hybrid_includes_optional_imports() -> None:
    imported: list[str] = []

    def fake_import(name: str) -> Any:
        imported.append(name)
        if name in {"mujoco", "pydrake", "pinocchio"}:
            return object()
        raise ImportError

    config = build_setup_configuration(Path("C:/repo"), "hybrid", importer=fake_import)

    assert config.available_engines == ("mujoco", "drake", "pinocchio")
    assert imported == ["mujoco", "pydrake", "pinocchio", "myosuite", "opensim"]


def test_build_setup_configuration_core_omits_api_executable() -> None:
    def fake_import(name: str) -> Any:
        if name == "mujoco":
            return object()
        raise ImportError

    config = build_setup_configuration(Path("C:/repo"), "core", importer=fake_import)

    assert [exe.target_name for exe in config.executables] == ["GolfModelingSuite.exe"]
    assert config.build_exe_options["build_exe"] == "build/upstream_drift_core"
    assert config.bdist_msi_options["initial_target_dir"].endswith("\\core")


def test_build_setup_configuration_full_includes_api_executable() -> None:
    def fake_import(name: str) -> Any:
        if name in {"mujoco", "pydrake"}:
            return object()
        raise ImportError

    config = build_setup_configuration(Path("C:/repo"), "full", importer=fake_import)

    assert [exe.target_name for exe in config.executables] == [
        "GolfModelingSuite.exe",
        "GolfAPI.exe",
    ]
    assert config.profile.discovery_mode == "provider-first"
    assert "pydrake" in config.build_exe_options["packages"]


def test_build_setup_configuration_requires_core_engine() -> None:
    def missing_import(_: str) -> NoReturn:
        raise ImportError

    with pytest.raises(ImportError):
        build_setup_configuration(Path("C:/repo"), "core", importer=missing_import)
