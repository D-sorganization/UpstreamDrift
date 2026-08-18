"""Regression tests for Docker feature installation command generation."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_FEATURES_PATH = REPO_ROOT / "scripts" / "docker" / "install_features.py"


def _load_install_features_module():
    spec = importlib.util.spec_from_file_location(
        "install_features", INSTALL_FEATURES_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pip_feature_install_commands_strip_shell_quotes() -> None:
    install_features = _load_install_features_module()

    commands = install_features._feature_install_argv("mujoco", REPO_ROOT)

    assert commands == [["pip", "install", "--no-cache-dir", "mujoco>=3.2.3,<4.0.0"]]


def test_slim_profile_matches_core_runtime_contract() -> None:
    install_features = _load_install_features_module()

    profiles = install_features._load_profiles(REPO_ROOT / "docker" / "profiles.yaml")
    features = install_features._resolve_profile_features(profiles["profiles"], "slim")

    assert features == ["api", "pendulum", "mujoco"]


def test_profile_dry_run_works_with_modular_dockerfile_early_copy_set(
    tmp_path: Path,
) -> None:
    """Dockerfile.modular runs profile validation before copying the full source."""
    (tmp_path / "docker").mkdir()
    (tmp_path / "scripts" / "docker").mkdir(parents=True)
    (tmp_path / "src" / "shared" / "python").mkdir(parents=True)

    shutil.copy(REPO_ROOT / "docker" / "profiles.yaml", tmp_path / "docker")
    shutil.copy(
        INSTALL_FEATURES_PATH,
        tmp_path / "scripts" / "docker" / "install_features.py",
    )
    shutil.copy(REPO_ROOT / "src" / "__init__.py", tmp_path / "src" / "__init__.py")
    shutil.copy(
        REPO_ROOT / "src" / "shared" / "__init__.py",
        tmp_path / "src" / "shared" / "__init__.py",
    )
    shutil.copy(
        REPO_ROOT / "src" / "shared" / "python" / "__init__.py",
        tmp_path / "src" / "shared" / "python" / "__init__.py",
    )
    shutil.copytree(
        REPO_ROOT / "src" / "shared" / "python" / "engine_core",
        tmp_path / "src" / "shared" / "python" / "engine_core",
    )
    shutil.copytree(
        REPO_ROOT / "src" / "shared" / "python" / "feature_registry",
        tmp_path / "src" / "shared" / "python" / "feature_registry",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(tmp_path / "scripts" / "docker" / "install_features.py"),
            "--repo-root",
            str(tmp_path),
            "--profile",
            "standard",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


def test_bunkershot3d_imports_through_src_package_namespace() -> None:
    """Runtime Docker health checks import BunkerShot3D through ``src``."""
    import bunkershot3d.backends.chrono.driver as chrono_driver
    import bunkershot3d.backends.liggghts.driver as liggghts_driver
    import bunkershot3d.backends.mpm.driver as mpm_driver
    from bunkershot3d.calibration.angle_of_repose import AngleOfReposeExperiment

    assert chrono_driver.ChronoDriver.__name__ == "ChronoDriver"
    assert liggghts_driver.LiggghtsDriver.__name__ == "LiggghtsDriver"
    assert mpm_driver.MPMDriver.__name__ == "MPMDriver"
    assert AngleOfReposeExperiment.__name__ == "AngleOfReposeExperiment"
