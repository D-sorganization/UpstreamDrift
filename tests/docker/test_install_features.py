"""Regression tests for Docker feature installation command generation."""

from __future__ import annotations

import importlib.util
from pathlib import Path


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
