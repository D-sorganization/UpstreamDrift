"""The shared pytest configuration must load in plugin-less venvs.

``pyproject.toml`` declares ini keys owned by pytest-asyncio and pytest-timeout
and runs with ``--strict-config``. CI lanes that install only ``pytest`` (the
hash-locked articulated-authority lane, the rolling MuJoCo/Pinocchio lane) must
still be able to collect tests, so the root ``conftest.py`` registers those keys
when their plugin is absent. ``-p no:<plugin>`` reproduces the plugin-less
environment inside a venv that has the plugins installed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CONFTEST = ROOT / "conftest.py"


def _run_pytest(*extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            *extra,
            "--collect-only",
            "-q",
            str(Path(__file__)),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def test_collection_survives_without_asyncio_and_timeout_plugins() -> None:
    result = _run_pytest("-p", "no:asyncio", "-p", "no:timeout")
    combined = result.stdout + result.stderr
    assert "Unknown config option" not in combined, combined
    assert result.returncode == 0, combined


class _FakePluginManager:
    def __init__(self, active: set[str]) -> None:
        self._active = active

    def has_plugin(self, name: str) -> bool:
        return name in self._active


def _load_root_conftest():
    import importlib.util

    spec = importlib.util.spec_from_file_location("root_conftest", CONFTEST)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("active", "expected"),
    [
        (set(), {"asyncio_mode", "timeout", "timeout_method"}),
        ({"asyncio"}, {"timeout", "timeout_method"}),
        ({"timeout"}, {"asyncio_mode"}),
        ({"asyncio", "timeout"}, set()),
    ],
)
def test_shim_registers_only_keys_of_inactive_plugins(
    active: set[str], expected: set[str]
) -> None:
    module = _load_root_conftest()
    keys = module._missing_plugin_ini_keys(_FakePluginManager(active))
    assert {name for name, _, _ in keys} == expected


def test_collection_unchanged_with_plugins_active() -> None:
    result = _run_pytest()
    combined = result.stdout + result.stderr
    assert "Unknown config option" not in combined, combined
    assert result.returncode == 0, combined
