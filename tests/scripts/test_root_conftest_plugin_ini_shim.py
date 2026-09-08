"""The shared pytest configuration must load in plugin-less venvs.

``pyproject.toml`` declares ini keys owned by pytest-asyncio and pytest-timeout
and runs with ``--strict-config``. CI lanes that install only ``pytest`` (the
hash-locked articulated-authority lane, the rolling MuJoCo/Pinocchio lane) must
still be able to collect tests, so the root ``conftest.py`` registers those keys
when their plugin is inactive.

The mechanism is exercised with ``pytester`` in an isolated temporary rootdir
that carries a copy of the real root conftest and the same three ini keys under
``--strict-config``. ``-p no:<plugin>`` reproduces the plugin-less environment
inside a venv that has the plugins installed, without collecting this
repository's heavy test tree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]
pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CONFTEST = ROOT / "conftest.py"

_PYPROJECT = """
[tool.pytest.ini_options]
addopts = "--strict-config -p no:cacheprovider"
asyncio_mode = "auto"
timeout = 60
timeout_method = "thread"
"""

_TRIVIAL_TEST = """
def test_ok():
    assert True
"""


def _isolated_repo(pytester: pytest.Pytester) -> None:
    pytester.makeconftest(CONFTEST.read_text(encoding="utf-8"))
    pytester.makefile(".toml", pyproject=_PYPROJECT)
    pytester.makepyfile(test_trivial=_TRIVIAL_TEST)


def test_collection_survives_without_asyncio_and_timeout_plugins(
    pytester: pytest.Pytester,
) -> None:
    _isolated_repo(pytester)
    result = pytester.runpytest_inprocess("-p", "no:asyncio", "-p", "no:timeout", "-q")
    combined = "\n".join(result.outlines + result.errlines)
    assert "Unknown config option" not in combined, combined
    assert result.ret == 0, combined
    result.assert_outcomes(passed=1)


def test_collection_unchanged_with_plugins_active(pytester: pytest.Pytester) -> None:
    _isolated_repo(pytester)
    result = pytester.runpytest_inprocess("-q")
    combined = "\n".join(result.outlines + result.errlines)
    assert "Unknown config option" not in combined, combined
    assert result.ret == 0, combined
    result.assert_outcomes(passed=1)


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
