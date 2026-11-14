"""Regression tests for root conftest os.name reporting and teardown guards (issue #15103)."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT_CONFTEST_PATH = REPO_ROOT / "conftest.py"


def _load_root_conftest() -> ModuleType:
    """Load the repository root conftest module cleanly for direct hook tests."""
    spec = importlib.util.spec_from_file_location(
        "ud_root_conftest_under_test", ROOT_CONFTEST_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load root conftest from {ROOT_CONFTEST_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_failure_reporting_and_teardown_survive_mutated_os_name(tmp_path: Path) -> None:
    """Failing with opposite os.name must not crash pytest or leak to later tests."""
    shutil.copy2(ROOT_CONFTEST_PATH, tmp_path / "conftest.py")
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")

    opposite = "posix" if os.name == "nt" else "nt"
    expected_host = os.name
    probe_file = tmp_path / "test_probe.py"
    probe_file.write_text(
        "\n".join(
            [
                "import os",
                "from pathlib import Path",
                "",
                "",
                "def test_patched_os_name_failure(monkeypatch):",
                f"    monkeypatch.setattr(os, 'name', {opposite!r})",
                "    assert 1 == 2",
                "",
                "",
                "def test_direct_os_name_mutation_without_monkeypatch():",
                f"    os.name = {opposite!r}",
                "",
                "",
                "def test_subsequent_test_sees_host_os_name():",
                f"    assert os.name == {expected_host!r}",
                "    assert Path.cwd().exists()",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:randomly",
            "-o",
            "addopts=",
            str(probe_file),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = f"{result.stdout}\n{result.stderr}"

    assert result.returncode == 1, combined
    assert "INTERNALERROR" not in combined, combined
    assert "NotImplementedError" not in combined, combined
    assert "UnsupportedOperation" not in combined, combined
    assert "1 failed, 2 passed" in combined, combined


def test_root_conftest_hooks_restore_os_name_and_preserve_monkeypatch_stack() -> None:
    """makereport restores os.name only during yield; teardown clears direct leaks."""
    module = _load_root_conftest()
    host_name = os.name
    opposite = "posix" if host_name == "nt" else "nt"

    assert hasattr(module, "pytest_runtest_makereport")
    assert hasattr(module, "pytest_runtest_teardown")

    original = module._os.name
    try:
        module._os.name = opposite
        hook_gen = module.pytest_runtest_makereport(item=None, call=None)
        next(hook_gen)
        assert module._os.name == host_name

        with pytest.raises(StopIteration):
            hook_gen.send(None)
        assert module._os.name == opposite

        module.pytest_runtest_teardown(item=None, nextitem=None)
        assert module._os.name == host_name
    finally:
        module._os.name = original
