from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ci.check_noop_tests import find_noop_tests, main

pytestmark = pytest.mark.unit


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_pyproject(root: Path, *testpaths: str) -> None:
    quoted = ", ".join(f'"{testpath}"' for testpath in testpaths)
    _write(
        root / "pyproject.toml",
        "[tool.pytest.ini_options]\n"
        f"testpaths = [{quoted}]\n"
        'python_files = "test_*.py"\n',
    )


def test_guard_rejects_assert_true_placeholder(tmp_path: Path) -> None:
    _write_pyproject(tmp_path, "tests", "src/pkg/tests")
    _write(
        tmp_path / "src" / "pkg" / "tests" / "test_simulator.py",
        "def test_placeholder():\n    assert True\n",
    )

    findings = find_noop_tests(tmp_path)

    assert findings == [
        "src/pkg/tests/test_simulator.py:1: test_placeholder is a no-op "
        "placeholder (assert-true)"
    ]


def test_guard_rejects_docstring_plus_pass(tmp_path: Path) -> None:
    _write_pyproject(tmp_path, "tests")
    _write(
        tmp_path / "tests" / "test_contract.py",
        'def test_placeholder():\n    """placeholder."""\n    pass\n',
    )

    assert "placeholder (pass)" in find_noop_tests(tmp_path)[0]


def test_guard_accepts_real_no_exception_test(tmp_path: Path) -> None:
    _write_pyproject(tmp_path, "tests")
    _write(
        tmp_path / "tests" / "test_contract.py",
        "def test_does_not_raise():\n"
        "    value = int('1')\n"
        "    assert True  # documents no exception after real setup\n",
    )

    assert find_noop_tests(tmp_path) == []


def test_guard_ignores_non_test_helpers(tmp_path: Path) -> None:
    _write_pyproject(tmp_path, "tests")
    _write(
        tmp_path / "tests" / "test_contract.py",
        "def helper():\n    pass\ndef test_real():\n    assert 1 + 1 == 2\n",
    )

    assert find_noop_tests(tmp_path) == []


def test_main_returns_nonzero_for_noop_test(tmp_path: Path) -> None:
    _write_pyproject(tmp_path, "tests")
    _write(
        tmp_path / "tests" / "test_contract.py", "def test_placeholder():\n    ...\n"
    )

    assert main(["--repo-root", str(tmp_path)]) == 1
