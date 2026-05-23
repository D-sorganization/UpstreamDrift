"""Unit tests for src/tools/code_quality_check.py."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from src.tools import code_quality_check as cqc


@pytest.fixture
def py_file(tmp_path: Path) -> Path:
    """Return a Path inside tmp_path for ad-hoc source content."""
    return tmp_path / "module.py"


class TestColors:
    def test_class_constants_present(self) -> None:
        for attr in (
            "HEADER",
            "BLUE",
            "CYAN",
            "GREEN",
            "WARNING",
            "FAIL",
            "ENDC",
            "BOLD",
        ):
            assert hasattr(cqc.Colors, attr)


class TestIsLegitimatePassContext:
    def test_returns_false_for_empty_lines(self) -> None:
        assert cqc.is_legitimate_pass_context([], 1) is False

    def test_returns_false_when_line_not_pass(self) -> None:
        assert cqc.is_legitimate_pass_context(["x = 1"], 1) is False

    def test_returns_false_out_of_range(self) -> None:
        assert cqc.is_legitimate_pass_context(["pass"], 0) is False
        assert cqc.is_legitimate_pass_context(["pass"], 5) is False

    def test_class_body_is_legitimate(self) -> None:
        lines = ["class Foo:", "    pass"]
        assert cqc.is_legitimate_pass_context(lines, 2) is True

    def test_def_body_not_legitimate(self) -> None:
        lines = ["def foo():", "    pass"]
        assert cqc.is_legitimate_pass_context(lines, 2) is False

    def test_try_block_legitimate(self) -> None:
        lines = ["try:", "    pass"]
        assert cqc.is_legitimate_pass_context(lines, 2) is True

    def test_except_block_legitimate(self) -> None:
        lines = ["try:", "    x()", "except Exception:", "    pass"]
        assert cqc.is_legitimate_pass_context(lines, 4) is True

    def test_with_block_legitimate(self) -> None:
        lines = ["with open('x') as f:", "    pass"]
        assert cqc.is_legitimate_pass_context(lines, 2) is True

    def test_if_type_checking_legitimate(self) -> None:
        lines = ["if TYPE_CHECKING:", "    pass"]
        assert cqc.is_legitimate_pass_context(lines, 2) is True

    def test_invalid_args_raise(self) -> None:
        with pytest.raises(Exception):  # noqa: BLE001, PT011, B017
            cqc.is_legitimate_pass_context("not a list", 1)  # type: ignore[arg-type]

    def test_try_within_5_lines(self) -> None:
        # try is within the second sweep (5-line window) without colon-end keyword
        lines = ["try:", "    a()", "    b()", "    c()", "    pass"]
        assert cqc.is_legitimate_pass_context(lines, 5) is True

    def test_with_within_3_lines(self) -> None:
        lines = ["with open('x') as f:", "    a()", "    pass"]
        assert cqc.is_legitimate_pass_context(lines, 3) is True

    def test_none_lines_raises(self) -> None:
        with pytest.raises(ValueError, match="lines must be provided"):
            cqc.is_legitimate_pass_context(None, 1)  # type: ignore[arg-type]


class TestCheckBannedPatterns:
    def test_detects_todo(self, py_file: Path) -> None:
        issues = cqc.check_banned_patterns(["# TO" + "DO: fix"], py_file)
        assert any(("TO" + "DO") in msg for _, msg, _ in issues)

    def test_detects_fixme(self, py_file: Path) -> None:
        issues = cqc.check_banned_patterns(["# FIX" + "ME: yes"], py_file)
        assert any(("FIX" + "ME") in msg for _, msg, _ in issues)

    def test_detects_ellipsis(self, py_file: Path) -> None:
        issues = cqc.check_banned_patterns(["    ...", "x = 1"], py_file)
        assert any("Ellipsis" in m for _, m, _ in issues)

    def test_detects_not_implemented(self, py_file: Path) -> None:
        issues = cqc.check_banned_patterns(["raise NotImplementedError()"], py_file)
        assert any("NotImplementedError" in m for _, m, _ in issues)

    def test_detects_template_placeholder(self, py_file: Path) -> None:
        issues = cqc.check_banned_patterns(["# your name here"], py_file)
        assert issues

    def test_detects_orphan_pass(self, py_file: Path) -> None:
        issues = cqc.check_banned_patterns(
            ["def f():", "    return 1", "pass"], py_file
        )
        assert any("Empty pass statement" in m for _, m, _ in issues)

    def test_self_skip_for_quality_check_files(self, tmp_path: Path) -> None:
        f = tmp_path / "code_quality_check.py"
        assert cqc.check_banned_patterns(["# TO" + "DO bad"], f) == []

    def test_requires_list(self, py_file: Path) -> None:
        with pytest.raises(Exception):  # noqa: BLE001, PT011, B017
            cqc.check_banned_patterns("not a list", py_file)  # type: ignore[arg-type]

    def test_none_lines_raises(self, py_file: Path) -> None:
        with pytest.raises(ValueError):
            cqc.check_banned_patterns(None, py_file)  # type: ignore[arg-type]


class TestCheckMagicNumbers:
    def test_detects_pi(self, py_file: Path) -> None:
        issues = cqc.check_magic_numbers(["x = 3.141"], py_file)
        assert any("math.pi" in m for _, m, _ in issues)

    def test_detects_gravity(self, py_file: Path) -> None:
        issues = cqc.check_magic_numbers(["g = 9.81"], py_file)
        assert any("GRAVITY" in m for _, m, _ in issues)

    def test_detects_gravitational(self, py_file: Path) -> None:
        issues = cqc.check_magic_numbers(["G = 6.67"], py_file)
        assert any("gravitational" in m for _, m, _ in issues)

    def test_ignores_comment_only(self, py_file: Path) -> None:
        # Magic number inside the comment portion should be stripped
        issues = cqc.check_magic_numbers(["x = 1  # 3.141 is pi"], py_file)
        assert issues == []

    def test_self_skip(self, tmp_path: Path) -> None:
        f = tmp_path / "quality_check.py"
        assert cqc.check_magic_numbers(["x = 3.141"], f) == []

    def test_none_lines_raises(self, py_file: Path) -> None:
        with pytest.raises(ValueError):
            cqc.check_magic_numbers(None, py_file)  # type: ignore[arg-type]


class TestCheckAstIssues:
    def test_missing_docstring(self, py_file: Path) -> None:
        issues = cqc.check_ast_issues("def foo():\n    return 1\n", py_file)
        assert any("missing docstring" in m for _, m, _ in issues)

    def test_has_docstring_no_issue(self, py_file: Path) -> None:
        src = 'def foo() -> int:\n    """doc."""\n    return 1\n'
        assert cqc.check_ast_issues(src, py_file) == []

    def test_syntax_error_reported(self, py_file: Path) -> None:
        issues = cqc.check_ast_issues("def broken(:\n", py_file)
        assert any("Syntax error" in m for _, m, _ in issues)

    def test_self_skip(self, tmp_path: Path) -> None:
        f = tmp_path / "code_quality_check.py"
        assert cqc.check_ast_issues("def f(): return 1", f) == []

    def test_none_content_raises(self, py_file: Path) -> None:
        with pytest.raises(ValueError):
            cqc.check_ast_issues(None, py_file)  # type: ignore[arg-type]


class TestCheckFile:
    def test_clean_file_returns_empty(self, py_file: Path) -> None:
        py_file.write_text(
            'def foo() -> int:\n    """doc."""\n    return 1\n',
            encoding="utf-8",
        )
        assert cqc.check_file(py_file) == []

    def test_dirty_file_reports(self, py_file: Path) -> None:
        py_file.write_text("# TO" + "DO fix\ndef f():\n    pass\n", encoding="utf-8")
        issues = cqc.check_file(py_file)
        assert issues

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(Exception):  # noqa: BLE001, PT011, B017
            cqc.check_file(tmp_path / "missing.py")

    def test_read_error_returns_message(self, py_file: Path) -> None:
        py_file.write_text("x = 1", encoding="utf-8")
        with patch.object(Path, "read_text", side_effect=OSError("boom")):
            issues = cqc.check_file(py_file)
        assert issues and "Error reading file" in issues[0][1]


class TestMain:
    def test_main_clean_exits_zero(self, tmp_path: Path, monkeypatch) -> None:
        clean = tmp_path / "good.py"
        clean.write_text(
            'def foo() -> int:\n    """doc."""\n    return 1\n', encoding="utf-8"
        )
        monkeypatch.setattr(sys, "argv", ["cqc", str(clean)])
        with pytest.raises(SystemExit) as ei:
            cqc.main()
        assert ei.value.code == 0

    def test_main_dirty_exits_one(self, tmp_path: Path, monkeypatch, capsys) -> None:
        bad = tmp_path / "bad.py"
        bad.write_text("# TO" + "DO fix\ndef f():\n    return 1\n", encoding="utf-8")
        monkeypatch.setattr(sys, "argv", ["cqc", str(bad)])
        with pytest.raises(SystemExit) as ei:
            cqc.main()
        assert ei.value.code == 1
        captured = capsys.readouterr()
        assert "Quality check FAILED" in captured.err

    def test_main_directory_scan(self, tmp_path: Path, monkeypatch) -> None:
        # No argv args -> rglob from cwd; exclude_dirs filter applied
        (tmp_path / "good.py").write_text(
            'def foo() -> int:\n    """d."""\n    return 1\n', encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["cqc"])
        with pytest.raises(SystemExit) as ei:
            cqc.main()
        assert ei.value.code == 0

    def test_main_reports_syntax_error_line_zero(
        self, tmp_path: Path, monkeypatch, capsys
    ) -> None:
        bad = tmp_path / "syntax.py"
        bad.write_text("def broken(:\n", encoding="utf-8")
        monkeypatch.setattr(sys, "argv", ["cqc", str(bad)])
        with pytest.raises(SystemExit) as ei:
            cqc.main()
        assert ei.value.code == 1
        err = capsys.readouterr().err
        assert "Syntax error" in err

    def test_main_excludes_archive_dir(self, tmp_path: Path, monkeypatch) -> None:
        archive = tmp_path / "archive"
        archive.mkdir()
        (archive / "bad.py").write_text("# TO" + "DO bad\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["cqc"])
        with pytest.raises(SystemExit) as ei:
            cqc.main()
        assert ei.value.code == 0
