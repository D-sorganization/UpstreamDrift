"""Tests for sidekick.__main__ CLI argparser — T1 (#5979).

Covers:
  - default subcommand (gui / chat-first)
  - --help lists both subcommands
  - malformed/unknown args exit with code 2
  - --profile restricted to valid choices
  - --data-dir resolved to absolute path
  - headless 'run' path does not import PyQt6
  - run required args missing → exit 2
  - run --format restricted to valid choices
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


def test_help_output_lists_both_subcommands() -> None:
    from sidekick.__main__ import build_parser

    parser = build_parser()
    help_text = parser.format_help()
    assert "gui" in help_text
    assert "run" in help_text


def test_default_subcommand_is_gui() -> None:
    from sidekick.__main__ import build_parser

    parser = build_parser()
    ns = parser.parse_args([])
    assert ns.subcommand == "gui"


def test_gui_default_profile_is_chat_first() -> None:
    from sidekick.__main__ import build_parser

    parser = build_parser()
    ns = parser.parse_args(["gui"])
    assert ns.profile == "chat-first"


def test_gui_calc_first_profile() -> None:
    from sidekick.__main__ import build_parser

    parser = build_parser()
    ns = parser.parse_args(["gui", "--profile", "calc-first"])
    assert ns.profile == "calc-first"


def test_invalid_profile_exits_code_2() -> None:
    from sidekick.__main__ import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["gui", "--profile", "bad-profile"])
    assert exc_info.value.code == 2


def test_unknown_argument_exits_code_2() -> None:
    from sidekick.__main__ import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--does-not-exist"])
    assert exc_info.value.code == 2


def test_data_dir_resolved_to_absolute(tmp_path: Path) -> None:
    from sidekick.__main__ import build_parser

    parser = build_parser()
    ns = parser.parse_args(["gui", "--data-dir", str(tmp_path)])
    assert Path(ns.data_dir).is_absolute()


def test_headless_run_path_no_pyqt6(monkeypatch: pytest.MonkeyPatch) -> None:
    """'sidekick run' parse must not trigger a PyQt6 import."""
    pyqt_keys = [k for k in sys.modules if "PyQt6" in k]
    backup = {k: sys.modules.pop(k) for k in pyqt_keys}
    try:
        from sidekick.__main__ import build_parser

        parser = build_parser()
        parser.parse_args(
            [
                "run",
                "--calculator",
                "test_calc",
                "--inputs",
                "/dev/null",
            ]
        )
        assert not any("PyQt6" in k for k in sys.modules), (
            "PyQt6 was imported on a headless 'run' parse path"
        )
    finally:
        sys.modules.update(backup)


def test_run_subcommand_required_args_missing() -> None:
    from sidekick.__main__ import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["run"])
    assert exc_info.value.code == 2


def test_run_subcommand_invalid_format_exits_2() -> None:
    from sidekick.__main__ import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(
            [
                "run",
                "--calculator",
                "x",
                "--inputs",
                "/dev/null",
                "--format",
                "xml",
            ]
        )
    assert exc_info.value.code == 2


def test_run_subcommand_valid_format_json() -> None:
    from sidekick.__main__ import build_parser

    parser = build_parser()
    ns = parser.parse_args(
        [
            "run",
            "--calculator",
            "wgs_reactor",
            "--inputs",
            "/dev/null",
        ]
    )
    assert ns.format == "json"


def test_run_subcommand_format_csv() -> None:
    from sidekick.__main__ import build_parser

    parser = build_parser()
    ns = parser.parse_args(
        [
            "run",
            "--calculator",
            "wgs_reactor",
            "--inputs",
            "/dev/null",
            "--format",
            "csv",
        ]
    )
    assert ns.format == "csv"


def test_gui_theme_arg() -> None:
    from sidekick.__main__ import build_parser

    parser = build_parser()
    ns = parser.parse_args(["gui", "--theme", "catppuccin-mocha"])
    assert ns.theme == "catppuccin-mocha"
