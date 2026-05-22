from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


def test_help_lists_subcommands_and_examples(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = importlib.import_module("sidekick.__main__")

    with pytest.raises(SystemExit) as exc_info:
        cli.parse_cli_args(["--help"])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "gui" in captured.out
    assert "run" in captured.out
    assert "python -m sidekick run" in captured.out


def test_default_gui_command_uses_chat_first_and_resolves_data_dir(
    tmp_path: Path,
) -> None:
    cli = importlib.import_module("sidekick.__main__")

    args = cli.parse_cli_args(["--profile", "calc-first", "--data-dir", str(tmp_path)])

    assert args.command == "gui"
    assert args.profile == "calc-first"
    assert args.data_dir == tmp_path.resolve()

    default_args = cli.parse_cli_args([])

    assert default_args.command == "gui"
    assert default_args.profile == "chat-first"


def test_invalid_gui_flag_suggests_closest_match(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = importlib.import_module("sidekick.__main__")

    with pytest.raises(SystemExit) as exc_info:
        cli.parse_cli_args(["--theem", "solarized"])

    assert exc_info.value.code == 2
    assert "--theme" in capsys.readouterr().err


def test_run_subcommand_stays_headless_during_parse(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs.json"
    inputs.write_text("{}", encoding="utf-8")
    pyqt_before = {name for name in sys.modules if name.startswith("PyQt6")}
    sys.modules.pop("sidekick.__main__", None)

    cli = importlib.import_module("sidekick.__main__")
    args = cli.parse_cli_args(
        ["run", "--calculator", "unit-converter", "--inputs", str(inputs)]
    )

    pyqt_after = {name for name in sys.modules if name.startswith("PyQt6")}
    assert args.command == "run"
    assert args.inputs == inputs.resolve()
    assert pyqt_after == pyqt_before
