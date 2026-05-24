from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest


def _load_cli():
    sys.modules.pop("sidekick.__main__", None)
    return importlib.import_module("sidekick.__main__")


def test_help_lists_subcommands_and_examples(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()

    with pytest.raises(SystemExit) as exc_info:
        cli.parse_cli_args(["--help"])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "gui" in captured.out
    assert "run" in captured.out
    assert "python -m sidekick run --calculator" in captured.out


def test_default_gui_command_uses_chat_first_and_resolves_data_dir(
    tmp_path: Path,
) -> None:
    cli = _load_cli()

    args = cli.parse_cli_args(
        [
            "--profile",
            "calc-first",
            "--theme",
            "solarized",
            "--data-dir",
            str(tmp_path),
            "--skip-onboarding",
        ]
    )

    assert args.command == "gui"
    assert args.profile == "calc-first"
    assert args.theme == "solarized"
    assert args.data_dir == tmp_path.resolve()
    assert args.skip_onboarding is True

    default_args = cli.parse_cli_args([])

    assert default_args.command == "gui"
    assert default_args.profile == "chat-first"


def test_invalid_gui_flag_suggests_closest_match(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()

    with pytest.raises(SystemExit) as exc_info:
        cli.parse_cli_args(["--theem", "solarized"])

    assert exc_info.value.code == 2
    assert "--theme" in capsys.readouterr().err


def test_run_subcommand_stays_headless_during_parse(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs.json"
    inputs.write_text("{}", encoding="utf-8")
    pyqt_before = {name for name in sys.modules if name.startswith("PyQt6")}

    cli = _load_cli()
    args = cli.parse_cli_args(
        ["run", "--calculator", "unit_converter", "--inputs", str(inputs)]
    )

    pyqt_after = {name for name in sys.modules if name.startswith("PyQt6")}
    assert args.command == "run"
    assert args.inputs == inputs.resolve()
    assert pyqt_after == pyqt_before


def test_invalid_profile_exits_code_2() -> None:
    cli = _load_cli()
    with pytest.raises(SystemExit) as exc_info:
        cli.parse_cli_args(["gui", "--profile", "bad-profile"])
    assert exc_info.value.code == 2


def test_unknown_argument_exits_code_2() -> None:
    cli = _load_cli()
    with pytest.raises(SystemExit) as exc_info:
        cli.parse_cli_args(["--does-not-exist"])
    assert exc_info.value.code == 2


def test_run_subcommand_rejects_gui_only_flags() -> None:
    cli = _load_cli()
    with pytest.raises(SystemExit) as exc_info:
        cli.parse_cli_args(["run", "--profile", "chat-first"])
    assert exc_info.value.code == 2


def test_run_subcommand_required_args_missing() -> None:
    cli = _load_cli()
    with pytest.raises(SystemExit) as exc_info:
        cli.parse_cli_args(["run"])
    assert exc_info.value.code == 2


def test_run_subcommand_invalid_format_exits_2() -> None:
    cli = _load_cli()
    with pytest.raises(SystemExit) as exc_info:
        cli.parse_cli_args(
            [
                "run",
                "--calculator",
                "x",
                "--inputs",
                str(Path(__file__)),
                "--format",
                "xml",
            ]
        )
    assert exc_info.value.code == 2


def test_run_subcommand_invalid_calculator_id_exits_2() -> None:
    cli = _load_cli()
    with pytest.raises(SystemExit) as exc_info:
        cli.parse_cli_args(
            [
                "run",
                "--calculator",
                "unit-converter",
                "--inputs",
                str(Path(__file__)),
            ]
        )
    assert exc_info.value.code == 2


def test_run_subcommand_valid_format_json() -> None:
    cli = _load_cli()
    ns = cli.parse_cli_args(
        [
            "run",
            "--calculator",
            "wgs_reactor",
            "--inputs",
            str(Path(__file__)),
        ]
    )
    assert ns.format == "json"


def test_run_subcommand_format_csv() -> None:
    cli = _load_cli()
    ns = cli.parse_cli_args(
        [
            "run",
            "--calculator",
            "wgs_reactor",
            "--inputs",
            str(Path(__file__)),
            "--format",
            "csv",
        ]
    )
    assert ns.format == "csv"


def test_run_headless_invalid_output_dir_returns_1(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()
    inputs = tmp_path / "inputs.json"
    inputs.write_text("{}", encoding="utf-8")

    args = cli.parse_cli_args(
        [
            "run",
            "--calculator",
            "wgs_reactor",
            "--inputs",
            str(inputs),
            "--output",
            str(tmp_path / "missing" / "result.json"),
        ]
    )

    assert cli.run_headless(args) == 1
    payload = json.loads(capsys.readouterr().err)
    assert payload["error"] == "sidekick_run_failed"


def test_launch_gui_delegates_to_launcher_factory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    calls: dict[str, object] = {}

    fake_launcher = types.ModuleType("sidekick.launcher_factory")

    def fake_create_launcher_config(
        app_module: str,
        window_title: str,
        min_width: int,
        min_height: int,
        **extra: object,
    ) -> str:
        calls["launcher_config"] = {
            "app_module": app_module,
            "window_title": window_title,
            "min_width": min_width,
            "min_height": min_height,
            "extra": extra,
        }
        return "LAUNCHER_CONFIG"

    def fake_launch_app(config: object, window_factory) -> int:
        calls["launch_app_config"] = config
        calls["window"] = window_factory()
        return 17

    fake_launcher.create_launcher_config = fake_create_launcher_config
    fake_launcher.launch_app = fake_launch_app

    fake_window_module = types.ModuleType("sidekick.standalone.window")

    class FakeStandaloneSidekickConfig:
        def __init__(
            self,
            profile: str,
            theme_name: str | None,
            session_store: object,
            host_action_port: object | None = None,
        ) -> None:
            self.profile = profile
            self.theme_name = theme_name
            self.session_store = session_store
            self.host_action_port = host_action_port

    class FakeStandaloneSidekickWindow:
        def __init__(self, config: FakeStandaloneSidekickConfig) -> None:
            self.config = config

    fake_window_module.StandaloneSidekickConfig = FakeStandaloneSidekickConfig
    fake_window_module.StandaloneSidekickWindow = FakeStandaloneSidekickWindow

    fake_store_module = types.ModuleType("sidekick.standalone.session_store")

    class FakeStandaloneSessionStore:
        def __init__(self, root: Path) -> None:
            self.root = root

    fake_store_module.StandaloneSessionStore = FakeStandaloneSessionStore

    monkeypatch.setitem(sys.modules, "sidekick.launcher_factory", fake_launcher)
    monkeypatch.setitem(sys.modules, "sidekick.standalone.window", fake_window_module)
    monkeypatch.setitem(
        sys.modules,
        "sidekick.standalone.session_store",
        fake_store_module,
    )

    args = cli.parse_cli_args(
        [
            "gui",
            "--profile",
            "calc-first",
            "--theme",
            "solarized",
            "--data-dir",
            str(tmp_path),
            "--skip-onboarding",
        ]
    )

    assert cli.launch_gui(args) == 17
    assert calls["launch_app_config"] == "LAUNCHER_CONFIG"
    assert calls["launcher_config"] == {
        "app_module": "sidekick.standalone.window",
        "window_title": "Sidekick",
        "min_width": 1280,
        "min_height": 800,
        "extra": {
            "profile": "calc-first",
            "theme_name": "solarized",
            "data_dir": str(tmp_path.resolve()),
            "skip_onboarding": True,
        },
    }
    window = calls["window"]
    assert isinstance(window, FakeStandaloneSidekickWindow)
    assert window.config.profile == "calc-first"
    assert window.config.theme_name == "solarized"
    assert window.config.session_store.root == tmp_path.resolve()
