"""Coverage for src/shared/python/cli_utils.py.

Covers parser builders, argparse type helpers, and main runner.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.shared.python import cli_utils

# ── Parser builders ─────────────────────────────────────────────────────


def test_create_base_parser() -> None:
    parser = cli_utils.create_base_parser("test program")
    assert isinstance(parser, argparse.ArgumentParser)
    assert parser.description == "test program"


def test_add_logging_args_no_file() -> None:
    parser = cli_utils.create_base_parser("x")
    cli_utils.add_logging_args(parser, include_file=False)
    args = parser.parse_args(["-v"])
    assert args.verbose is True
    assert args.quiet is False
    assert args.log_level == "INFO"


def test_add_logging_args_with_file() -> None:
    parser = cli_utils.create_base_parser("x")
    cli_utils.add_logging_args(parser, include_file=True)
    args = parser.parse_args(["--log-file", "/tmp/x.log", "--log-level", "DEBUG"])
    assert args.log_file == Path("/tmp/x.log")
    assert args.log_level == "DEBUG"


def test_add_output_args() -> None:
    parser = cli_utils.create_base_parser("x")
    cli_utils.add_output_args(parser, include_format=True)
    args = parser.parse_args(["-o", "out.json", "--overwrite", "--format", "yaml"])
    assert args.output == Path("out.json")
    assert args.overwrite is True
    assert args.format == "yaml"


def test_add_config_args() -> None:
    parser = cli_utils.create_base_parser("x")
    cli_utils.add_config_args(parser, default_config="default.yaml")
    args = parser.parse_args([])
    assert args.config == Path("default.yaml")
    args2 = parser.parse_args(["--no-config"])
    assert args2.no_config is True


def test_add_simulation_args() -> None:
    parser = cli_utils.create_base_parser("x")
    cli_utils.add_simulation_args(parser)
    args = parser.parse_args(
        ["--time-step", "0.01", "--duration", "5", "--engine", "mujoco"]
    )
    assert args.time_step == 0.01
    assert args.duration == 5
    assert args.engine == "mujoco"


def test_add_dry_run_and_force() -> None:
    parser = cli_utils.create_base_parser("x")
    cli_utils.add_dry_run_arg(parser)
    cli_utils.add_force_arg(parser)
    args = parser.parse_args(["-n", "-f"])
    assert args.dry_run is True
    assert args.force is True


def test_add_parallel_args() -> None:
    parser = cli_utils.create_base_parser("x")
    cli_utils.add_parallel_args(parser, default_workers=4)
    args = parser.parse_args(["--sequential"])
    assert args.sequential is True
    assert args.jobs == 4


# ── setup_from_args ─────────────────────────────────────────────────────


def test_setup_from_args_verbose(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_setup_logging(*, level, filename=None):  # type: ignore[no-untyped-def]
        captured["level"] = level
        captured["filename"] = filename

    monkeypatch.setattr(cli_utils, "setup_logging", fake_setup_logging)
    ns = argparse.Namespace(verbose=True, quiet=False, log_level="INFO", log_file=None)
    cli_utils.setup_from_args(ns)
    assert captured["level"] == cli_utils.LogLevel.DEBUG


def test_setup_from_args_quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        cli_utils,
        "setup_logging",
        lambda **kw: captured.update(kw),
    )
    ns = argparse.Namespace(verbose=False, quiet=True, log_level="INFO", log_file=None)
    cli_utils.setup_from_args(ns)
    assert captured["level"] == cli_utils.LogLevel.WARNING


def test_setup_from_args_explicit_level(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        cli_utils,
        "setup_logging",
        lambda **kw: captured.update(kw),
    )
    ns = argparse.Namespace(
        verbose=False, quiet=False, log_level="ERROR", log_file=None
    )
    cli_utils.setup_from_args(ns)
    assert captured["level"] == cli_utils.LogLevel.ERROR


# ── get_effective_log_level ─────────────────────────────────────────────


def test_get_effective_log_level_verbose() -> None:
    ns = argparse.Namespace(verbose=True)
    assert cli_utils.get_effective_log_level(ns) == "DEBUG"


def test_get_effective_log_level_quiet() -> None:
    ns = argparse.Namespace(verbose=False, quiet=True)
    assert cli_utils.get_effective_log_level(ns) == "WARNING"


def test_get_effective_log_level_explicit() -> None:
    ns = argparse.Namespace(verbose=False, quiet=False, log_level="ERROR")
    assert cli_utils.get_effective_log_level(ns) == "ERROR"


def test_get_effective_log_level_default() -> None:
    ns = argparse.Namespace()
    assert cli_utils.get_effective_log_level(ns) == "INFO"


# ── resolve_output_path ─────────────────────────────────────────────────


def test_resolve_output_path_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    ns = argparse.Namespace(output=None)
    result = cli_utils.resolve_output_path(ns, default_name="report", extension=".json")
    assert result.name == "report.json"


def test_resolve_output_path_directory(tmp_path: Path) -> None:
    target_dir = tmp_path / "results"
    target_dir.mkdir()
    ns = argparse.Namespace(output=target_dir)
    result = cli_utils.resolve_output_path(ns, default_name="data", extension=".csv")
    assert result == target_dir / "data.csv"


def test_resolve_output_path_with_suffix(tmp_path: Path) -> None:
    ns = argparse.Namespace(output=tmp_path / "result.json")
    result = cli_utils.resolve_output_path(ns, extension=".json")
    assert result.suffix == ".json"


def test_resolve_output_path_no_suffix_adds_extension(tmp_path: Path) -> None:
    ns = argparse.Namespace(output=tmp_path / "no_ext_path_that_does_not_exist")
    # treated as dir-like since no suffix and doesn't exist
    result = cli_utils.resolve_output_path(ns, default_name="x", extension=".txt")
    assert result.name.endswith(".txt")


# ── validate_input_files ────────────────────────────────────────────────


def test_validate_input_files_must_exist(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("hello")
    result = cli_utils.validate_input_files([f])
    assert result == [f]


def test_validate_input_files_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="does not exist"):
        cli_utils.validate_input_files([tmp_path / "missing.txt"])


def test_validate_input_files_extension_filter(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("hi")
    with pytest.raises(argparse.ArgumentTypeError, match="Invalid file type"):
        cli_utils.validate_input_files([f], extensions=[".csv"])


def test_validate_input_files_skip_existence(tmp_path: Path) -> None:
    paths = [tmp_path / "missing.txt"]
    result = cli_utils.validate_input_files(paths, must_exist=False)
    assert result == paths


# ── path_type ───────────────────────────────────────────────────────────


def test_path_type_exists(tmp_path: Path) -> None:
    fn = cli_utils.path_type(must_exist=True)
    assert fn(str(tmp_path)) == tmp_path  # type: ignore[operator]


def test_path_type_missing_raises(tmp_path: Path) -> None:
    fn = cli_utils.path_type(must_exist=True)
    with pytest.raises(argparse.ArgumentTypeError, match="does not exist"):
        fn(str(tmp_path / "nope"))  # type: ignore[operator]


def test_path_type_not_file(tmp_path: Path) -> None:
    fn = cli_utils.path_type(must_be_file=True)
    with pytest.raises(argparse.ArgumentTypeError, match="not a file"):
        fn(str(tmp_path))  # type: ignore[operator]


def test_path_type_not_dir(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("hi")
    fn = cli_utils.path_type(must_be_dir=True)
    with pytest.raises(argparse.ArgumentTypeError, match="not a directory"):
        fn(str(f))  # type: ignore[operator]


# ── numeric types ──────────────────────────────────────────────────────


def test_positive_int_ok() -> None:
    assert cli_utils.positive_int("5") == 5


def test_positive_int_zero_rejected() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="Must be positive"):
        cli_utils.positive_int("0")


def test_positive_int_invalid() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="Invalid integer"):
        cli_utils.positive_int("abc")


def test_non_negative_int_ok() -> None:
    assert cli_utils.non_negative_int("0") == 0
    assert cli_utils.non_negative_int("3") == 3


def test_non_negative_int_negative() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="non-negative"):
        cli_utils.non_negative_int("-1")


def test_non_negative_int_invalid() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="Invalid integer"):
        cli_utils.non_negative_int("x")


def test_positive_float_ok() -> None:
    assert cli_utils.positive_float("1.5") == 1.5


def test_positive_float_zero() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="Must be positive"):
        cli_utils.positive_float("0")


def test_positive_float_invalid() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="Invalid float"):
        cli_utils.positive_float("nope")


# ── run_main ───────────────────────────────────────────────────────────


def test_run_main_success(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = cli_utils.create_base_parser("x")
    cli_utils.add_logging_args(parser)
    monkeypatch.setattr(cli_utils, "setup_from_args", MagicMock())
    monkeypatch.setattr("sys.argv", ["prog"])
    rc = cli_utils.run_main(lambda _ns: 0, parser)
    assert rc == 0


def test_run_main_none_return(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = cli_utils.create_base_parser("x")
    cli_utils.add_logging_args(parser)
    monkeypatch.setattr(cli_utils, "setup_from_args", MagicMock())
    monkeypatch.setattr("sys.argv", ["prog"])
    rc = cli_utils.run_main(lambda _ns: None, parser)
    assert rc == 0


def test_run_main_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = cli_utils.create_base_parser("x")
    cli_utils.add_logging_args(parser)
    monkeypatch.setattr(cli_utils, "setup_from_args", MagicMock())
    monkeypatch.setattr("sys.argv", ["prog"])

    def boom(_ns: argparse.Namespace) -> int:
        raise KeyboardInterrupt

    rc = cli_utils.run_main(boom, parser)
    assert rc == 130


def test_run_main_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = cli_utils.create_base_parser("x")
    cli_utils.add_logging_args(parser)
    monkeypatch.setattr(cli_utils, "setup_from_args", MagicMock())
    monkeypatch.setattr("sys.argv", ["prog"])

    def boom(_ns: argparse.Namespace) -> int:
        raise RuntimeError("nope")

    rc = cli_utils.run_main(boom, parser)
    assert rc == 1


def test_run_main_no_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = cli_utils.create_base_parser("x")
    cli_utils.add_logging_args(parser)
    setup_mock = MagicMock()
    monkeypatch.setattr(cli_utils, "setup_from_args", setup_mock)
    monkeypatch.setattr("sys.argv", ["prog"])
    rc = cli_utils.run_main(lambda _ns: 7, parser, setup_logging_from_args=False)
    assert rc == 7
    setup_mock.assert_not_called()
