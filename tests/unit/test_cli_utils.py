"""Tests for cli_utils module.

Tests command-line utility functions for argument parsing
and validation across the codebase.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from src.shared.python.cli_utils import (
    add_config_args,
    add_dry_run_arg,
    add_force_arg,
    add_logging_args,
    add_output_args,
    add_parallel_args,
    add_simulation_args,
    create_base_parser,
    get_effective_log_level,
    path_type,
    resolve_output_path,
    validate_input_files,
)


class TestCreateBaseParser:
    """Tests for create_base_parser function."""

    def test_creates_parser_with_description(self) -> None:
        """Parser should have the provided description."""
        parser = create_base_parser("Test program description")
        assert True
        # ArgumentParser stores description, checking it created successfully
        assert isinstance(parser, argparse.ArgumentParser)

    def test_creates_parser_with_prog(self) -> None:
        """Parser should use provided program name."""
        parser = create_base_parser("Desc", prog="myprogram")
        assert parser.prog == "myprogram"

    def test_default_formatter_class(self) -> None:
        """Parser should use RawDescriptionHelpFormatter by default."""
        parser = create_base_parser("Test")
        assert parser.formatter_class == argparse.RawDescriptionHelpFormatter


class TestAddLoggingArgs:
    """Tests for add_logging_args function."""

    def test_adds_verbose_flag(self) -> None:
        """Should add -v/--verbose flag."""
        parser = create_base_parser("Test")
        add_logging_args(parser)
        args = parser.parse_args(["-v"])
        assert args.verbose is True

    def test_adds_quiet_flag(self) -> None:
        """Should add -q/--quiet flag."""
        parser = create_base_parser("Test")
        add_logging_args(parser)
        args = parser.parse_args(["--quiet"])
        assert args.quiet is True

    def test_adds_log_level_option(self) -> None:
        """Should add --log-level option."""
        parser = create_base_parser("Test")
        add_logging_args(parser)
        args = parser.parse_args(["--log-level", "DEBUG"])
        assert args.log_level == "DEBUG"

    def test_default_log_level(self) -> None:
        """Should use INFO as default log level."""
        parser = create_base_parser("Test")
        add_logging_args(parser, default_level="WARNING")
        args = parser.parse_args([])
        assert args.log_level == "WARNING"


class TestAddOutputArgs:
    """Tests for add_output_args function."""

    def test_adds_output_option(self) -> None:
        """Should add -o/--output option."""
        parser = create_base_parser("Test")
        add_output_args(parser)
        args = parser.parse_args(["-o", "/tmp/output.txt"])
        assert args.output == Path("/tmp/output.txt")

    def test_adds_overwrite_flag(self) -> None:
        """Should add --overwrite flag."""
        parser = create_base_parser("Test")
        add_output_args(parser)
        args = parser.parse_args(["--overwrite"])
        assert args.overwrite is True

    def test_default_output(self) -> None:
        """Should use provided default output."""
        parser = create_base_parser("Test")
        add_output_args(parser, default_output="/default/path")
        args = parser.parse_args([])
        assert args.output == Path("/default/path")


class TestAddConfigArgs:
    """Tests for add_config_args function."""

    def test_adds_config_option(self) -> None:
        """Should add -c/--config option."""
        parser = create_base_parser("Test")
        add_config_args(parser)
        args = parser.parse_args(["-c", "config.yaml"])
        assert args.config == Path("config.yaml")

    def test_adds_no_config_flag(self) -> None:
        """Should add --no-config flag."""
        parser = create_base_parser("Test")
        add_config_args(parser)
        args = parser.parse_args(["--no-config"])
        assert args.no_config is True


class TestAddSimulationArgs:
    """Tests for add_simulation_args function."""

    def test_adds_time_step_option(self) -> None:
        """Should add --time-step option."""
        parser = create_base_parser("Test")
        add_simulation_args(parser)
        args = parser.parse_args(["--time-step", "0.002"])
        assert args.time_step == 0.002

    def test_adds_duration_option(self) -> None:
        """Should add --duration option."""
        parser = create_base_parser("Test")
        add_simulation_args(parser)
        args = parser.parse_args(["--duration", "10.0"])
        assert args.duration == 10.0

    def test_adds_engine_option(self) -> None:
        """Should add --engine option."""
        parser = create_base_parser("Test")
        add_simulation_args(parser)
        args = parser.parse_args(["--engine", "mujoco"])
        assert args.engine == "mujoco"


class TestAddDryRunArg:
    """Tests for add_dry_run_arg function."""

    def test_adds_dry_run_flag(self) -> None:
        """Should add --dry-run flag."""
        parser = create_base_parser("Test")
        add_dry_run_arg(parser)
        args = parser.parse_args(["--dry-run"])
        assert args.dry_run is True

    def test_dry_run_default_false(self) -> None:
        """Dry run should default to False."""
        parser = create_base_parser("Test")
        add_dry_run_arg(parser)
        args = parser.parse_args([])
        assert args.dry_run is False


class TestAddForceArg:
    """Tests for add_force_arg function."""

    def test_adds_force_flag(self) -> None:
        """Should add --force flag."""
        parser = create_base_parser("Test")
        add_force_arg(parser)
        args = parser.parse_args(["--force"])
        assert args.force is True


class TestAddParallelArgs:
    """Tests for add_parallel_args function."""

    def test_adds_jobs_option(self) -> None:
        """Should add -j/--jobs option."""
        parser = create_base_parser("Test")
        add_parallel_args(parser)
        args = parser.parse_args(["-j", "4"])
        assert args.jobs == 4

    def test_adds_sequential_flag(self) -> None:
        """Should add --sequential flag."""
        parser = create_base_parser("Test")
        add_parallel_args(parser)
        args = parser.parse_args(["--sequential"])
        assert args.sequential is True


class TestGetEffectiveLogLevel:
    """Tests for get_effective_log_level function."""

    def test_verbose_returns_debug(self) -> None:
        """Verbose flag should return DEBUG level."""
        args = argparse.Namespace(verbose=True, quiet=False, log_level="INFO")
        assert get_effective_log_level(args) == "DEBUG"

    def test_quiet_returns_warning(self) -> None:
        """Quiet flag should return WARNING level."""
        args = argparse.Namespace(verbose=False, quiet=True, log_level="INFO")
        assert get_effective_log_level(args) == "WARNING"

    def test_explicit_level_takes_precedence(self) -> None:
        """Explicit log level should be used when no flags set."""
        args = argparse.Namespace(verbose=False, quiet=False, log_level="ERROR")
        assert get_effective_log_level(args) == "ERROR"


class TestPathType:
    """Tests for path_type function."""

    def test_returns_path(self) -> None:
        """Should return Path object."""
        validator = path_type()
        result = validator("/some/path")
        assert isinstance(result, Path)
        assert result == Path("/some/path")

    def test_must_exist_raises_for_missing(self, tmp_path: Path) -> None:
        """Should raise for non-existent path when must_exist=True."""
        validator = path_type(must_exist=True)
        with pytest.raises(argparse.ArgumentTypeError):
            validator(str(tmp_path / "nonexistent"))

    def test_must_be_file_raises_for_directory(self, tmp_path: Path) -> None:
        """Should raise for directory when must_be_file=True."""
        validator = path_type(must_exist=True, must_be_file=True)
        with pytest.raises(argparse.ArgumentTypeError):
            validator(str(tmp_path))  # tmp_path is a directory

    def test_must_be_dir_raises_for_file(self, tmp_path: Path) -> None:
        """Should raise for file when must_be_dir=True."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        validator = path_type(must_exist=True, must_be_dir=True)
        with pytest.raises(argparse.ArgumentTypeError):
            validator(str(test_file))


class TestValidateInputFiles:
    """Tests for validate_input_files function."""

    def test_returns_path_list(self, tmp_path: Path) -> None:
        """Should return list of Path objects."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        result = validate_input_files([str(test_file)])
        assert len(result) == 1
        assert result[0] == test_file

    def test_raises_for_missing_file(self, tmp_path: Path) -> None:
        """Should raise for non-existent files when must_exist=True."""
        with pytest.raises(argparse.ArgumentTypeError):
            validate_input_files([str(tmp_path / "missing.txt")], must_exist=True)

    def test_validates_extensions(self, tmp_path: Path) -> None:
        """Should raise for invalid extensions."""
        test_file = tmp_path / "test.csv"
        test_file.write_text("data")
        with pytest.raises(argparse.ArgumentTypeError):
            validate_input_files([str(test_file)], extensions=[".json", ".yaml"])


class TestResolveOutputPath:
    """Tests for resolve_output_path function."""

    def test_returns_path_from_output_arg(self, tmp_path: Path) -> None:
        """Should return the output path from args."""
        args = argparse.Namespace(output=tmp_path / "result.json")
        result = resolve_output_path(args)
        assert result == tmp_path / "result.json"

    def test_adds_extension_if_missing(self, tmp_path: Path) -> None:
        """Should add extension if not present."""
        args = argparse.Namespace(output=tmp_path / "result")
        result = resolve_output_path(args, extension=".json")
        assert str(result).endswith(".json")

    def test_uses_default_name_for_directory(self, tmp_path: Path) -> None:
        """Should use default name when output is a directory."""
        args = argparse.Namespace(output=tmp_path)
        result = resolve_output_path(args, default_name="report", extension=".txt")
        assert result.name == "report.txt"
