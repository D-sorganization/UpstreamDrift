"""
Tests for the CLI module.
"""

import tempfile
from pathlib import Path

import pytest

SIMPLE_URDF = """<?xml version="1.0"?>
<robot name="test_robot">
    <link name="base_link">
        <inertial>
            <mass value="1.0"/>
            <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
        </inertial>
    </link>
</robot>
"""


class TestCLIParser:
    """Tests for CLI argument parsing."""

    def test_create_parser(self):
        """Test parser creation."""
        from model_generation.cli import create_parser

        parser = create_parser()
        assert parser is not None

    def test_parse_generate_command(self):
        """Test parsing generate command."""
        from model_generation.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["generate", "my_robot", "--humanoid"])

        assert args.command == "generate"
        assert args.name == "my_robot"
        assert args.humanoid is True

    def test_parse_convert_command(self):
        """Test parsing convert command."""
        from model_generation.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(
            ["convert", "input.slx", "-o", "output.urdf", "-f", "simscape"]
        )

        assert args.command == "convert"
        assert args.input == "input.slx"
        assert args.output == "output.urdf"
        assert args.from_format == "simscape"

    def test_parse_validate_command(self):
        """Test parsing validate command."""
        from model_generation.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["validate", "robot.urdf", "--json"])

        assert args.command == "validate"
        assert args.input == "robot.urdf"
        assert args.json is True

    def test_parse_diff_command(self):
        """Test parsing diff command."""
        from model_generation.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["diff", "file1.urdf", "file2.urdf", "--side-by-side"])

        assert args.command == "diff"
        assert args.file_a == "file1.urdf"
        assert args.file_b == "file2.urdf"
        assert args.side_by_side is True

    def test_parse_info_command(self):
        """Test parsing info command."""
        from model_generation.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["info", "robot.urdf", "--json"])

        assert args.command == "info"
        assert args.input == "robot.urdf"
        assert args.json is True

    def test_parse_inertia_command(self):
        """Test parsing inertia command."""
        from model_generation.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(
            ["inertia", "box", "1.0", "0.1", "0.2", "0.3", "--json"]
        )

        assert args.command == "inertia"
        assert args.shape == "box"
        assert args.mass == 1.0
        assert args.dimensions == [0.1, 0.2, 0.3]
        assert args.json is True

    def test_parse_library_list(self):
        """Test parsing library list command."""
        from model_generation.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["library", "list", "-c", "humanoid", "--json"])

        assert args.command == "library"
        assert args.lib_command == "list"
        assert args.category == "humanoid"

    def test_parse_compose_command(self):
        """Test parsing compose command."""
        from model_generation.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(
            [
                "compose",
                "-s",
                "body:body.urdf",
                "arm:arm.urdf",
                "-o",
                "output.urdf",
                "--operations",
                "copy:body:torso",
                "paste:base_link",
            ]
        )

        assert args.command == "compose"
        assert len(args.sources) == 2
        assert args.output == "output.urdf"
        assert len(args.operations) == 2


class TestCLICommands:
    """Tests for CLI command execution."""

    def test_cmd_validate_valid_urdf(self):
        """Test validate command with valid URDF."""
        import argparse

        from model_generation.cli.main import cmd_validate

        with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
            f.write(SIMPLE_URDF)
            temp_path = Path(f.name)

        try:
            args = argparse.Namespace(
                input=str(temp_path),
                json=False,
                errors_only=False,
                show_info=False,
                verbose=False,
            )

            result = cmd_validate(args)
            assert result == 0  # Success
        finally:
            temp_path.unlink()

    def test_cmd_validate_invalid_urdf(self):
        """Test validate command with invalid URDF."""
        import argparse

        from model_generation.cli.main import cmd_validate

        invalid_urdf = "<robot><invalid></robot>"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
            f.write(invalid_urdf)
            temp_path = Path(f.name)

        try:
            args = argparse.Namespace(
                input=str(temp_path),
                json=False,
                errors_only=False,
                show_info=False,
                verbose=False,
            )

            result = cmd_validate(args)
            assert result == 1  # Error
        finally:
            temp_path.unlink()

    def test_cmd_info(self):
        """Test info command."""
        import argparse

        from model_generation.cli.main import cmd_info

        with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
            f.write(SIMPLE_URDF)
            temp_path = Path(f.name)

        try:
            args = argparse.Namespace(
                input=str(temp_path),
                json=False,
                verbose=False,
            )

            result = cmd_info(args)
            assert result == 0
        finally:
            temp_path.unlink()

    def test_cmd_inertia_box(self):
        """Test inertia calculation for box."""
        import argparse

        from model_generation.cli.main import cmd_inertia

        args = argparse.Namespace(
            shape="box",
            mass=1.0,
            dimensions=[0.1, 0.2, 0.3],
            json=False,
        )

        result = cmd_inertia(args)
        assert result == 0

    def test_cmd_inertia_cylinder(self):
        """Test inertia calculation for cylinder."""
        import argparse

        from model_generation.cli.main import cmd_inertia

        args = argparse.Namespace(
            shape="cylinder",
            mass=1.0,
            dimensions=[0.05, 0.2],
            json=False,
        )

        result = cmd_inertia(args)
        assert result == 0

    def test_cmd_inertia_sphere(self):
        """Test inertia calculation for sphere."""
        import argparse

        from model_generation.cli.main import cmd_inertia

        args = argparse.Namespace(
            shape="sphere",
            mass=1.0,
            dimensions=[0.1],
            json=False,
        )

        result = cmd_inertia(args)
        assert result == 0

    def test_cmd_diff(self):
        """Test diff command."""
        import argparse

        from model_generation.cli.main import cmd_diff

        urdf_v1 = SIMPLE_URDF
        urdf_v2 = SIMPLE_URDF.replace("test_robot", "modified_robot")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f1:
            f1.write(urdf_v1)
            path1 = Path(f1.name)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f2:
            f2.write(urdf_v2)
            path2 = Path(f2.name)

        try:
            args = argparse.Namespace(
                file_a=str(path1),
                file_b=str(path2),
                json=False,
                side_by_side=False,
                fail_on_diff=False,
                verbose=False,
            )

            result = cmd_diff(args)
            assert result == 0
        finally:
            path1.unlink()
            path2.unlink()


class TestCLIMain:
    """Tests for main entry point."""

    def test_main_no_args(self):
        """Test main with no arguments shows help."""
        from model_generation.cli import main

        # Should not crash and return 0
        result = main([])
        assert result == 0

    def test_main_help(self):
        """Test main with --help."""
        from model_generation.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_main_version(self):
        """Test main with --version."""
        from model_generation.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0
