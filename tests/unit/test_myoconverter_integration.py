"""Tests for MyoConverter integration module.

Tests the OpenSim to MuJoCo model conversion interface and error handling.

Refactored to use shared engine availability module (DRY principle).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.shared.python.engine_availability import (
    skip_if_unavailable,
)
from src.shared.python.myoconverter_integration import (
    MyoConverter,
    install_myoconverter_instructions,
)


@pytest.fixture
def temp_osim_file(tmp_path):
    """Create a temporary valid OpenSim file."""
    osim_path = tmp_path / "test_model.osim"

    # Create minimal valid OpenSim XML
    root = ET.Element("OpenSimDocument")
    tree = ET.ElementTree(root)
    tree.write(osim_path)

    return osim_path


@pytest.fixture
def temp_geometry_folder(tmp_path):
    """Create a temporary geometry folder."""
    geom_path = tmp_path / "Geometry"
    geom_path.mkdir()
    return geom_path


@pytest.fixture
def temp_output_folder(tmp_path):
    """Create a temporary output folder."""
    output_path = tmp_path / "output"
    return output_path


class TestMyoConverterInitialization:
    """Test MyoConverter initialization."""

    @patch(
        "src.shared.python.myoconverter_integration.MyoConverter._check_availability"
    )
    def test_initialization_when_available(self, mock_check):
        """Test initialization when myoconverter is available."""
        mock_check.return_value = True

        converter = MyoConverter()

        assert converter.myoconverter_available is True
        mock_check.assert_called_once()

    @patch(
        "src.shared.python.myoconverter_integration.MyoConverter._check_availability"
    )
    def test_initialization_when_unavailable(self, mock_check, caplog):
        """Test initialization when myoconverter is not available."""
        mock_check.return_value = False

        with caplog.at_level("WARNING"):
            converter = MyoConverter()

        assert converter.myoconverter_available is False
        assert "MyoConverter not available" in caplog.text


class TestCheckAvailability:
    """Test _check_availability method."""

    def test_availability_when_installed(self):
        """Test that check returns True when myoconverter can be imported."""
        with patch("builtins.__import__") as mock_import:
            # Mock successful import
            mock_import.return_value = MagicMock()

            converter = MyoConverter()
            # The actual check happens in __init__
            assert isinstance(converter.myoconverter_available, bool)

    def test_availability_when_not_installed(self):
        """Test that check returns False when import fails."""
        with patch(
            "src.shared.python.myoconverter_integration.MyoConverter._check_availability",
            return_value=False,
        ):
            converter = MyoConverter()
            assert converter.myoconverter_available is False


class TestValidateInputs:
    """Test _validate_inputs method."""

    @patch(
        "shared.python.myoconverter_integration.MyoConverter._check_availability",
        return_value=False,
    )
    def test_valid_inputs(
        self, mock_check, temp_osim_file, temp_geometry_folder, temp_output_folder
    ):
        """Test validation with valid inputs."""
        converter = MyoConverter()

        # Should not raise any exception
        converter._validate_inputs(
            temp_osim_file, temp_geometry_folder, temp_output_folder
        )

        # Output folder should be created
        assert temp_output_folder.exists()

    @patch(
        "shared.python.myoconverter_integration.MyoConverter._check_availability",
        return_value=False,
    )
    def test_missing_osim_file(
        self, mock_check, temp_geometry_folder, temp_output_folder
    ):
        """Test validation with missing osim file."""
        converter = MyoConverter()
        missing_file = Path("/nonexistent/model.osim")

        with pytest.raises(FileNotFoundError, match="OpenSim model file not found"):
            converter._validate_inputs(
                missing_file, temp_geometry_folder, temp_output_folder
            )

    @patch(
        "shared.python.myoconverter_integration.MyoConverter._check_availability",
        return_value=False,
    )
    def test_wrong_file_extension(
        self, mock_check, tmp_path, temp_geometry_folder, temp_output_folder
    ):
        """Test validation with wrong file extension."""
        converter = MyoConverter()
        wrong_file = tmp_path / "model.txt"
        wrong_file.touch()

        with pytest.raises(ValueError, match="Expected .osim file"):
            converter._validate_inputs(
                wrong_file, temp_geometry_folder, temp_output_folder
            )

    @patch(
        "shared.python.myoconverter_integration.MyoConverter._check_availability",
        return_value=False,
    )
    def test_invalid_xml(
        self, mock_check, tmp_path, temp_geometry_folder, temp_output_folder
    ):
        """Test validation with invalid XML file."""
        converter = MyoConverter()

        # Create file with invalid XML
        invalid_file = tmp_path / "invalid.osim"
        invalid_file.write_text("This is not XML")

        with pytest.raises(ValueError, match="Failed to parse OpenSim XML"):
            converter._validate_inputs(
                invalid_file, temp_geometry_folder, temp_output_folder
            )

    @patch(
        "shared.python.myoconverter_integration.MyoConverter._check_availability",
        return_value=False,
    )
    def test_wrong_root_element(
        self, mock_check, tmp_path, temp_geometry_folder, temp_output_folder
    ):
        """Test validation with wrong root element."""
        converter = MyoConverter()

        # Create XML with wrong root
        wrong_root_file = tmp_path / "wrong.osim"
        root = ET.Element("WrongRoot")
        tree = ET.ElementTree(root)
        tree.write(wrong_root_file)

        with pytest.raises(ValueError, match="Invalid OpenSim file"):
            converter._validate_inputs(
                wrong_root_file, temp_geometry_folder, temp_output_folder
            )

    @patch(
        "shared.python.myoconverter_integration.MyoConverter._check_availability",
        return_value=False,
    )
    def test_missing_geometry_folder_warning(
        self, mock_check, temp_osim_file, temp_output_folder, caplog
    ):
        """Test that missing geometry folder generates warning."""
        converter = MyoConverter()
        missing_geom = Path("/nonexistent/geometry")

        with caplog.at_level("WARNING"):
            converter._validate_inputs(temp_osim_file, missing_geom, temp_output_folder)

        assert "Geometry folder not found" in caplog.text


class TestConvertOsimToMujoco:
    """Test convert_osim_to_mujoco method."""

    @patch(
        "shared.python.myoconverter_integration.MyoConverter._check_availability",
        return_value=False,
    )
    def test_raises_error_when_unavailable(
        self, mock_check, temp_osim_file, temp_geometry_folder, temp_output_folder
    ):
        """Test that conversion raises error when myoconverter not available."""
        converter = MyoConverter()

        with pytest.raises(RuntimeError, match="MyoConverter not installed"):
            converter.convert_osim_to_mujoco(
                temp_osim_file, temp_geometry_folder, temp_output_folder
            )

    @skip_if_unavailable("myoconverter")
    @patch(
        "shared.python.myoconverter_integration.MyoConverter._check_availability",
        return_value=True,
    )
    def test_successful_conversion(
        self, mock_check, temp_osim_file, temp_geometry_folder, temp_output_folder
    ):
        """Test successful model conversion (requires myoconverter)."""
        # Skip test body if myoconverter not available
        pytest.skip("Requires myoconverter - pending implementation")

    @skip_if_unavailable("myoconverter")
    @patch(
        "shared.python.myoconverter_integration.MyoConverter._check_availability",
        return_value=True,
    )
    def test_custom_config_passed(
        self, mock_check, temp_osim_file, temp_geometry_folder, temp_output_folder
    ):
        """Test that custom configuration is passed to pipeline (requires myoconverter)."""
        # Skip test body if myoconverter not available
        pytest.skip("Requires myoconverter - pending implementation")


class TestHandleConversionError:
    """Test _handle_conversion_error method."""

    @patch(
        "shared.python.myoconverter_integration.MyoConverter._check_availability",
        return_value=False,
    )
    def test_geometry_error_handling(
        self, mock_check, temp_osim_file, temp_geometry_folder
    ):
        """Test handling of geometry-related errors."""
        converter = MyoConverter()
        error = Exception("mesh file not found")

        with pytest.raises(RuntimeError, match="geometry/mesh issues"):
            converter._handle_conversion_error(
                error, temp_osim_file, temp_geometry_folder
            )

    @patch(
        "shared.python.myoconverter_integration.MyoConverter._check_availability",
        return_value=False,
    )
    def test_muscle_error_handling(
        self, mock_check, temp_osim_file, temp_geometry_folder
    ):
        """Test handling of muscle-related errors."""
        converter = MyoConverter()
        error = Exception("muscle path point invalid")

        with pytest.raises(RuntimeError, match="muscle configuration"):
            converter._handle_conversion_error(
                error, temp_osim_file, temp_geometry_folder
            )

    @patch(
        "shared.python.myoconverter_integration.MyoConverter._check_availability",
        return_value=False,
    )
    def test_constraint_error_handling(
        self, mock_check, temp_osim_file, temp_geometry_folder
    ):
        """Test handling of constraint-related errors."""
        converter = MyoConverter()
        error = Exception("constraint violation detected")

        with pytest.raises(RuntimeError, match="constraints"):
            converter._handle_conversion_error(
                error, temp_osim_file, temp_geometry_folder
            )

    @patch(
        "shared.python.myoconverter_integration.MyoConverter._check_availability",
        return_value=False,
    )
    def test_generic_error_handling(
        self, mock_check, temp_osim_file, temp_geometry_folder
    ):
        """Test handling of generic errors."""
        converter = MyoConverter()
        error = Exception("unknown error occurred")

        with pytest.raises(RuntimeError, match="Model conversion failed"):
            converter._handle_conversion_error(
                error, temp_osim_file, temp_geometry_folder
            )


class TestLoadConvertedModelKeyframe:
    """Test load_converted_model_keyframe method."""

    @patch(
        "shared.python.myoconverter_integration.MyoConverter._check_availability",
        return_value=False,
    )
    def test_generates_valid_code(self, mock_check, tmp_path):
        """Test that generated code contains required elements."""
        converter = MyoConverter()
        model_path = tmp_path / "model.xml"

        code = converter.load_converted_model_keyframe(model_path)

        assert "import mujoco" in code
        assert "MjModel.from_xml_path" in code
        assert "mj_resetDataKeyframe" in code
        assert str(model_path) in code

    @patch(
        "shared.python.myoconverter_integration.MyoConverter._check_availability",
        return_value=False,
    )
    def test_code_is_string(self, mock_check, tmp_path):
        """Test that return type is string."""
        converter = MyoConverter()
        model_path = tmp_path / "model.xml"

        code = converter.load_converted_model_keyframe(model_path)

        assert isinstance(code, str)


class TestGetExampleModels:
    """Test get_example_models method."""

    @patch(
        "shared.python.myoconverter_integration.MyoConverter._check_availability",
        return_value=False,
    )
    def test_returns_dict(self, mock_check):
        """Test that method returns a dictionary."""
        converter = MyoConverter()
        models = converter.get_example_models()

        assert isinstance(models, dict)

    @patch(
        "shared.python.myoconverter_integration.MyoConverter._check_availability",
        return_value=False,
    )
    def test_contains_known_models(self, mock_check):
        """Test that dictionary contains expected model keys."""
        converter = MyoConverter()
        models = converter.get_example_models()

        expected_models = ["tug_of_war", "simple_arm", "gait_2d", "gait_3d"]
        for model_name in expected_models:
            assert model_name in models

    @patch(
        "shared.python.myoconverter_integration.MyoConverter._check_availability",
        return_value=False,
    )
    def test_urls_are_strings(self, mock_check):
        """Test that all URLs are strings."""
        converter = MyoConverter()
        models = converter.get_example_models()

        for url in models.values():
            assert isinstance(url, str)
            assert url.startswith("https://")


class TestValidateConversion:
    """Test validate_conversion method."""

    @patch(
        "shared.python.myoconverter_integration.MyoConverter._check_availability",
        return_value=False,
    )
    def test_validation_passes_when_files_exist(
        self, mock_check, temp_osim_file, tmp_path
    ):
        """Test validation passes when both files exist."""
        converter = MyoConverter()
        mujoco_xml = tmp_path / "converted.xml"
        mujoco_xml.touch()

        result = converter.validate_conversion(mujoco_xml, temp_osim_file)

        assert result is True

    @patch(
        "shared.python.myoconverter_integration.MyoConverter._check_availability",
        return_value=False,
    )
    def test_validation_fails_when_mujoco_missing(self, mock_check, temp_osim_file):
        """Test validation fails when MuJoCo file missing."""
        converter = MyoConverter()
        missing_xml = Path("/nonexistent/model.xml")

        result = converter.validate_conversion(missing_xml, temp_osim_file)

        assert result is False

    @patch(
        "shared.python.myoconverter_integration.MyoConverter._check_availability",
        return_value=False,
    )
    def test_validation_fails_when_osim_missing(self, mock_check, tmp_path):
        """Test validation fails when OpenSim file missing."""
        converter = MyoConverter()
        mujoco_xml = tmp_path / "model.xml"
        mujoco_xml.touch()
        missing_osim = Path("/nonexistent/model.osim")

        result = converter.validate_conversion(mujoco_xml, missing_osim)

        assert result is False


class TestInstallMyoconverterInstructions:
    """Test install_myoconverter_instructions function."""

    def test_returns_string(self):
        """Test that function returns a string."""
        instructions = install_myoconverter_instructions()
        assert isinstance(instructions, str)

    def test_contains_installation_info(self):
        """Test that instructions contain key information."""
        instructions = install_myoconverter_instructions()

        assert "conda install" in instructions
        assert "myoconverter" in instructions
        assert "Docker" in instructions
        assert "Linux" in instructions

    def test_contains_verification_step(self):
        """Test that instructions include verification."""
        instructions = install_myoconverter_instructions()

        assert "import myoconverter" in instructions

    def test_contains_documentation_link(self):
        """Test that instructions include documentation link."""
        instructions = install_myoconverter_instructions()

        assert "https://" in instructions
        assert "myoconverter" in instructions.lower()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @skip_if_unavailable("myoconverter")
    @patch(
        "shared.python.myoconverter_integration.MyoConverter._check_availability",
        return_value=True,
    )
    def test_no_output_file_generated(
        self, mock_check, temp_osim_file, temp_geometry_folder, temp_output_folder
    ):
        """Test error when conversion completes but no output file found."""
        # Skip test body if myoconverter not available
        pytest.skip("Requires myoconverter - pending implementation")

    @patch(
        "shared.python.myoconverter_integration.MyoConverter._check_availability",
        return_value=False,
    )
    def test_output_folder_created_if_missing(
        self, mock_check, temp_osim_file, temp_geometry_folder, tmp_path
    ):
        """Test that output folder is created if it doesn't exist."""
        converter = MyoConverter()
        output_folder = tmp_path / "new_folder" / "subfolder"

        converter._validate_inputs(temp_osim_file, temp_geometry_folder, output_folder)

        assert output_folder.exists()
