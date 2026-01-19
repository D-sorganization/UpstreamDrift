"""Tests for PhysicalConstant safe usage in XML generation."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from shared.python.constants import (
    AIR_DENSITY_SEA_LEVEL_KG_M3,
    GOLF_BALL_MASS_KG,
    GRAVITY_M_S2,
    PhysicalConstant,
)


class TestPhysicalConstantXMLSafety:
    """Test that PhysicalConstants work safely in XML templates."""

    def test_physical_constant_in_f_string_with_float_conversion(self):
        """PhysicalConstants must be converted to float in XML f-strings."""
        # Correct usage: explicit float() conversion
        xml_string = f'<option gravity="0 0 -{float(GRAVITY_M_S2)}"/>'

        # Should parse without error
        root = ET.fromstring(f"<root>{xml_string}</root>")

        # Verify numeric value
        option_elem = root.find("option")
        assert option_elem is not None, "option element not found"
        gravity_attr = option_elem.get("gravity")
        assert gravity_attr is not None, "gravity attribute not found"
        assert (
            "PhysicalConstant" not in gravity_attr
        ), "PhysicalConstant.__repr__ leaked into XML"

        # Verify parseable as floats
        g_components = gravity_attr.split()
        assert len(g_components) == 3
        gx, gy, gz = map(float, g_components)
        assert gx == 0.0
        assert gy == 0.0
        assert 9.0 < abs(gz) < 10.0, f"Gravity value {gz} out of expected range"

    def test_physical_constant_without_float_fails(self):
        """PhysicalConstant __repr__ in XML should be detectable."""
        # Incorrect usage (what we want to prevent)
        xml_string = f'<option gravity="0 0 -{GRAVITY_M_S2}"/>'

        # This will include the __repr__ representation
        assert (
            "PhysicalConstant" in xml_string
        ), "Test assumption wrong: __repr__ should appear"

        # Parsing may succeed but value is garbage
        root = ET.fromstring(f"<root>{xml_string}</root>")
        option_elem = root.find("option")
        assert option_elem is not None, "option element not found"
        gravity_attr = option_elem.get("gravity")
        assert gravity_attr is not None, "gravity attribute not found"

        # The string will contain "PhysicalConstant(...)"
        assert "PhysicalConstant" in gravity_attr

    def test_all_gravity_usages_in_codebase_pattern(self):
        """Verify the float() pattern works for various use cases."""
        # MuJoCo XML pattern
        timestep = 0.001
        mujoco_xml = f"""
        <mujoco>
            <option timestep="{timestep}" gravity="0 0 -{float(GRAVITY_M_S2)}" integrator="RK4"/>
        </mujoco>
        """

        root = ET.fromstring(mujoco_xml)
        option = root.find("option")
        assert option is not None, "option element not found"
        gravity = option.get("gravity")
        assert gravity is not None, "gravity attribute not found"

        # Extract numerical value
        gz = float(gravity.split()[-1])
        assert 9.0 < abs(gz) < 10.0

    def test_multiple_physical_constants_in_xml(self):
        """Multiple PhysicalConstants can be used in same XML."""
        xml = f"""
        <physics>
            <gravity value="{float(GRAVITY_M_S2)}"/>
            <mass value="{float(GOLF_BALL_MASS_KG)}"/>
            <air_density value="{float(AIR_DENSITY_SEA_LEVEL_KG_M3)}"/>
        </physics>
        """

        root = ET.fromstring(xml)

        # All should have pure numeric values
        gravity_elem = root.find("gravity")
        mass_elem = root.find("mass")
        density_elem = root.find("air_density")
        assert (
            gravity_elem is not None
            and mass_elem is not None
            and density_elem is not None
        )

        gravity_val = gravity_elem.get("value")
        mass_val = mass_elem.get("value")
        density_val = density_elem.get("value")
        assert (
            gravity_val is not None and mass_val is not None and density_val is not None
        )

        gravity = float(gravity_val)
        mass = float(mass_val)
        density = float(density_val)

        assert 9.0 < gravity < 10.0
        assert 0.04 < mass < 0.05  # ~45g
        assert 1.0 < density < 1.5  # ~1.225 kg/m³

    def test_physical_constant_arithmetic_in_xml(self):
        """PhysicalConstants work in arithmetic expressions."""
        # Compute effective gravity (e.g., for incline)
        angle_rad = np.pi / 6  # 30 degrees
        g_eff = float(GRAVITY_M_S2 * np.sin(angle_rad))

        xml = f'<force value="{g_eff}"/>'
        root = ET.fromstring(xml)

        force_val = root.get("value")
        assert force_val is not None, "value attribute not found"
        force = float(force_val)
        assert abs(force - (9.80665 * 0.5)) < 0.01

    def test_physical_constant_behaves_as_float(self):
        """PhysicalConstants should work in all float contexts."""
        g = GRAVITY_M_S2

        # Arithmetic
        assert isinstance(g * 2.0, (int, float))
        assert isinstance(g + 1.0, (int, float))
        assert isinstance(g / 2.0, (int, float))

        # Comparisons
        assert g > 9.0
        assert g < 10.0
        assert g == pytest.approx(9.80665)

        # Formatting
        formatted = f"{g:.3f}"
        assert formatted == "9.807"

        # Explicit float conversion
        assert isinstance(float(g), float)
        assert float(g) == pytest.approx(9.80665)

    def test_custom_physical_constant_in_xml(self):
        """User-defined PhysicalConstants work the same way."""
        custom_gravity = PhysicalConstant(
            3.71, "m/s^2", "NASA Mars Fact Sheet", "Mars surface gravity"
        )

        xml = f'<mars gravity="{float(custom_gravity)}"/>'
        root = ET.fromstring(xml)

        mars_g_val = root.get("gravity")
        assert mars_g_val is not None, "gravity attribute not found"
        mars_g = float(mars_g_val)
        assert mars_g == pytest.approx(3.71)

    def test_prevent_accidental_string_concat(self):
        """Ensure float() prevents string concatenation issues."""
        # Wrong: string concatenation
        bad_xml = '<val>" + str(GRAVITY_M_S2) + "</val>'  # Deliberate bad example  # noqa: F841

        # Right: numeric interpolation
        good_xml = f"<val>{float(GRAVITY_M_S2)}</val>"

        root = ET.fromstring(good_xml)
        assert root.text is not None, "element text not found"
        val = float(root.text)
        assert val == pytest.approx(9.80665)


class TestPhysicalConstantEdgeCases:
    """Test edge cases and potential failure modes."""

    def test_very_small_physical_constant(self):
        """Small constants (e.g., epsilon) format correctly."""
        epsilon = PhysicalConstant(
            1e-15, "dimensionless", "Machine precision", "Float64 epsilon"
        )

        xml = f"<tolerance>{float(epsilon)}</tolerance>"
        root = ET.fromstring(xml)

        assert root.text is not None, "element text not found"
        tol = float(root.text)
        assert tol == 1e-15

    def test_very_large_physical_constant(self):
        """Large constants (e.g., speed of light) format correctly."""
        from shared.python.constants import SPEED_OF_LIGHT_M_S

        xml = f"<speed>{float(SPEED_OF_LIGHT_M_S)}</speed>"
        root = ET.fromstring(xml)

        assert root.text is not None, "element text not found"
        speed = float(root.text)
        assert speed == 299792458.0

    def test_negative_physical_constant(self):
        """Negative values (e.g., downward gravity) work."""
        # Gravity pointing down
        gz = -float(GRAVITY_M_S2)

        xml = f"<gravity_z>{gz}</gravity_z>"
        root = ET.fromstring(xml)

        assert root.text is not None, "element text not found"
        val = float(root.text)
        assert val < 0
        assert val == pytest.approx(-9.80665)

    def test_physical_constant_in_attribute_vs_text(self):
        """PhysicalConstants work as both attributes and text content."""
        xml = f"""
        <param gravity_attr="{float(GRAVITY_M_S2)}">
            {float(GRAVITY_M_S2)}
        </param>
        """

        root = ET.fromstring(xml)
        attr_val_str = root.get("gravity_attr")
        assert attr_val_str is not None and root.text is not None
        attr_val = float(attr_val_str)
        text_val = float(root.text.strip())

        assert attr_val == pytest.approx(9.80665)
        assert text_val == pytest.approx(9.80665)


def test_regression_pr303_gravity_xml_bug():
    """Regression test for PR303 PhysicalConstant XML bug.

    Bug: Using {GRAVITY_M_S2} directly in f-string produced:
        gravity="0 0 -PhysicalConstant(9.807, unit='m/s^2')"

    Fix: Use {float(GRAVITY_M_S2)} to get numeric value:
        gravity="0 0 -9.80665"
    """
    # This is the pattern that caused the bug (before fix)
    buggy_template = f"<gravity>{GRAVITY_M_S2}</gravity>"
    assert "PhysicalConstant" in buggy_template, "Bug pattern should include repr"

    # This is the correct pattern (after fix)
    fixed_template = f"<gravity>{float(GRAVITY_M_S2)}</gravity>"
    assert "PhysicalConstant" not in fixed_template, "Fix should only have number"

    # Verify fix works
    root = ET.fromstring(fixed_template)
    assert root.text is not None, "element text not found"
    val = float(root.text)
    assert val == pytest.approx(9.80665)
