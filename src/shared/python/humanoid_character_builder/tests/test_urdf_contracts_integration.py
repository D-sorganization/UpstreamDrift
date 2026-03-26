"""test_urdf_contracts_integration.py module."""

from unittest.mock import MagicMock, patch

import pytest
from humanoid_character_builder.contracts import ContractViolationError
from humanoid_character_builder.core.body_parameters import BodyParameters
from humanoid_character_builder.core.segment_definitions import (
    GeometrySpec,
    GeometryType,
    JointDefinition,
    JointLimits,
    JointType,
    SegmentDefinition,
)
from humanoid_character_builder.generators.urdf_generator import HumanoidURDFGenerator


class TestURDFContracts:
    def test_negative_mass_violation(self):
        generator = HumanoidURDFGenerator()
        params = BodyParameters()

        # We need to call _generate_link with negative mass
        # Since it is a private method, we can call it directly for testing purposes
        # provided we construct valid arguments for other params

        segment_def = SegmentDefinition(name="test", visual_geometry=GeometrySpec(GeometryType.BOX))

        with pytest.raises(ContractViolationError, match="Mass must be positive"):
            generator._generate_link(
                segment_name="test",
                segment_def=segment_def,
                params=params,
                mass=-1.0,  # Violation
                dimensions={},
                gender_factor=1.0,
                mesh_dir=None,
            )

    def test_invalid_inertia_violation(self):
        generator = HumanoidURDFGenerator()
        # params is mocked below
        segment_def = SegmentDefinition(name="test", visual_geometry=GeometrySpec(GeometryType.BOX))

        # Mock params to have inertia override with negative values
        # We need to ensure we can access the mocked override

        # Create a mock that behaves like the expected segment params object
        seg_params = MagicMock()
        seg_params.has_inertia_override.return_value = True
        # ixx negative -> invalid
        seg_params.inertia_override = {
            "ixx": -1.0,
            "iyy": 1.0,
            "izz": 1.0,
            "ixy": 0.0,
            "ixz": 0.0,
            "iyz": 0.0,
        }

        with pytest.raises(ContractViolationError, match="Inertia must be positive definite"):
            generator._compute_segment_inertia(
                segment_name="test",
                segment_def=segment_def,
                seg_params=seg_params,
                mass=1.0,
                dimensions={"length": 0.1, "width": 0.1, "depth": 0.1},
                gender_factor=1.0,
                mesh_dir=None,
            )

    def test_invalid_joint_limits_violation(self):
        generator = HumanoidURDFGenerator()

        # Create invalid joint definition
        joint_def = JointDefinition(
            name="bad_joint",
            joint_type=JointType.REVOLUTE,
            parent_segment="a",
            child_segment="b",
            limits=JointLimits(lower=1.0, upper=0.0),  # Invalid: lower > upper
        )

        with pytest.raises(ContractViolationError, match="Joint limits invalid"):
            generator._generate_single_joint("bad_joint", joint_def)

    def test_generate_postcondition(self):
        generator = HumanoidURDFGenerator()
        params = BodyParameters()

        # Mock internal methods to avoid actual computation (which might fail due to defaults)
        # and mock _build_urdf_xml to return invalid XML to trigger the postcondition
        with (
            patch.object(HumanoidURDFGenerator, "_generate_link"),
            patch.object(HumanoidURDFGenerator, "_generate_joint"),
            patch.object(HumanoidURDFGenerator, "_build_urdf_xml", return_value="invalid xml"),
            pytest.raises(ContractViolationError, match="Generated URDF must be valid XML"),
        ):
            generator.generate(params)
