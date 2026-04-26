"""Tests for calc_backend protocols and Pydantic contracts.

These tests verify the structural typing contracts and data validation
without requiring FastAPI or external calculator packages.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError


class TestCalculationEngineProtocol:
    """Tests for the CalculationEngine structural protocol."""

    def test_protocol_is_importable(self) -> None:
        """Protocol should be importable from calc_backend."""
        from src.shared.python.calc_backend import CalculationEngine

        assert CalculationEngine is not None

    def test_protocol_structural_check(self) -> None:
        """A class with calculate() should satisfy the protocol at runtime."""
        from pydantic import BaseModel

        from src.shared.python.calc_backend import CalculationEngine

        class FakeRequest(BaseModel):
            value: float

        class FakeResponse(BaseModel):
            result: float

        class ConcreteEngine:
            def calculate(self, request: FakeRequest) -> FakeResponse:
                return FakeResponse(result=request.value * 2)

        engine = ConcreteEngine()
        assert isinstance(engine, CalculationEngine)

    def test_protocol_negative_no_calculate_method(self) -> None:
        """A class without calculate() should NOT satisfy the protocol."""
        from src.shared.python.calc_backend import CalculationEngine

        class NotAnEngine:
            def compute(self, x: float) -> float:
                return x

        obj = NotAnEngine()
        assert not isinstance(obj, CalculationEngine)

    def test_validation_mixin_protocol(self) -> None:
        """ValidationMixin protocol should be checkable at runtime."""
        from src.shared.python.calc_backend import ValidationMixin

        class ConcreteValidator:
            def validate_inputs(self, request: object) -> None:
                if not request:
                    raise ValueError("empty request")

        validator = ConcreteValidator()
        assert isinstance(validator, ValidationMixin)

    def test_expression_evaluator_protocol(self) -> None:
        """ExpressionEvaluator protocol should be importable."""
        from src.shared.python.calc_backend import ExpressionEvaluator

        assert ExpressionEvaluator is not None


class TestFlowRateContracts:
    """Tests for flow rate Pydantic contracts."""

    def test_valid_request(self) -> None:
        """Valid flow rate request should parse correctly."""
        from src.shared.python.calc_backend.contracts.flow_rate import (
            FlowRateConvertRequest,
        )

        req = FlowRateConvertRequest(value=10.0, from_unit="kg/s", to_unit="lb/h")
        assert req.value == 10.0
        assert req.from_unit == "kg/s"
        assert req.to_unit == "lb/h"
        assert req.category == "mass"  # default

    def test_default_category_is_mass(self) -> None:
        """Default category should be 'mass'."""
        from src.shared.python.calc_backend.contracts.flow_rate import (
            FlowRateConvertRequest,
        )

        req = FlowRateConvertRequest(value=1.0, from_unit="kg/s", to_unit="g/s")
        assert req.category == "mass"

    def test_custom_category(self) -> None:
        """Custom category should be accepted."""
        from src.shared.python.calc_backend.contracts.flow_rate import (
            FlowRateConvertRequest,
        )

        req = FlowRateConvertRequest(
            value=5.0, from_unit="m3/s", to_unit="L/s", category="volumetric"
        )
        assert req.category == "volumetric"

    def test_response_model(self) -> None:
        """Response model should store result and echo units."""
        from src.shared.python.calc_backend.contracts.flow_rate import (
            FlowRateConvertResponse,
        )

        resp = FlowRateConvertResponse(
            result=3600.0, from_unit="kg/s", to_unit="kg/h", category="mass"
        )
        assert resp.result == pytest.approx(3600.0)
        assert resp.from_unit == "kg/s"

    def test_missing_required_field_raises(self) -> None:
        """Missing required fields should raise ValidationError."""
        from src.shared.python.calc_backend.contracts.flow_rate import (
            FlowRateConvertRequest,
        )

        with pytest.raises(ValidationError):
            FlowRateConvertRequest(from_unit="kg/s", to_unit="lb/h")  # type: ignore[call-arg]


class TestRotationConverterContracts:
    """Tests for rotation converter Pydantic contracts."""

    def test_valid_quaternion_request(self) -> None:
        """Quaternion request should parse correctly."""
        from src.shared.python.calc_backend.contracts.rotation_converter import (
            RotationConverterRequest,
        )

        req = RotationConverterRequest(
            type="quaternion",
            value=[1.0, 0.0, 0.0, 0.0],
        )
        assert req.type == "quaternion"
        assert req.value == [1.0, 0.0, 0.0, 0.0]

    def test_valid_euler_request(self) -> None:
        """Euler request should parse with default convention."""
        from src.shared.python.calc_backend.contracts.rotation_converter import (
            RotationConverterRequest,
        )

        req = RotationConverterRequest(type="euler", value=[0.1, 0.2, 0.3])
        assert req.euler_convention == "xyz"  # default

    def test_invalid_rotation_type_raises(self) -> None:
        """Invalid rotation type should raise ValidationError."""
        from src.shared.python.calc_backend.contracts.rotation_converter import (
            RotationConverterRequest,
        )

        with pytest.raises(ValidationError):
            RotationConverterRequest(type="invalid_type", value=[1.0, 0.0, 0.0, 0.0])

    def test_reference_frame_twist_requires_transform_and_twist(self) -> None:
        """twist_frame_conversion requires both transform and twist."""
        from src.shared.python.calc_backend.contracts.rotation_converter import (
            ReferenceFrameConversionRequest,
        )

        identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        twist = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]

        req = ReferenceFrameConversionRequest(
            operation="twist_frame_conversion",
            transform=identity,
            twist=twist,
        )
        assert req.operation == "twist_frame_conversion"

    def test_reference_frame_missing_transform_raises(self) -> None:
        """twist_frame_conversion without transform should raise ValidationError."""
        from src.shared.python.calc_backend.contracts.rotation_converter import (
            ReferenceFrameConversionRequest,
        )

        with pytest.raises(ValidationError):
            ReferenceFrameConversionRequest(
                operation="twist_frame_conversion",
                twist=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                # missing transform
            )

    def test_homogeneous_transform_requires_rotation_and_translation(self) -> None:
        """homogeneous_transform requires both rotation_matrix and translation."""
        from src.shared.python.calc_backend.contracts.rotation_converter import (
            ReferenceFrameConversionRequest,
        )

        rot = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        req = ReferenceFrameConversionRequest(
            operation="homogeneous_transform",
            rotation_matrix=rot,
            translation=[0.0, 0.0, 1.0],
        )
        assert req.operation == "homogeneous_transform"

    def test_so3_maps_requires_exactly_one_source(self) -> None:
        """so3_so3_maps with both so3_vector and rotation_matrix should raise."""
        from src.shared.python.calc_backend.contracts.rotation_converter import (
            ReferenceFrameConversionRequest,
        )

        with pytest.raises(ValidationError):
            ReferenceFrameConversionRequest(
                operation="so3_so3_maps",
                so3_vector=[0.1, 0.2, 0.3],
                rotation_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            )


class TestCalcBackendPackageVersion:
    """Tests for calc_backend package metadata."""

    def test_version_is_set(self) -> None:
        """calc_backend should expose a version string."""
        from src.shared.python import calc_backend

        assert hasattr(calc_backend, "__version__")
        assert isinstance(calc_backend.__version__, str)
        assert len(calc_backend.__version__) > 0

    def test_all_exports_importable(self) -> None:
        """All items in __all__ should be importable."""

        from src.shared.python import calc_backend

        for name in calc_backend.__all__:
            obj = getattr(calc_backend, name, None)
            assert obj is not None, f"{name} in __all__ but not importable"
