"""Tests for src.shared.python.calc_backend.contracts.rotation_converter (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from src.shared.python.calc_backend.contracts.rotation_converter import (
    ReferenceFrameConversionRequest,
    ReferenceFrameConversionResponse,
    RotationConverterRequest,
)


class TestRotationConverterRequest:
    def test_quaternion_construction(self) -> None:
        req = RotationConverterRequest(type="quaternion", value=[1.0, 0.0, 0.0, 0.0])
        assert req.type == "quaternion"
        assert len(req.value) == 4

    def test_euler_construction(self) -> None:
        req = RotationConverterRequest(type="euler", value=[0.0, 0.0, 0.0])
        assert req.type == "euler"

    def test_axis_angle_construction(self) -> None:
        req = RotationConverterRequest(type="axis_angle", value=[0.0, 0.0, 1.0, 0.0])
        assert req.type == "axis_angle"

    def test_rodrigues_construction(self) -> None:
        req = RotationConverterRequest(type="rodrigues", value=[0.0, 0.0, 0.0])
        assert req.type == "rodrigues"

    def test_default_euler_convention(self) -> None:
        req = RotationConverterRequest(type="euler", value=[0.0, 0.0, 0.0])
        assert req.euler_convention == "xyz"

    def test_custom_euler_convention(self) -> None:
        req = RotationConverterRequest(
            type="euler", value=[0.0, 0.0, 0.0], euler_convention="zyx"
        )
        assert req.euler_convention == "zyx"

    def test_contracts_rotation_converter_invalid_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            RotationConverterRequest(type="invalid_type", value=[1.0, 0.0, 0.0, 0.0])  # type: ignore[arg-type]

    def test_missing_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            RotationConverterRequest(value=[1.0, 0.0, 0.0, 0.0])  # type: ignore[call-arg]


class TestReferenceFrameConversionRequest:
    def test_twist_frame_conversion_construction(self) -> None:
        req = ReferenceFrameConversionRequest(
            operation="twist_frame_conversion",
            transform=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            twist=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        assert req.operation == "twist_frame_conversion"

    def test_homogeneous_transform_construction(self) -> None:
        req = ReferenceFrameConversionRequest(
            operation="homogeneous_transform",
            rotation_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            translation=[0.0, 0.0, 0.0],
        )
        assert req.operation == "homogeneous_transform"

    def test_so3_maps_with_so3_vector(self) -> None:
        req = ReferenceFrameConversionRequest(
            operation="so3_so3_maps",
            so3_vector=[0.0, 0.0, 0.0],
        )
        assert req.operation == "so3_so3_maps"

    def test_so3_maps_with_so3_matrix(self) -> None:
        req = ReferenceFrameConversionRequest(
            operation="so3_so3_maps",
            so3_matrix=[[0, -1, 0], [1, 0, 0], [0, 0, 0]],
        )
        assert req.operation == "so3_so3_maps"

    def test_twist_without_transform_raises(self) -> None:
        with pytest.raises(ValidationError):
            ReferenceFrameConversionRequest(
                operation="twist_frame_conversion",
                twist=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                # missing transform
            )

    def test_homogeneous_without_rotation_raises(self) -> None:
        with pytest.raises(ValidationError):
            ReferenceFrameConversionRequest(
                operation="homogeneous_transform",
                translation=[0.0, 0.0, 0.0],
                # missing rotation_matrix
            )

    def test_so3_maps_multiple_sources_raises(self) -> None:
        with pytest.raises(ValidationError):
            ReferenceFrameConversionRequest(
                operation="so3_so3_maps",
                so3_vector=[0.0, 0.0, 0.0],
                so3_matrix=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            )

    def test_invalid_operation_raises(self) -> None:
        with pytest.raises(ValidationError):
            ReferenceFrameConversionRequest(operation="invalid_op")  # type: ignore[arg-type]


class TestReferenceFrameConversionResponse:
    def test_contracts_rotation_converter_construction(self) -> None:
        resp = ReferenceFrameConversionResponse(
            operation="homogeneous_transform",
            results={
                "matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            },
            explanation_markdown="The identity transform.",
            explanation_latex=r"T = I",
        )
        assert resp.operation == "homogeneous_transform"
        assert isinstance(resp.results, dict)

    def test_contracts_rotation_converter_has_required_fields(self) -> None:
        resp = ReferenceFrameConversionResponse(
            operation="so3_so3_maps",
            results={},
            explanation_markdown="",
            explanation_latex="",
        )
        assert resp.explanation_markdown == ""
        assert resp.explanation_latex == ""
