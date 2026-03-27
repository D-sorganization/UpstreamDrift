"""Tests for the Rotation Converter API router.

This test file adheres to the Fleet-Wide Shared Component Testing Strategy.
It mocks the `rotation_converter` package from the Tools repo to verify
that the API layer correctly implements the contract boundaries.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Skip this module if the rotation_converter package is not installed
pytest.importorskip(
    "rotation_converter",
    reason="rotation_converter package not installed",
)

from src.shared.python.calc_backend.routers.rotation_converter import (
    router,
)  # noqa: E402,I001

_app = FastAPI()
_app.include_router(router)
client = TestClient(_app)


@pytest.fixture
def mock_rotation_class():
    """Mock the Rotation class from rotation_converter.converter."""
    with patch(
        "src.shared.python.calc_backend.routers.rotation_converter.Rotation"
    ) as mock_cls:
        mock_rot = MagicMock()
        mock_cls.return_value = mock_rot
        # Set up output representations
        mock_rot.as_quaternion.return_value = MagicMock(
            tolist=lambda: [0.0, 0.0, 0.0, 1.0]
        )
        mock_rot.as_euler.return_value = [0.0, 0.0, 0.0]
        mock_rot.as_axis_angle.return_value = (
            MagicMock(tolist=lambda: [0.0, 0.0, 1.0]),
            0.0,
        )
        mock_rot.as_rodrigues.return_value = MagicMock(tolist=lambda: [0.0, 0.0, 0.0])
        mock_rot.as_rotation_matrix.return_value = MagicMock(
            tolist=lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        )
        yield mock_cls, mock_rot


def test_compute_rotation_quaternion_success(mock_rotation_class) -> None:
    """Valid quaternion input should return all rotation representations."""
    mock_cls, mock_rot = mock_rotation_class
    with patch(
        "src.shared.python.calc_backend.routers.rotation_converter.Rotation", mock_cls
    ):
        payload = {
            "type": "quaternion",
            "value": [0.0, 0.0, 0.0, 1.0],
            "euler_convention": "xyz",
        }
        response = client.post("/api/calc/rotation-converter", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "representations" in data
        mock_cls.from_quaternion.assert_called_once_with([0.0, 0.0, 0.0, 1.0])


def test_compute_rotation_euler_success(mock_rotation_class) -> None:
    """Valid Euler angles should be parsed and passed to the Rotation class."""
    mock_cls, mock_rot = mock_rotation_class
    with patch(
        "src.shared.python.calc_backend.routers.rotation_converter.Rotation", mock_cls
    ):
        payload = {
            "type": "euler",
            "value": [0.1, 0.2, 0.3],
            "euler_convention": "xyz",
        }
        response = client.post("/api/calc/rotation-converter", json=payload)

        assert response.status_code == 200
        mock_cls.from_euler.assert_called_once_with(0.1, 0.2, 0.3, "xyz")


def test_compute_rotation_invalid_euler_length(mock_rotation_class) -> None:
    """Euler angles with wrong number of elements → 422 via internal validation."""
    mock_cls, _ = mock_rotation_class
    with patch(
        "src.shared.python.calc_backend.routers.rotation_converter.Rotation", mock_cls
    ):
        payload = {
            "type": "euler",
            "value": [0.1, 0.2],  # Only 2, needs 3
            "euler_convention": "xyz",
        }
        response = client.post("/api/calc/rotation-converter", json=payload)
        assert response.status_code == 422


def test_compute_rotation_unknown_type(mock_rotation_class) -> None:
    """Unknown rotation type should return 422."""
    mock_cls, _ = mock_rotation_class
    with patch(
        "src.shared.python.calc_backend.routers.rotation_converter.Rotation", mock_cls
    ):
        payload = {
            "type": "unknown_format",
            "value": [1.0],
            "euler_convention": "xyz",
        }
        response = client.post("/api/calc/rotation-converter", json=payload)
        assert response.status_code == 422


def test_reference_frame_conversion() -> None:
    """Reference frame operation (Lie group) should route correctly."""
    with patch(
        "src.shared.python.calc_backend.routers.rotation_converter.compute_reference_frame_operation"
    ) as mock_op:
        mock_result = MagicMock()
        mock_result.operation = "exp_map"
        mock_result.results = {"matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
        mock_result.explanation_markdown = "## Test"
        mock_result.explanation_latex = "$$I$$"
        mock_op.return_value = mock_result

        payload = {
            "operation": "exp_map",
            "so3_vector": [0.0, 0.0, 0.0],
        }
        response = client.post("/reference-frame", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["operation"] == "exp_map"
        mock_op.assert_called_once()
