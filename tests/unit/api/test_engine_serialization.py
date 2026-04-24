"""Regression tests for engine API response serialization.

Ensures numpy arrays in engine state are properly converted to
JSON-serializable types (issue: PydanticSerializationError).
"""

import numpy as np
import pytest

from src.api.routes.engines import _sanitize_for_json


class TestSanitizeForJson:
    """Verify _sanitize_for_json converts numpy types to native Python."""

    def test_numpy_array_to_list(self) -> None:
        result = _sanitize_for_json(np.array([1.0, 2.0, 3.0]))
        assert result == [1.0, 2.0, 3.0]
        assert isinstance(result, list)

    def test_numpy_2d_array(self) -> None:
        result = _sanitize_for_json(np.array([[1, 2], [3, 4]]))
        assert result == [[1, 2], [3, 4]]

    def test_numpy_integer(self) -> None:
        result = _sanitize_for_json(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_numpy_float(self) -> None:
        result = _sanitize_for_json(np.float64(3.14))
        assert result == pytest.approx(3.14)
        assert isinstance(result, float)

    def test_nested_dict_with_numpy(self) -> None:
        data = {"state": np.array([1.0, 2.0]), "count": np.int32(5), "name": "test"}
        result = _sanitize_for_json(data)
        assert result == {"state": [1.0, 2.0], "count": 5, "name": "test"}

    def test_list_with_numpy(self) -> None:
        data = [np.array([1.0]), np.float32(2.0), "hello"]
        result = _sanitize_for_json(data)
        assert result[0] == [1.0]
        assert isinstance(result[1], float)
        assert result[2] == "hello"

    def test_none_passthrough(self) -> None:
        assert _sanitize_for_json(None) is None

    def test_plain_dict_passthrough(self) -> None:
        data = {"a": 1, "b": "hello"}
        assert _sanitize_for_json(data) == data

    def test_empty_array(self) -> None:
        result = _sanitize_for_json(np.array([]))
        assert result == []
