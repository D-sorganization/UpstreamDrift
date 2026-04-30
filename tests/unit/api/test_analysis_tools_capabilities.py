"""Tests for analysis_tools route capability checking (issue #2452).

Routes must return 501 Not Implemented (or 400) when the active physics engine
does not support the requested operation, rather than silently returning
fabricated success or zero-value measurements.
"""

from unittest.mock import MagicMock

import pytest


class TestBodyPositionSupportCheck:
    """set_body_position must surface unsupported-engine errors."""

    def test_check_raises_when_engine_has_neither_setter(self) -> None:
        """Engine with no setter methods → HTTPException (not silent success)."""
        from fastapi import HTTPException

        from src.api.routes.analysis_tools import _check_position_support

        engine = MagicMock()
        # Remove the setter attributes so hasattr returns False
        del engine.set_body_position
        del engine.set_body_rotation

        with pytest.raises(HTTPException) as exc_info:
            _check_position_support(engine)

        assert exc_info.value.status_code in (400, 501)

    def test_check_passes_when_engine_has_position_setter(self) -> None:
        """Engine with set_body_position → no exception raised."""
        from src.api.routes.analysis_tools import _check_position_support

        engine = MagicMock(spec=["set_body_position"])
        _check_position_support(engine)  # should not raise

    def test_check_passes_when_engine_has_rotation_setter(self) -> None:
        """Engine with set_body_rotation alone → no exception raised."""
        from src.api.routes.analysis_tools import _check_position_support

        engine = MagicMock(spec=["set_body_rotation"])
        _check_position_support(engine)  # should not raise


class TestBodyMeasurementSupportCheck:
    """measure_distance must raise when engine lacks get_body_position."""

    def test_get_positions_raises_when_engine_has_no_getter(self) -> None:
        """Engine without get_body_position → HTTPException (not zero vectors)."""
        from fastapi import HTTPException

        from src.api.routes.analysis_tools import _get_body_position_vectors

        engine = MagicMock()
        del engine.get_body_position  # ensure hasattr returns False

        with pytest.raises(HTTPException) as exc_info:
            _get_body_position_vectors(engine, "body_a", "body_b")

        assert exc_info.value.status_code in (400, 501)

    def test_get_positions_returns_vectors_when_supported(self) -> None:
        """Engine with get_body_position → returns actual position vectors."""
        import numpy as np

        from src.api.routes.analysis_tools import _get_body_position_vectors

        engine = MagicMock(spec=["get_body_position"])
        engine.get_body_position.side_effect = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
        ]

        pos_a, pos_b = _get_body_position_vectors(engine, "body_a", "body_b")
        assert pos_a == [1.0, 2.0, 3.0]
        assert pos_b == [4.0, 5.0, 6.0]

    def test_get_positions_handles_none_return(self) -> None:
        """Engine with get_body_position returning None → raises HTTPException."""
        from fastapi import HTTPException

        from src.api.routes.analysis_tools import _get_body_position_vectors

        engine = MagicMock(spec=["get_body_position"])
        engine.get_body_position.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            _get_body_position_vectors(engine, "body_a", "body_b")

        assert exc_info.value.status_code in (400, 501)
