"""Unit tests for data validation models introduced in issue #5918."""

import math

import pytest
from pydantic import ValidationError

from src.api.routes.simulation_ws import SetSpeedMessage


class TestSetSpeedMessageValidation:
    """Tests for SetSpeedMessage Pydantic model validation."""

    def test_rejects_nan_speed_factor(self) -> None:
        """SetSpeedMessage must reject NaN speed_factor."""
        with pytest.raises(ValidationError):
            SetSpeedMessage(action="set_speed", speed_factor=math.nan)

    def test_rejects_negative_speed_factor(self) -> None:
        """SetSpeedMessage must reject negative speed_factor."""
        with pytest.raises(ValidationError):
            SetSpeedMessage(action="set_speed", speed_factor=-1.0)

    def test_rejects_zero_speed_factor(self) -> None:
        """SetSpeedMessage must reject zero speed_factor (must be > 0)."""
        with pytest.raises(ValidationError):
            SetSpeedMessage(action="set_speed", speed_factor=0.0)

    def test_rejects_inf_speed_factor(self) -> None:
        """SetSpeedMessage must reject infinite speed_factor."""
        with pytest.raises(ValidationError):
            SetSpeedMessage(action="set_speed", speed_factor=math.inf)

    def test_accepts_valid_speed_factor(self) -> None:
        """SetSpeedMessage must accept a valid positive finite speed_factor."""
        msg = SetSpeedMessage(action="set_speed", speed_factor=2.0)
        assert msg.speed_factor == pytest.approx(2.0)
        assert msg.action == "set_speed"

    def test_default_speed_factor_is_one(self) -> None:
        """SetSpeedMessage default speed_factor should be 1.0."""
        msg = SetSpeedMessage(action="set_speed")
        assert msg.speed_factor == pytest.approx(1.0)

    def test_model_validate_from_dict(self) -> None:
        """SetSpeedMessage.model_validate should work with a plain dict."""
        msg = SetSpeedMessage.model_validate(
            {"action": "set_speed", "speed_factor": 0.5}
        )
        assert msg.speed_factor == pytest.approx(0.5)

    def test_model_validate_rejects_nan_via_dict(self) -> None:
        """model_validate must also reject NaN when called with a dict."""
        with pytest.raises(ValidationError):
            SetSpeedMessage.model_validate(
                {"action": "set_speed", "speed_factor": math.nan}
            )
