"""Unit tests for AI module exceptions."""

from __future__ import annotations

import pytest
from shared.python.ai.exceptions import (
    AIConnectionError,
    AIError,
    AIProviderError,
    AIRateLimitError,
    AITimeoutError,
    ScientificValidationError,
    ToolExecutionError,
    WorkflowError,
)


class TestAIError:
    """Tests for base AIError exception."""

    def test_error_message(self) -> None:
        """Test basic error message."""
        error = AIError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"

    def test_error_with_details(self) -> None:
        """Test error with additional details."""
        error = AIError("Error", details={"code": 42, "context": "test"})
        assert "code=42" in str(error)
        assert "context=test" in str(error)

    def test_error_details_default(self) -> None:
        """Test error with no details defaults to empty dict."""
        error = AIError("Error")
        assert error.details == {}


class TestAIProviderError:
    """Tests for AIProviderError exception."""

    def test_provider_error_basic(self) -> None:
        """Test basic provider error."""
        error = AIProviderError("API failed", provider="openai")
        assert error.provider == "openai"
        assert error.status_code is None

    def test_provider_error_with_status(self) -> None:
        """Test provider error with HTTP status."""
        error = AIProviderError("Not found", provider="anthropic", status_code=404)
        assert error.status_code == 404


class TestAIConnectionError:
    """Tests for AIConnectionError exception."""

    def test_connection_error(self) -> None:
        """Test connection error inherits from provider error."""
        error = AIConnectionError("Network failed", provider="ollama")
        assert isinstance(error, AIProviderError)
        assert error.provider == "ollama"


class TestAIRateLimitError:
    """Tests for AIRateLimitError exception."""

    def test_rate_limit_error(self) -> None:
        """Test rate limit error with retry info."""
        error = AIRateLimitError(
            "Too many requests",
            provider="openai",
            retry_after=30.5,
        )
        assert error.retry_after == 30.5
        assert error.status_code == 429


class TestAITimeoutError:
    """Tests for AITimeoutError exception."""

    def test_timeout_error(self) -> None:
        """Test timeout error with duration."""
        error = AITimeoutError(
            "Request timed out",
            provider="anthropic",
            timeout=60.0,
        )
        assert error.timeout == 60.0


class TestScientificValidationError:
    """Tests for ScientificValidationError exception."""

    def test_validation_error_basic(self) -> None:
        """Test basic scientific validation error."""
        error = ScientificValidationError(
            "Energy conservation violated",
            check_name="energy_balance",
        )
        assert error.check_name == "energy_balance"

    def test_validation_error_with_values(self) -> None:
        """Test validation error with value and threshold."""
        error = ScientificValidationError(
            "Torque too high",
            check_name="torque_limit",
            value=1000.0,
            threshold=500.0,
        )
        assert error.value == 1000.0
        assert error.threshold == 500.0


class TestWorkflowError:
    """Tests for WorkflowError exception."""

    def test_workflow_error_basic(self) -> None:
        """Test basic workflow error."""
        error = WorkflowError(
            "Step failed",
            workflow_id="first_analysis",
        )
        assert error.workflow_id == "first_analysis"
        assert error.step_id is None

    def test_workflow_error_with_step(self) -> None:
        """Test workflow error with step ID."""
        error = WorkflowError(
            "Validation failed",
            workflow_id="inverse_dynamics",
            step_id="run_simulation",
        )
        assert error.step_id == "run_simulation"


class TestToolExecutionError:
    """Tests for ToolExecutionError exception."""

    def test_tool_error_basic(self) -> None:
        """Test basic tool execution error."""
        error = ToolExecutionError(
            "Tool crashed",
            tool_name="load_c3d",
        )
        assert error.tool_name == "load_c3d"
        assert error.parameters == {}

    def test_tool_error_with_parameters(self) -> None:
        """Test tool error with parameters."""
        error = ToolExecutionError(
            "Invalid file",
            tool_name="load_c3d",
            parameters={"file_path": "/bad/path.c3d"},
        )
        assert error.parameters == {"file_path": "/bad/path.c3d"}


class TestExceptionHierarchy:
    """Tests for exception inheritance hierarchy."""

    def test_provider_errors_inherit_from_ai_error(self) -> None:
        """Test that all provider errors inherit from AIError."""
        assert issubclass(AIProviderError, AIError)
        assert issubclass(AIConnectionError, AIProviderError)
        assert issubclass(AIRateLimitError, AIProviderError)
        assert issubclass(AITimeoutError, AIProviderError)

    def test_other_errors_inherit_from_ai_error(self) -> None:
        """Test that other AI errors inherit from AIError."""
        assert issubclass(ScientificValidationError, AIError)
        assert issubclass(WorkflowError, AIError)
        assert issubclass(ToolExecutionError, AIError)

    def test_catch_all_with_ai_error(self) -> None:
        """Test that AIError can catch all AI exceptions."""
        exceptions = [
            AIProviderError("test", provider="test"),
            AIConnectionError("test", provider="test"),
            AIRateLimitError("test", provider="test"),
            AITimeoutError("test", provider="test"),
            ScientificValidationError("test", check_name="test"),
            WorkflowError("test", workflow_id="test"),
            ToolExecutionError("test", tool_name="test"),
        ]
        for exc in exceptions:
            with pytest.raises(AIError):
                raise exc
