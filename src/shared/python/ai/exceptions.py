"""Exception hierarchy for the AI Assistant integration layer.

All AI-related exceptions inherit from AIError, providing a consistent
error handling pattern across the module.

Exception Hierarchy:
    AIError (base)
    ├── AIProviderError (provider communication issues)
    │   ├── AIConnectionError (network/connection failures)
    │   ├── AIRateLimitError (rate limit exceeded)
    │   └── AITimeoutError (request timeout)
    ├── ScientificValidationError (physics validation failures)
    ├── WorkflowError (workflow execution issues)
    └── ToolExecutionError (tool call failures)

Example:
    >>> from shared.python.ai.exceptions import AIProviderError
    >>> try:
    ...     adapter.send_message("test")
    ... except AIProviderError as e:
    ...     logger.error("AI provider failed: %s", e)
"""

from __future__ import annotations


class AIError(Exception):
    """Base exception for all AI-related errors.

    All AI module exceptions inherit from this class, enabling
    catch-all error handling when needed.

    Attributes:
        message: Human-readable error description.
        details: Optional dictionary with additional context.
    """

    def __init__(
        self,
        message: str,
        details: dict[str, object] | None = None,
    ) -> None:
        """Initialize AI error.

        Args:
            message: Human-readable error description.
            details: Optional dictionary with additional context.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """Return string representation."""
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


class AIProviderError(AIError):
    """Exception for AI provider communication errors.

    Raised when there are issues communicating with the AI provider,
    such as authentication failures or API errors.

    Attributes:
        provider: Name of the AI provider that failed.
        status_code: HTTP status code if applicable.
    """

    def __init__(
        self,
        message: str,
        provider: str = "unknown",
        status_code: int | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        """Initialize provider error.

        Args:
            message: Human-readable error description.
            provider: Name of the AI provider.
            status_code: HTTP status code if applicable.
            details: Optional dictionary with additional context.
        """
        super().__init__(message, details)
        self.provider = provider
        self.status_code = status_code


class AIConnectionError(AIProviderError):
    """Exception for network connection failures.

    Raised when the AI provider cannot be reached due to network
    issues, DNS failures, or server unavailability.
    """



class AIRateLimitError(AIProviderError):
    """Exception for rate limit exceeded errors.

    Raised when the AI provider rejects requests due to rate limiting.
    Includes retry information when available.

    Attributes:
        retry_after: Seconds to wait before retrying [s].
    """

    def __init__(
        self,
        message: str,
        provider: str = "unknown",
        retry_after: float | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        """Initialize rate limit error.

        Args:
            message: Human-readable error description.
            provider: Name of the AI provider.
            retry_after: Seconds to wait before retrying [s].
            details: Optional dictionary with additional context.
        """
        super().__init__(message, provider, 429, details)
        self.retry_after = retry_after


class AITimeoutError(AIProviderError):
    """Exception for request timeout errors.

    Raised when an AI provider request exceeds the configured timeout.

    Attributes:
        timeout: The timeout value that was exceeded [s].
    """

    def __init__(
        self,
        message: str,
        provider: str = "unknown",
        timeout: float | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        """Initialize timeout error.

        Args:
            message: Human-readable error description.
            provider: Name of the AI provider.
            timeout: The timeout value that was exceeded [s].
            details: Optional dictionary with additional context.
        """
        super().__init__(message, provider, None, details)
        self.timeout = timeout


class ScientificValidationError(AIError):
    """Exception for scientific validation failures.

    Raised when AI-generated outputs fail physics consistency checks.
    This ensures AI never produces scientifically invalid results.

    Attributes:
        check_name: Name of the validation check that failed.
        value: The value that failed validation.
        threshold: The threshold that was exceeded.
    """

    def __init__(
        self,
        message: str,
        check_name: str,
        value: float | None = None,
        threshold: float | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        """Initialize scientific validation error.

        Args:
            message: Human-readable error description.
            check_name: Name of the validation check that failed.
            value: The value that failed validation.
            threshold: The threshold that was exceeded.
            details: Optional dictionary with additional context.
        """
        super().__init__(message, details)
        self.check_name = check_name
        self.value = value
        self.threshold = threshold


class WorkflowError(AIError):
    """Exception for workflow execution errors.

    Raised when a guided workflow encounters an unrecoverable error.

    Attributes:
        workflow_id: ID of the workflow that failed.
        step_id: ID of the step that failed.
    """

    def __init__(
        self,
        message: str,
        workflow_id: str,
        step_id: str | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        """Initialize workflow error.

        Args:
            message: Human-readable error description.
            workflow_id: ID of the workflow that failed.
            step_id: ID of the step that failed.
            details: Optional dictionary with additional context.
        """
        super().__init__(message, details)
        self.workflow_id = workflow_id
        self.step_id = step_id


class ToolExecutionError(AIError):
    """Exception for tool execution failures.

    Raised when an AI-invoked tool fails to execute properly.

    Attributes:
        tool_name: Name of the tool that failed.
        parameters: Parameters that were passed to the tool.
    """

    def __init__(
        self,
        message: str,
        tool_name: str,
        parameters: dict[str, object] | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        """Initialize tool execution error.

        Args:
            message: Human-readable error description.
            tool_name: Name of the tool that failed.
            parameters: Parameters that were passed to the tool.
            details: Optional dictionary with additional context.
        """
        super().__init__(message, details)
        self.tool_name = tool_name
        self.parameters = parameters or {}
