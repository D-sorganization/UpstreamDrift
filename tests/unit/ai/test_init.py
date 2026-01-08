"""Unit tests for AI package init."""


class TestAIPackageInit:
    """Tests for AI package exports."""

    def test_types_are_exported(self) -> None:
        """Test that core types are exported from package."""
        from shared.python.ai import (
            ExpertiseLevel,
            ProviderCapability,
        )

        # Verify types are accessible
        assert ExpertiseLevel.BEGINNER.value == 1
        assert ProviderCapability.STREAMING is not None

    def test_exceptions_are_exported(self) -> None:
        """Test that exceptions are exported from package."""
        from shared.python.ai import (
            AIError,
            AIProviderError,
            ScientificValidationError,
        )

        # Verify exceptions are accessible and inherit correctly
        assert issubclass(AIProviderError, AIError)
        assert issubclass(ScientificValidationError, AIError)

    def test_version_is_defined(self) -> None:
        """Test that package version is defined."""
        from shared.python.ai import __version__

        assert __version__ is not None
        assert isinstance(__version__, str)
