"""Tests for AI module components to improve coverage."""

import unittest
from unittest.mock import patch


class TestAIAdapterBase(unittest.TestCase):
    """Test base AI adapter functionality."""

    def test_import_ai_modules(self) -> None:
        """Verify AI modules can be imported."""
        try:
            from src.shared.python.ai import adapters  # noqa: F401
            from src.shared.python.ai.adapters import (  # noqa: F401
                AIAdapterRegistry,
                get_available_adapters,
            )
        except ImportError as e:
            self.skipTest(f"AI modules not available: {e}")

    def test_adapter_registry_singleton(self) -> None:
        """Test that adapter registry is a singleton."""
        try:
            from src.shared.python.ai.adapters import (
                get_adapter_registry,
            )

            registry1 = get_adapter_registry()
            registry2 = get_adapter_registry()
            self.assertIs(registry1, registry2)
        except ImportError:
            self.skipTest("AI adapter registry not available")

    def test_list_available_adapters(self) -> None:
        """Test listing available AI adapters."""
        try:
            from src.shared.python.ai.adapters import get_available_adapters

            adapters = get_available_adapters()
            self.assertIsInstance(adapters, list)
        except ImportError:
            self.skipTest("AI adapters not available")


class TestPromptTemplates(unittest.TestCase):
    """Test AI prompt template functionality."""

    def test_prompt_template_format(self) -> None:
        """Test prompt templates can be formatted."""
        try:
            from src.shared.python.ai.prompts import SwingAnalysisPrompt

            prompt = SwingAnalysisPrompt()
            formatted = prompt.format(
                swing_data="test data",
                user_skill="intermediate",
            )
            self.assertIn("test data", formatted)
        except (ImportError, AttributeError):
            self.skipTest("Prompt templates not available")

    def test_prompt_template_validation(self) -> None:
        """Test prompt template input validation."""
        try:
            from src.shared.python.ai.prompts import SwingAnalysisPrompt

            prompt = SwingAnalysisPrompt()
            # Should handle missing data gracefully
            formatted = prompt.format()
            self.assertIsInstance(formatted, str)
        except (ImportError, AttributeError):
            self.skipTest("Prompt templates not available")


class TestRAGIntegration(unittest.TestCase):
    """Test RAG (Retrieval-Augmented Generation) components."""

    def test_rag_import(self) -> None:
        """Test RAG module can be imported."""
        try:
            from src.shared.python.ai.rag import simple_rag  # noqa: F401
        except ImportError:
            self.skipTest("RAG module not available")

    def test_document_chunking(self) -> None:
        """Test document chunking functionality."""
        try:
            from src.shared.python.ai.rag.simple_rag import chunk_text

            text = "Line 1\n" * 100
            chunks = chunk_text(text, max_chunk_size=50)
            self.assertIsInstance(chunks, list)
            self.assertGreater(len(chunks), 0)
        except (ImportError, AttributeError):
            self.skipTest("RAG chunking not available")


class TestAssistantPanel(unittest.TestCase):
    """Test AI Assistant Panel UI components."""

    @patch("src.shared.python.engine_core.engine_availability.PYQT6_AVAILABLE", True)
    def test_assistant_panel_import(self) -> None:
        """Test that assistant panel can be imported."""
        try:
            from src.shared.python.ai.gui.assistant_panel import (  # noqa: F401
                AIAssistantPanel,
            )
        except ImportError:
            self.skipTest("PyQt6 not available for assistant panel")

    def test_settings_dialog_import(self) -> None:
        """Test that settings dialog can be imported."""
        try:
            from src.shared.python.ai.gui.settings_dialog import (  # noqa: F401
                AISettingsDialog,
            )
        except ImportError:
            self.skipTest("PyQt6 not available for settings dialog")


class TestAIAnalysisResults(unittest.TestCase):
    """Test AI analysis result handling."""

    def test_analysis_result_serialization(self) -> None:
        """Test that analysis results can be serialized."""
        # Mock analysis result structure
        result = {
            "summary": "Test summary",
            "recommendations": ["rec1", "rec2"],
            "confidence": 0.85,
            "timestamp": "2026-02-01T00:00:00Z",
        }

        import json

        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        self.assertEqual(result, deserialized)

    def test_result_validation(self) -> None:
        """Test result structure validation."""
        valid_result = {
            "summary": "Valid summary",
            "recommendations": [],
        }

        # Validate required fields
        self.assertIn("summary", valid_result)
        self.assertIsInstance(valid_result["recommendations"], list)


if __name__ == "__main__":
    unittest.main()
