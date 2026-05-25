"""TDD tests for AIAssistantPanel sub-module decomposition.

Covers three extracted modules:
  - composer.py   — ChatInput + input-area factory helpers
  - transcript.py — MessageWidget + transcript/scroll area
  - streaming.py  — StreamWorker

These tests are written BEFORE the sub-modules exist (red phase).
They will pass once the split is in place.

All tests are import-only or inspect class attributes without constructing
Qt widgets, so they run safely in headless CI environments.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

# Skip the entire module if PyQt6 is not available.
try:
    import PyQt6.QtCore  # noqa: F401
except (ImportError, OSError) as _exc:
    pytest.skip(f"PyQt6 not loadable: {_exc}", allow_module_level=True)


# ---------------------------------------------------------------------------
# transcript.py — MessageWidget
# ---------------------------------------------------------------------------


class TestMessageWidgetImports:
    """MessageWidget is importable from the transcript sub-module."""

    def test_importable_from_transcript(self) -> None:
        """MessageWidget should be importable from the transcript sub-module."""
        from src.shared.python.ai.gui.assistant import transcript  # noqa: F401

        assert hasattr(transcript, "MessageWidget")

    def test_message_widget_is_a_class(self) -> None:
        """MessageWidget must be a class (not a function or alias)."""
        from src.shared.python.ai.gui.assistant.transcript import MessageWidget

        assert isinstance(MessageWidget, type)

    def test_message_widget_has_required_methods(self) -> None:
        """MessageWidget declares the expected public API."""
        from src.shared.python.ai.gui.assistant.transcript import MessageWidget

        for method in (
            "get_content",
            "set_content",
            "append_content",
            "refresh_theme",
            "_get_role_display",
        ):
            assert hasattr(MessageWidget, method), f"Missing method: {method}"


# ---------------------------------------------------------------------------
# streaming.py — StreamWorker
# ---------------------------------------------------------------------------


class TestStreamWorkerImports:
    """StreamWorker is importable from the streaming sub-module."""

    def test_importable_from_streaming(self) -> None:
        """StreamWorker should be importable from the streaming sub-module."""
        from src.shared.python.ai.gui.assistant import streaming  # noqa: F401

        assert hasattr(streaming, "StreamWorker")

    def test_stream_worker_is_a_class(self) -> None:
        """StreamWorker must be a class."""
        from src.shared.python.ai.gui.assistant.streaming import StreamWorker

        assert isinstance(StreamWorker, type)

    def test_has_signals(self) -> None:
        """StreamWorker exposes chunk_received, finished, and error signals."""
        from src.shared.python.ai.gui.assistant.streaming import StreamWorker

        assert hasattr(StreamWorker, "chunk_received")
        assert hasattr(StreamWorker, "finished")
        assert hasattr(StreamWorker, "error")

    def test_has_run_method(self) -> None:
        """StreamWorker implements the QThread.run interface."""
        from src.shared.python.ai.gui.assistant.streaming import StreamWorker

        assert hasattr(StreamWorker, "run")


# ---------------------------------------------------------------------------
# composer.py — ChatInput
# ---------------------------------------------------------------------------


class TestChatInputImports:
    """ChatInput is importable from the composer sub-module."""

    def test_importable_from_composer(self) -> None:
        """ChatInput should be importable from the composer sub-module."""
        from src.shared.python.ai.gui.assistant import composer  # noqa: F401

        assert hasattr(composer, "ChatInput")

    def test_chat_input_is_a_class(self) -> None:
        """ChatInput must be a class."""
        from src.shared.python.ai.gui.assistant.composer import ChatInput

        assert isinstance(ChatInput, type)

    def test_has_submit_signal(self) -> None:
        """ChatInput exposes submit_requested signal."""
        from src.shared.python.ai.gui.assistant.composer import ChatInput

        assert hasattr(ChatInput, "submit_requested")

    def test_has_key_press_handler(self) -> None:
        """ChatInput overrides keyPressEvent to intercept Enter."""
        from src.shared.python.ai.gui.assistant.composer import ChatInput

        assert hasattr(ChatInput, "keyPressEvent")


# ---------------------------------------------------------------------------
# package __init__ — sub-package is importable
# ---------------------------------------------------------------------------


class TestAssistantPackage:
    """The assistant sub-package is a proper Python package."""

    def test_package_importable(self) -> None:
        """src.shared.python.ai.gui.assistant must be importable."""
        import src.shared.python.ai.gui.assistant  # noqa: F401

    def test_package_exposes_all_classes(self) -> None:
        """The package __init__ should re-export all three key classes."""
        import src.shared.python.ai.gui.assistant as pkg

        for name in ("MessageWidget", "StreamWorker", "ChatInput"):
            assert hasattr(pkg, name), f"Package missing: {name}"


# ---------------------------------------------------------------------------
# assistant_panel.py — backward-compat re-exports
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """AIAssistantPanel must remain importable from the old path."""

    def test_assistant_panel_still_importable(self) -> None:
        """The original import path must still work."""

    def test_message_widget_still_importable(self) -> None:
        """MessageWidget must still be importable from the original module."""
        from src.shared.python.ai.gui.assistant_panel import MessageWidget  # noqa: F401

    def test_stream_worker_still_importable(self) -> None:
        """StreamWorker must still be importable from the original module."""
        from src.shared.python.ai.gui.assistant_panel import StreamWorker  # noqa: F401

    def test_chat_input_still_importable(self) -> None:
        """ChatInput must still be importable from the original module."""
        from src.shared.python.ai.gui.assistant_panel import ChatInput  # noqa: F401

    def test_submodule_classes_are_same_objects(self) -> None:
        """Re-exports must be the canonical classes, not copies."""
        from src.shared.python.ai.gui.assistant_panel import MessageWidget as Old
        from src.shared.python.ai.gui.assistant.transcript import (
            MessageWidget as New,
        )

        assert Old is New

    def test_stream_worker_same_object(self) -> None:
        """StreamWorker re-export is the same class as the canonical location."""
        from src.shared.python.ai.gui.assistant_panel import StreamWorker as Old
        from src.shared.python.ai.gui.assistant.streaming import StreamWorker as New

        assert Old is New

    def test_chat_input_same_object(self) -> None:
        """ChatInput re-export is the same class as the canonical location."""
        from src.shared.python.ai.gui.assistant_panel import ChatInput as Old
        from src.shared.python.ai.gui.assistant.composer import ChatInput as New

        assert Old is New
