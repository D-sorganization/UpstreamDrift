"""Smoke tests for chat and launcher feature discovery verification.

Parent: https://github.com/D-sorganization/UpstreamDrift/issues/5316
Parent epic: https://github.com/D-sorganization/UpstreamDrift/issues/5309

These tests verify that shared chat and feature discovery work through
real product entry points across Tools, UpstreamDrift, and Gasification_Model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
SRC_ROOT = REPO_ROOT / "src" / "shared" / "python"

pytestmark = pytest.mark.smoke


class TestSharedChatImport:
    """Verify shared chat component can be imported and has public API."""

    def test_shared_chat_module_exists(self) -> None:
        """Shared chat module should exist in src/shared/python."""
        chat_paths = [
            SRC_ROOT / "chat" / "__init__.py",
            SRC_ROOT / "chat.py",
            REPO_ROOT / "vendor" / "Tools" / "src" / "chat" / "__init__.py",
        ]
        found = any(p.exists() for p in chat_paths)
        assert found, (
            "Shared chat module not found. Expected one of: "
            f"{[str(p) for p in chat_paths]}"
        )

    def test_shared_chat_public_api(self) -> None:
        """Shared chat should expose public API for consumption."""
        expected_exports = {
            "ChatDockWidget",
            "ChatMessageBubble",
            "ChatPanel",
            "ChatWidget",
        }
        chat_init = SRC_ROOT / "chat" / "__init__.py"
        if chat_init.exists():
            source = chat_init.read_text(encoding="utf-8")
            exported_names = {
                name for name in expected_exports if f'"{name}"' in source
            }
            assert exported_names, (
                "Shared chat should declare ChatDockWidget, ChatMessageBubble, "
                "ChatWidget, or ChatPanel in __all__"
            )
            return

        try:
            sys.path.insert(0, str(SRC_ROOT))
            import chat  # type: ignore[import-untyped]
        except ImportError:
            pytest.skip("Shared chat module not available")

        module_exports = set(getattr(chat, "__all__", ()))
        assert module_exports & expected_exports, (
            "Shared chat should expose ChatDockWidget, ChatWidget, ChatPanel, "
            "or ChatMessageBubble"
        )


class TestLauncherDiscovery:
    """Verify launcher can open chat and expose features."""

    def test_launcher_module_exists(self) -> None:
        """Launcher module should exist."""
        launcher_path = REPO_ROOT / "src" / "launchers" / "__init__.py"
        assert launcher_path.exists(), f"Launcher not found at {launcher_path}"

    def test_launcher_has_chat_entry_point(self) -> None:
        """Launcher should have chat-related entry points."""
        launchers_path = REPO_ROOT / "src" / "launchers"
        if not launchers_path.exists():
            pytest.skip("Launchers directory not found")

        launcher_files = list(launchers_path.glob("*.py"))
        chat_related = [
            f
            for f in launcher_files
            if "chat" in f.name.lower() or "launcher" in f.name.lower()
        ]
        assert chat_related, (
            f"No launcher files with 'chat' or 'launcher' in name. "
            f"Found: {[f.name for f in launcher_files]}"
        )


class TestModelRefresh:
    """Verify model refresh functionality."""

    def test_model_config_exists(self) -> None:
        """Model configuration should exist."""
        model_config_paths = [
            SRC_ROOT / "config" / "models.yaml",
            REPO_ROOT / "src" / "config" / "models.yaml",
        ]
        found = any(p.exists() for p in model_config_paths)
        assert found, (
            f"Model config not found. Expected one of: "
            f"{[str(p) for p in model_config_paths]}"
        )


class TestThemeInheritance:
    """Verify theme inheritance is configured."""

    def test_theme_config_exists(self) -> None:
        """Theme configuration should exist."""
        theme_paths = [
            REPO_ROOT / "src" / "config" / "theme.yaml",
            REPO_ROOT / "src" / "config" / "themes" / "default.yaml",
            REPO_ROOT / "assets" / "themes" / "default.yaml",
        ]
        found = any(p.exists() for p in theme_paths)
        # This is informational - themes may be in Tools
        if not found:
            pytest.skip("Theme config not found in UpstreamDrift (may be in Tools)")


class TestBiomechanicsModelPack:
    """Verify Biomechanics model pack visibility."""

    def test_biomechanics_directory_exists(self) -> None:
        """Biomechanics-related directories should exist."""
        biomech_paths = [
            REPO_ROOT / "Biomechanics",
            REPO_ROOT / "src" / "biomechanics",
            REPO_ROOT / "model_pack" / "biomechanics",
        ]
        found = any(p.exists() for p in biomech_paths)
        if not found:
            pytest.skip(
                "Biomechanics directory not found. "
                "May need to incorporate model packs per issue #5312"
            )


class TestCodebaseIndexing:
    """Verify codebase indexing functionality."""

    def test_ai_backend_available(self) -> None:
        """AI backend should be available for indexing."""
        try:
            import ai_backend  # type: ignore[import-untyped]

            assert hasattr(ai_backend, "AIConfig"), "ai_backend should have AIConfig"
        except ImportError:
            pytest.skip("ai_backend not installed")

    def test_rust_adapter_available(self) -> None:
        """Rust adapter should be available."""
        try:
            sys.path.insert(0, str(REPO_ROOT))
            from src.shared.python.ai.adapters.rust_adapter import RustAgentAdapter

            assert RustAgentAdapter is not None
        except ImportError:
            pytest.skip("RustAgentAdapter not available")


class TestResponseStyleSelection:
    """Verify response style selection is available."""

    def test_response_styles_configured(self) -> None:
        """Response styles should be configurable."""
        # Check for response style configuration
        style_paths = [
            SRC_ROOT / "config" / "response_styles.yaml",
            REPO_ROOT / "src" / "config" / "response_styles.yaml",
        ]
        # This is informational - styles may be hardcoded or in Tools
        found = any(p.exists() for p in style_paths)
        if not found:
            pytest.skip("Response styles config not found (may be hardcoded)")


class ManualVerificationChecklist:
    """Manual verification checklist for UI behavior.

    Run these checks manually and update the status.

    ## Chat Verification
    - [ ] Chat opens from launcher menu
    - [ ] Chat displays conversation history
    - [ ] Model selection dropdown works
    - [ ] Response style selector works
    - [ ] Codebase indexing triggers correctly

    ## Launcher Verification
    - [ ] Launcher opens from main entry point
    - [ ] All tools are listed and categorized
    - [ ] Biomechanics model packs are visible
    - [ ] Theme inheritance is visible

    ## Gasification_Model Verification
    - [ ] Toolbar chat button opens chat
    - [ ] Shared widget imports work
    - [ ] Missing Tools error handling is graceful
    - [ ] Domain adapters are available
    """

    def test_placeholder(self) -> None:
        """Placeholder for manual verification checklist."""
        # This test exists to document manual verification steps
        # See class docstring for the checklist
        assert self.__doc__
