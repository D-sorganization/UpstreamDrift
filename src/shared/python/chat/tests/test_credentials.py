"""Tests for CredentialManager secure API key storage.

Covers:
- Construction validation
- Store/retrieve/delete API keys (mocked keyring)
- Environment variable fallback
- Provider listing
- Error handling when keyring is unavailable
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from chat.credentials import CredentialManager

# ── Construction tests ───────────────────────────────────────────────


class TestCredentialManagerConstruction:
    def test_default_service_name(self) -> None:
        mgr = CredentialManager()
        assert mgr.service_name == "shared_chat_ai"

    def test_custom_service_name(self) -> None:
        mgr = CredentialManager(service_name="my_app")
        assert mgr.service_name == "my_app"

    def test_empty_service_name_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            CredentialManager(service_name="")

    def test_whitespace_service_name_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            CredentialManager(service_name="   ")


# ── Keyring storage tests (mocked) ──────────────────────────────────


class TestCredentialManagerWithKeyring:
    """Tests using a mocked keyring backend."""

    def _mock_keyring(self) -> MagicMock:
        """Create a mock keyring that behaves like a real one."""
        kr = MagicMock()
        store: dict[str, str] = {}

        def set_password(_service: str, username: str, password: str) -> None:
            store[username] = password

        def get_password(_service: str, username: str) -> str | None:
            return store.get(username)

        def delete_password(_service: str, username: str) -> None:
            store.pop(username, None)

        kr.set_password = set_password
        kr.get_password = get_password
        kr.delete_password = delete_password
        return kr

    def test_store_and_retrieve(self) -> None:
        mgr = CredentialManager(service_name="test")
        mock_kr = self._mock_keyring()
        with patch("chat.credentials._get_keyring", return_value=mock_kr):
            result = mgr.store_api_key("anthropic", "sk-test-123")
            assert result is True
            key = mgr.get_api_key("anthropic")
            assert key == "sk-test-123"

    def test_store_empty_provider_raises(self) -> None:
        mgr = CredentialManager()
        with pytest.raises(ValueError, match="non-empty"):
            mgr.store_api_key("", "key")

    def test_store_empty_key_raises(self) -> None:
        mgr = CredentialManager()
        with pytest.raises(ValueError, match="non-empty"):
            mgr.store_api_key("openai", "")

    def test_get_empty_provider_raises(self) -> None:
        mgr = CredentialManager()
        with pytest.raises(ValueError, match="non-empty"):
            mgr.get_api_key("")

    def test_delete_key(self) -> None:
        mgr = CredentialManager(service_name="test")
        mock_kr = self._mock_keyring()
        with patch("chat.credentials._get_keyring", return_value=mock_kr):
            mgr.store_api_key("openai", "sk-xyz")
            assert mgr.get_api_key("openai") == "sk-xyz"

            result = mgr.delete_api_key("openai")
            assert result is True
            assert mgr.get_api_key("openai") is None

    def test_delete_nonexistent_key(self) -> None:
        mgr = CredentialManager(service_name="test")
        mock_kr = self._mock_keyring()
        with patch("chat.credentials._get_keyring", return_value=mock_kr):
            result = mgr.delete_api_key("nonexistent")
            assert result is True  # delete is idempotent

    def test_get_nonexistent_key(self) -> None:
        mgr = CredentialManager(service_name="test")
        mock_kr = self._mock_keyring()
        with patch("chat.credentials._get_keyring", return_value=mock_kr):
            key = mgr.get_api_key("nonexistent")
            assert key is None

    def test_has_credentials_true(self) -> None:
        mgr = CredentialManager(service_name="test")
        mock_kr = self._mock_keyring()
        with patch("chat.credentials._get_keyring", return_value=mock_kr):
            mgr.store_api_key("anthropic", "sk-abc")
            assert mgr.has_credentials("anthropic") is True

    def test_has_credentials_false(self) -> None:
        mgr = CredentialManager(service_name="test")
        mock_kr = self._mock_keyring()
        with patch("chat.credentials._get_keyring", return_value=mock_kr):
            assert mgr.has_credentials("anthropic") is False

    def test_list_configured_providers(self) -> None:
        mgr = CredentialManager(service_name="test")
        mock_kr = self._mock_keyring()
        with patch("chat.credentials._get_keyring", return_value=mock_kr):
            mgr.store_api_key("openai", "sk-1")
            mgr.store_api_key("anthropic", "sk-2")
            providers = mgr.list_configured_providers()
            assert "anthropic" in providers
            assert "openai" in providers

    def test_provider_names_are_lowercased(self) -> None:
        mgr = CredentialManager(service_name="test")
        mock_kr = self._mock_keyring()
        with patch("chat.credentials._get_keyring", return_value=mock_kr):
            mgr.store_api_key("OpenAI", "sk-mixed-case")
            key = mgr.get_api_key("openai")
            assert key == "sk-mixed-case"


# ── Env var fallback tests ───────────────────────────────────────────


class TestCredentialManagerEnvFallback:
    def test_fallback_to_env_var(self) -> None:
        mgr = CredentialManager()
        with (
            patch("chat.credentials._get_keyring", return_value=None),
            patch.dict("os.environ", {"OPENAI_API_KEY": "sk-env-key"}),
        ):
            key = mgr.get_api_key("openai")
            assert key == "sk-env-key"

    def test_no_env_var_returns_none(self) -> None:
        mgr = CredentialManager()
        with (
            patch("chat.credentials._get_keyring", return_value=None),
            patch.dict("os.environ", {}, clear=True),
        ):
            key = mgr.get_api_key("openai")
            assert key is None

    def test_store_without_keyring_returns_false(self) -> None:
        mgr = CredentialManager()
        with patch("chat.credentials._get_keyring", return_value=None):
            result = mgr.store_api_key("openai", "sk-key")
            assert result is False

    def test_delete_without_keyring_returns_false(self) -> None:
        mgr = CredentialManager()
        with patch("chat.credentials._get_keyring", return_value=None):
            result = mgr.delete_api_key("openai")
            assert result is False

    def test_delete_empty_provider_raises(self) -> None:
        mgr = CredentialManager()
        with pytest.raises(ValueError, match="non-empty"):
            mgr.delete_api_key("")
