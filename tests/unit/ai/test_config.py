"""Tests for src.shared.python.ai.config (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from src.shared.python.ai.config import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_ANTHROPIC_TIMEOUT,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_TIMEOUT,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENAI_TIMEOUT,
    get_anthropic_api_key,
    get_anthropic_model,
    get_anthropic_timeout,
    get_gemini_api_key,
    get_gemini_model,
    get_ollama_host,
    get_ollama_model,
    get_ollama_timeout,
    get_openai_api_key,
    get_openai_model,
    get_openai_organization,
    get_openai_timeout,
)


class TestOllamaConfig:
    def test_default_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        host = get_ollama_host()
        assert host == DEFAULT_OLLAMA_HOST

    def test_default_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        model = get_ollama_model()
        assert model == DEFAULT_OLLAMA_MODEL

    def test_default_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OLLAMA_TIMEOUT", raising=False)
        timeout = get_ollama_timeout()
        assert timeout == pytest.approx(DEFAULT_OLLAMA_TIMEOUT)

    def test_timeout_is_float(self) -> None:
        timeout = get_ollama_timeout()
        assert isinstance(timeout, float)

    def test_env_var_overrides_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_HOST", "http://remotehost:11434")
        host = get_ollama_host()
        assert host == "http://remotehost:11434"

    def test_env_var_overrides_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_MODEL", "llama3.2:latest")
        model = get_ollama_model()
        assert model == "llama3.2:latest"


class TestOpenAIConfig:
    def test_default_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        model = get_openai_model()
        assert model == DEFAULT_OPENAI_MODEL

    def test_default_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_TIMEOUT", raising=False)
        timeout = get_openai_timeout()
        assert timeout == pytest.approx(DEFAULT_OPENAI_TIMEOUT)

    def test_api_key_not_required_returns_none_or_str(self) -> None:
        key = get_openai_api_key(required=False)
        assert key is None or isinstance(key, str)

    def test_organization_not_required_returns_none_or_str(self) -> None:
        org = get_openai_organization()
        assert org is None or isinstance(org, str)

    def test_env_var_overrides_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
        model = get_openai_model()
        assert model == "gpt-4o"


class TestAnthropicConfig:
    def test_default_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
        model = get_anthropic_model()
        assert model == DEFAULT_ANTHROPIC_MODEL

    def test_default_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_TIMEOUT", raising=False)
        timeout = get_anthropic_timeout()
        assert timeout == pytest.approx(DEFAULT_ANTHROPIC_TIMEOUT)

    def test_api_key_not_required_returns_none_or_str(self) -> None:
        key = get_anthropic_api_key(required=False)
        assert key is None or isinstance(key, str)

    def test_env_var_overrides_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-3-opus-20240229")
        model = get_anthropic_model()
        assert model == "claude-3-opus-20240229"


class TestGeminiConfig:
    def test_default_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GEMINI_MODEL", raising=False)
        model = get_gemini_model()
        assert model == DEFAULT_GEMINI_MODEL

    def test_api_key_not_required_returns_none_or_str(self) -> None:
        key = get_gemini_api_key(required=False)
        assert key is None or isinstance(key, str)

    def test_env_var_overrides_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_MODEL", "gemini-1.5-pro")
        model = get_gemini_model()
        assert model == "gemini-1.5-pro"
