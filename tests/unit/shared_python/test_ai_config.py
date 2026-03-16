from __future__ import annotations

import os
from unittest.mock import patch

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
from src.shared.python.config.environment import EnvironmentError


def test_ollama_config_defaults():
    with patch.dict(os.environ, {}, clear=True):
        assert get_ollama_host() == DEFAULT_OLLAMA_HOST
        assert get_ollama_model() == DEFAULT_OLLAMA_MODEL
        assert get_ollama_timeout() == DEFAULT_OLLAMA_TIMEOUT


def test_ollama_config_env():
    env = {
        "OLLAMA_HOST": "http://my-host",
        "OLLAMA_MODEL": "my-model",
        "OLLAMA_TIMEOUT": "50.5",
    }
    with patch.dict(os.environ, env, clear=True):
        assert get_ollama_host() == "http://my-host"
        assert get_ollama_model() == "my-model"
        assert get_ollama_timeout() == 50.5


def test_openai_config_defaults():
    with patch.dict(os.environ, {}, clear=True):
        assert get_openai_api_key() is None
        assert get_openai_model() == DEFAULT_OPENAI_MODEL
        assert get_openai_timeout() == DEFAULT_OPENAI_TIMEOUT
        assert get_openai_organization() is None

        with pytest.raises(EnvironmentError):
            get_openai_api_key(required=True)


def test_openai_config_env():
    env = {
        "OPENAI_API_KEY": "sk-123",
        "OPENAI_MODEL": "gpt-custom",
        "OPENAI_TIMEOUT": "10",
        "OPENAI_ORGANIZATION": "org-456",
    }
    with patch.dict(os.environ, env, clear=True):
        assert get_openai_api_key(required=True) == "sk-123"
        assert get_openai_model() == "gpt-custom"
        assert get_openai_timeout() == 10.0
        assert get_openai_organization() == "org-456"


def test_anthropic_config_defaults():
    with patch.dict(os.environ, {}, clear=True):
        assert get_anthropic_api_key() is None
        assert get_anthropic_model() == DEFAULT_ANTHROPIC_MODEL
        assert get_anthropic_timeout() == DEFAULT_ANTHROPIC_TIMEOUT

        with pytest.raises(EnvironmentError):
            get_anthropic_api_key(required=True)


def test_anthropic_config_env():
    env = {
        "ANTHROPIC_API_KEY": "sk-ant-123",
        "ANTHROPIC_MODEL": "claude-custom",
        "ANTHROPIC_TIMEOUT": "15",
    }
    with patch.dict(os.environ, env, clear=True):
        assert get_anthropic_api_key(required=True) == "sk-ant-123"
        assert get_anthropic_model() == "claude-custom"
        assert get_anthropic_timeout() == 15.0


def test_gemini_config_defaults():
    with patch.dict(os.environ, {}, clear=True):
        assert get_gemini_api_key() is None
        assert get_gemini_model() == DEFAULT_GEMINI_MODEL

        with pytest.raises(EnvironmentError):
            get_gemini_api_key(required=True)


def test_gemini_config_env():
    env = {
        "GEMINI_API_KEY": "gx-123",
        "GEMINI_MODEL": "gemini-custom",
    }
    with patch.dict(os.environ, env, clear=True):
        assert get_gemini_api_key() == "gx-123"
        assert get_gemini_model() == "gemini-custom"
