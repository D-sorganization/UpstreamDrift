from __future__ import annotations

from src.shared.python.ai.gui.assistant_panel import _rust_ollama_endpoint_paths


def test_rust_ollama_paths_do_not_duplicate_v1_prefix() -> None:
    assert _rust_ollama_endpoint_paths("http://localhost:11434/v1") == (
        "/chat/completions",
        "/embeddings",
    )


def test_rust_ollama_paths_add_v1_for_plain_ollama_host() -> None:
    assert _rust_ollama_endpoint_paths("http://localhost:11434") == (
        "/v1/chat/completions",
        "/v1/embeddings",
    )


def test_rust_ollama_paths_ignore_trailing_slash() -> None:
    assert _rust_ollama_endpoint_paths("http://localhost:11434/v1/") == (
        "/chat/completions",
        "/embeddings",
    )
