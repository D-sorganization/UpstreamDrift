"""BitNet local adapter using direct subprocess execution.

This adapter allows the shared Tools AI chat interface to run 1.58b models locally
without requiring an external FastAPI server. It manages a llama-cli subprocess
directly.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Iterator

from src.shared.python.ai.adapters.base import BaseAgentAdapter, ToolDeclaration
from src.shared.python.ai.exceptions import (
    AIConnectionError,
    AIProviderError,
    AITimeoutError,
)
from src.shared.python.ai.types import (
    AgentChunk,
    AgentResponse,
    ConversationContext,
    ProviderCapabilities,
    ProviderCapability,
)
from src.shared.python.contracts import precondition
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

# Default wall-clock budget for a single llama-cli invocation. Local
# inference of 512 tokens on CPU can be slow, so this is generous; a hung
# or stalled process is killed once the deadline elapses (issue #3175).
DEFAULT_BITNET_TIMEOUT: float = 300.0


class BitnetAdapter(BaseAgentAdapter):
    """Adapter for running local BitNet models via direct subprocess.

    This adapter launches and manages a llama.cpp / llama-cli process directly,
    providing a seamless local LLM experience within the shared chat interface.
    """

    def __init__(
        self,
        model: str | None = None,
        bitnet_root: str | None = None,
        timeout: float | None = None,
    ) -> None:
        """Initialize the BitNet adapter.

        Args:
            model: Name or path to the model file to run.
            bitnet_root: Path to the root of the bitnet installation.
            timeout: Wall-clock budget [s] for a single ``llama-cli``
                invocation (both ``send_message`` and ``stream_response``).
                Defaults to :data:`DEFAULT_BITNET_TIMEOUT`. Must be > 0.
        """
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be a positive number of seconds")
        self.timeout: float = (
            float(timeout) if timeout is not None else DEFAULT_BITNET_TIMEOUT
        )
        self.model = model or "bitnet-1.58b-q4_0.gguf"
        self.bitnet_root = bitnet_root or os.environ.get("BITNET_ROOT", "")
        self.llama_cli = (
            os.path.join(self.bitnet_root, "llama-cli")
            if self.bitnet_root
            else "llama-cli"
        )
        self._process: subprocess.Popen | None = None
        self._capabilities = ProviderCapabilities(
            supported=frozenset(
                {
                    ProviderCapability.STREAMING,
                    ProviderCapability.SYSTEM_MESSAGE,
                }
            ),
            max_tokens=2048,
            model_name=self.model,
            provider_name="bitnet",
        )

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    # ------------------------------------------------------------------ #
    # Tools issue #2871: provider catalogue + reasoning capabilities
    # ------------------------------------------------------------------ #

    _STATIC_MODELS: tuple[str, ...] = (
        "bitnet-1.58b-q4_0.gguf",
        "bitnet-2b-q4_0.gguf",
        "bitnet-3b-q4_0.gguf",
    )

    def list_models(self) -> list[str]:
        """Return BitNet model catalogue; configured model is always included."""
        models = list(self._STATIC_MODELS)
        if self.model and self.model not in models:
            models.insert(0, self.model)
        return models

    def thinking_capabilities(self):  # type: ignore[no-untyped-def]
        """BitNet local models do not expose reasoning budgets."""
        from shared.python.chat_contracts.models import make_none_only_capabilities

        return make_none_only_capabilities(provider="bitnet")

    def _handle_error(self, error: Exception) -> AgentResponse:
        """Classify BitNet subprocess errors into the AIProviderError hierarchy."""
        if isinstance(error, FileNotFoundError):
            raise AIConnectionError(
                f"Failed to run BitNet: llama-cli not found at {self.llama_cli}",
                provider="bitnet",
            ) from error
        if isinstance(error, subprocess.TimeoutExpired):
            raise AITimeoutError(
                f"BitNet timed out after {self.timeout}s",
                provider="bitnet",
                timeout=self.timeout,
            ) from error
        if isinstance(error, subprocess.CalledProcessError):
            raise AIProviderError(
                f"BitNet process failed (code {error.returncode}): {error.stderr}",
                provider="bitnet",
            ) from error
        raise self._classify_error(
            error, provider="bitnet", timeout=self.timeout
        ) from error

    def validate_connection(self) -> tuple[bool, str]:
        """Validate that the llama-cli executable is available."""
        try:
            # Just test if the executable exists and can be run
            result = subprocess.run(
                [self.llama_cli, "--help"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5.0,
            )
            if result.returncode == 0 or "usage" in result.stdout.lower():
                return True, f"Found {self.llama_cli}"
            return False, f"Executable failed: {result.stderr}"
        except Exception as e:
            return False, f"Failed to execute {self.llama_cli}: {e}"

    def _format_prompt(self, context: ConversationContext, message: str) -> str:
        """Format the conversation into a plain text prompt."""
        prompt = ""
        if context and context.messages:
            for msg in context.messages:
                if msg.role == "system":
                    prompt += f"System: {msg.content}\n"
                elif msg.role == "user":
                    prompt += f"User: {msg.content}\n"
                elif msg.role == "assistant":
                    prompt += f"Assistant: {msg.content}\n"
        prompt += f"User: {message}\nAssistant:"
        return prompt

    @precondition(
        lambda message: bool(message.strip()), "message must not be empty or blank"
    )
    def send_message(
        self,
        message: str,
        context: ConversationContext,
        tools: list[ToolDeclaration],
    ) -> AgentResponse:
        """Send a message synchronously."""
        prompt = self._format_prompt(context, message)

        try:
            cmd = [
                self.llama_cli,
                "-m",
                self.model,
                "-p",
                prompt,
                "-n",
                "512",
                "--log-disable",
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=self.timeout,
            )

            output = result.stdout.replace(prompt, "").strip()

            # BitNet does not report token counts; emit canonical zeros so
            # callers always see the same keys (issue #2763).
            return AgentResponse(
                content=output,
                usage=self._normalize_token_counts({}),
                metadata={"stdout": result.stdout},
            )
        except Exception as e:
            logger.error("Failed to run BitNet: %s", e)
            return self._handle_error(e)

    def stream_response(
        self,
        message: str,
        context: ConversationContext,
        tools: list[ToolDeclaration],
    ) -> Iterator[AgentChunk]:
        """Stream the response using subprocess."""
        prompt = self._format_prompt(context, message)

        cmd = [
            self.llama_cli,
            "-m",
            self.model,
            "-p",
            prompt,
            "-n",
            "512",
            "--log-disable",
        ]

        process: subprocess.Popen | None = None
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
            )

            if not process.stdout:
                raise AIProviderError(
                    "BitNet process stdout is unavailable", provider="bitnet"
                )

            # Drain stdout on a daemon reader thread so the main loop can
            # enforce a wall-clock deadline. A hung llama-cli that never
            # emits EOF would otherwise block ``for line in stdout`` forever
            # (issue #3175). The sentinel ``None`` marks normal EOF.
            import queue as _queue
            import threading as _threading

            line_queue: _queue.Queue[str | None] = _queue.Queue()
            stdout = process.stdout

            def _pump() -> None:
                try:
                    for raw_line in stdout:
                        line_queue.put(raw_line)
                finally:
                    line_queue.put(None)

            reader = _threading.Thread(target=_pump, daemon=True)
            reader.start()

            deadline = time.monotonic() + self.timeout
            index = 0
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(cmd, self.timeout)
                try:
                    line = line_queue.get(timeout=remaining)
                except _queue.Empty as exc:
                    raise subprocess.TimeoutExpired(cmd, self.timeout) from exc
                if line is None:  # EOF sentinel
                    break
                yield AgentChunk(content=line, is_final=False, index=index)
                index += 1

            process.wait(timeout=max(0.0, deadline - time.monotonic()))
            yield AgentChunk(content="", is_final=True)

        except subprocess.TimeoutExpired:
            logger.error("BitNet stream timed out after %ss", self.timeout)
            self._terminate(process)
            yield AgentChunk(
                content=f"\n[Error: BitNet stream timed out after {self.timeout}s]",
                is_final=True,
            )
        except Exception as e:
            logger.error("Failed to stream BitNet: %s", e)
            self._terminate(process)
            yield AgentChunk(content=f"\n[Error: {e}]", is_final=True)

    @staticmethod
    def _terminate(process: subprocess.Popen | None) -> None:
        """Kill and reap a (possibly hung) child process; never raises."""
        if process is None:
            return
        if process.poll() is not None:  # already exited
            return
        try:
            process.kill()
            process.wait(timeout=5.0)
        except Exception:  # noqa: BLE001 - best-effort reaping
            logger.warning("Failed to fully reap BitNet child process", exc_info=True)
