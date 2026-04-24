"""Minimal headless entrypoint demonstrating AssistantSession usage.

Run this script directly to interact with the assistant from the command line:

.. code-block:: bash

    python3 -m src.shared.python.ai.headless_entrypoint

The session uses the Ollama adapter by default (no API key required).
Set ``PROVIDER``, ``API_KEY``, and ``MODEL`` environment variables to switch
providers:

.. code-block:: bash

    PROVIDER=anthropic API_KEY=sk-ant-... MODEL=claude-3-5-sonnet-20241022 \\
        python3 -m src.shared.python.ai.headless_entrypoint

Disabling the assistant
-----------------------
Downstream applications can disable the assistant entirely by checking an
environment variable or config flag before instantiating
:class:`~src.shared.python.ai.assistant_core.AssistantSession`.  Example:

.. code-block:: python

    import os
    from src.shared.python.ai.assistant_core import AssistantSession

    if os.getenv("ASSISTANT_DISABLED") == "1":
        # No-op path — assistant not loaded
        pass
    else:
        session = AssistantSession(provider="ollama")
"""

from __future__ import annotations

import asyncio
import os
import sys

from src.shared.python.ai.assistant_core import AssistantSession
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


async def run_interactive_session() -> None:
    """Run a simple interactive session, streaming responses to stdout.

    Reads user input line by line and prints streamed assistant chunks.
    Type ``quit`` or ``exit`` to end the session, ``reset`` to clear history.
    """
    provider = os.getenv("PROVIDER", "ollama")
    api_key = os.getenv("API_KEY", "")
    model = os.getenv("MODEL", "")

    adapter_kwargs: dict[str, str] = {}
    if api_key:
        adapter_kwargs["api_key"] = api_key
    if model:
        adapter_kwargs["model"] = model

    session = AssistantSession(provider=provider, adapter_kwargs=adapter_kwargs)

    logger.info("Headless assistant started (provider=%s)", provider)

    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            sys.stdout.write("\n")
            break

        if not text:
            continue
        if text.lower() in {"quit", "exit"}:
            break
        if text.lower() == "reset":
            session.reset()
            sys.stdout.write("[Session reset]\n")
            continue

        sys.stdout.write("Assistant: ")
        sys.stdout.flush()
        async for chunk in session.send_message(text):
            sys.stdout.write(chunk)
            sys.stdout.flush()
        sys.stdout.write("\n")


async def demo() -> None:
    """Demonstrate AssistantSession without interactive input.

    This function sends a single question and prints the streamed response,
    showing the minimal integration pattern for downstream consumers.
    """
    session = AssistantSession(provider="ollama", tool_registry=None)
    async for chunk in session.send_message("What engines are available?"):
        sys.stdout.write(chunk)
        sys.stdout.flush()
    sys.stdout.write("\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        asyncio.run(demo())
    else:
        asyncio.run(run_interactive_session())
