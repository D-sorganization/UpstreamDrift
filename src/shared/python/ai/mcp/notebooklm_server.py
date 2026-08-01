"""NotebookLM MCP shim.

There is no first-party Google NotebookLM MCP server at the time of writing
(2026-05) — the official NotebookLM API is still gated. This module therefore
implements a small **Python shim** that respects the MCP wire format and can
be invoked as ``python -m src.shared.python.ai.mcp.notebooklm_server``.

Phase 1 (PR #2884) advertised two tools and one resource family:

- ``search_notebook(query: str, notebook_id: str) -> {hits: [...]}``
- ``summarize_notebook(notebook_id: str) -> {summary: str}``
- Resource ``notebook://{id}`` (read-only metadata).

Phase 2 (Tools #2900) expands the tool surface so the assistant can drive
the product rather than only read it:

- ``list_notebooks() -> {notebooks: [{id, title, modified_at}]}``
- ``create_notebook(title, sources) -> {id, url}``
- ``add_source_to_notebook(notebook_id, source_url_or_path) -> {source_id}``
- ``generate_audio_overview(notebook_id, voice="default")
  -> {audio_url, duration_seconds}`` (requires user confirmation)
- ``follow_citation(notebook_id, citation_id)
  -> {source_id, snippet, source_url}``
- ``attach_to_chat(notebook_id) -> {context_size_tokens}``
  (requires user confirmation)

Resources extended:

- ``notebook://{id}`` metadata now carries ``sources: [{id, title, type, url}]``.
- ``notebook://{id}/source/{source_id}`` reads the text body of a single
  source as an MCP resource.

Dependency choice:
    We deliberately avoid adding a heavy upstream dependency (e.g.
    ``notebooklm-python``, which is unmaintained as of 2026-05). The shim
    delegates the actual NotebookLM API calls to a pluggable
    ``NotebookBackend`` Protocol. The default backend returns deterministic
    stub data so the surface is testable and useful for local development
    without credentials.

The shim is intentionally small and synchronous on its inner backend;
``main()`` runs the stdin/stdout JSON-RPC loop asynchronously.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sys
from typing import Any, Protocol

_LOG = logging.getLogger(__name__)

PROTOCOL_VERSION = "2024-11-05"


class NotebookBackend(Protocol):
    """Pluggable backend for the NotebookLM shim.

    Phase 1 methods (Tools #2884):
        search, summarize, metadata

    Phase 2 methods (Tools #2900):
        list_notebooks, create_notebook, add_source,
        generate_audio_overview, follow_citation, attach_to_chat,
        read_source
    """

    # --- Phase 1 ---
    def search(self, notebook_id: str, query: str) -> list[dict[str, Any]]: ...
    def summarize(self, notebook_id: str) -> str: ...
    def metadata(self, notebook_id: str) -> dict[str, Any]: ...

    # --- Phase 2 ---
    def list_notebooks(self) -> list[dict[str, Any]]: ...
    def create_notebook(self, title: str, sources: list[str]) -> dict[str, Any]: ...
    def add_source(
        self, notebook_id: str, source_url_or_path: str
    ) -> dict[str, Any]: ...
    def generate_audio_overview(
        self, notebook_id: str, voice: str
    ) -> dict[str, Any]: ...
    def follow_citation(self, notebook_id: str, citation_id: str) -> dict[str, Any]: ...
    def attach_to_chat(self, notebook_id: str) -> dict[str, Any]: ...
    def read_source(self, notebook_id: str, source_id: str) -> str: ...


class StubNotebookBackend:
    """Deterministic in-memory backend used when no real API is configured.

    All methods are pure functions of their inputs — repeated calls with the
    same arguments return byte-identical dicts. This is essential for the
    Phase 2 test suite, which asserts determinism explicitly.
    """

    # ------------------------------------------------------------------ Phase 1
    def search(self, notebook_id: str, query: str) -> list[dict[str, Any]]:
        return [
            {
                "notebook_id": notebook_id,
                "snippet": f"stub hit for {query!r}",
                "score": 1.0,
            }
        ]

    def summarize(self, notebook_id: str) -> str:
        return f"Stub summary for notebook {notebook_id}."

    def metadata(self, notebook_id: str) -> dict[str, Any]:
        return {
            "id": notebook_id,
            "title": f"Notebook {notebook_id}",
            "sources": [
                {
                    "id": f"src-{notebook_id}-1",
                    "title": "Stub source A",
                    "type": "url",
                    "url": f"notebook://{notebook_id}/source/src-{notebook_id}-1",
                },
            ],
        }

    # ------------------------------------------------------------------ Phase 2
    def list_notebooks(self) -> list[dict[str, Any]]:
        _stub_mtime = "2026-05-16T00:00:00Z"
        return [
            {"id": "nb-1", "title": "Stub Notebook 1", "modified_at": _stub_mtime},
            {"id": "nb-2", "title": "Stub Notebook 2", "modified_at": _stub_mtime},
        ]

    def create_notebook(self, title: str, sources: list[str]) -> dict[str, Any]:
        # Deterministic ID derived from title + sources.
        digest = hashlib.sha256(
            (title + "|" + "|".join(sources)).encode("utf-8")
        ).hexdigest()[:12]
        nb_id = f"nb-{digest}"
        return {"id": nb_id, "url": f"notebook://{nb_id}"}

    def add_source(self, notebook_id: str, source_url_or_path: str) -> dict[str, Any]:
        digest = hashlib.sha256(
            (notebook_id + "|" + source_url_or_path).encode("utf-8")
        ).hexdigest()[:12]
        return {"source_id": f"src-{digest}"}

    def generate_audio_overview(self, notebook_id: str, voice: str) -> dict[str, Any]:
        return {
            "audio_url": f"https://stub.notebooklm/audio/{notebook_id}/{voice}.mp3",
            "duration_seconds": 180,
        }

    def follow_citation(self, notebook_id: str, citation_id: str) -> dict[str, Any]:
        return {
            "source_id": f"src-{citation_id}",
            "snippet": f"stub snippet for citation {citation_id} in {notebook_id}",
            "source_url": f"notebook://{notebook_id}/source/src-{citation_id}",
        }

    def attach_to_chat(self, notebook_id: str) -> dict[str, Any]:
        # Deterministic token count based on notebook id length, in a
        # plausible range for a small notebook.
        return {"context_size_tokens": 1024 + len(notebook_id)}

    def read_source(self, notebook_id: str, source_id: str) -> str:
        return f"Stub source body for {source_id} in notebook {notebook_id}."


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

_PHASE1_TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_notebook",
        "description": "Search within a NotebookLM notebook for a query string.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "notebook_id": {"type": "string"},
                "query": {"type": "string"},
            },
            "required": ["notebook_id", "query"],
        },
    },
    {
        "name": "summarize_notebook",
        "description": "Summarize the contents of a NotebookLM notebook.",
        "inputSchema": {
            "type": "object",
            "properties": {"notebook_id": {"type": "string"}},
            "required": ["notebook_id"],
        },
    },
]

_PHASE2_TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_notebooks",
        "description": "List notebooks accessible to the authenticated user.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "create_notebook",
        "description": (
            "Create a new NotebookLM notebook with an optional list of initial "
            "source URLs or file paths."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "sources": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["title"],
        },
    },
    {
        "name": "add_source_to_notebook",
        "description": "Attach a new source (URL or allowed local path) to a notebook.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "notebook_id": {"type": "string"},
                "source_url_or_path": {"type": "string"},
            },
            "required": ["notebook_id", "source_url_or_path"],
        },
    },
    {
        "name": "generate_audio_overview",
        "description": (
            "Generate a NotebookLM Audio Overview (podcast-style summary) for a "
            "notebook. Returns the audio URL and its duration. User confirmation "
            "is required before invocation."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "notebook_id": {"type": "string"},
                "voice": {"type": "string", "default": "default"},
            },
            "required": ["notebook_id"],
        },
        "metadata": {"requires_confirmation": True},
    },
    {
        "name": "follow_citation",
        "description": (
            "Resolve a citation ID into its source document, returning the "
            "snippet, source ID, and source URL."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "notebook_id": {"type": "string"},
                "citation_id": {"type": "string"},
            },
            "required": ["notebook_id", "citation_id"],
        },
    },
    {
        "name": "attach_to_chat",
        "description": (
            "Attach a notebook as a context source for the active chat session. "
            "Subsequent chat turns receive the notebook's content as RAG context. "
            "User confirmation is required before invocation."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"notebook_id": {"type": "string"}},
            "required": ["notebook_id"],
        },
        "metadata": {"requires_confirmation": True},
    },
]


def _extend_tools_list(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten one or more tool groups into a single ``tools/list`` payload.

    Keeping this helper means new phases never need to mutate Phase 1's list
    in place — they just register another group.
    """
    merged: list[dict[str, Any]] = []
    for group in groups:
        merged.extend(group)
    return merged


_TOOL_DEFINITIONS: list[dict[str, Any]] = _extend_tools_list(
    _PHASE1_TOOLS, _PHASE2_TOOLS
)


# ---------------------------------------------------------------------------
# Resource helpers
# ---------------------------------------------------------------------------


def _resource_for(notebook_id: str) -> dict[str, Any]:
    return {
        "uri": f"notebook://{notebook_id}",
        "name": f"Notebook {notebook_id}",
        "mimeType": "application/json",
    }


def _parse_notebook_uri(uri: str) -> tuple[str, str | None]:
    """Parse ``notebook://{id}`` or ``notebook://{id}/source/{source_id}``.

    Returns ``(notebook_id, source_id_or_None)``. Raises ``ValueError`` for
    any other URI shape.
    """
    if not uri.startswith("notebook://"):
        raise ValueError(f"unsupported resource URI: {uri!r}")
    body = uri[len("notebook://") :]
    if not body:
        raise ValueError(f"notebook URI missing notebook id: {uri!r}")
    if "/source/" in body:
        notebook_id, _, source_id = body.partition("/source/")
        if not notebook_id or not source_id:
            raise ValueError(f"malformed source URI: {uri!r}")
        return notebook_id, source_id
    # Reject deeper paths we don't understand.
    if "/" in body:
        raise ValueError(f"unsupported resource URI: {uri!r}")
    return body, None


# ---------------------------------------------------------------------------
# JSON-RPC dispatch
# ---------------------------------------------------------------------------


def handle_request(request: dict[str, Any], backend: NotebookBackend) -> dict[str, Any]:
    """Dispatch a single JSON-RPC request and return the response dict."""
    method = request.get("method")
    params = request.get("params") or {}
    request_id = request.get("id")
    try:
        result = _dispatch(method, params, backend)
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    except Exception as exc:  # noqa: BLE001 — JSON-RPC error envelope
        _LOG.exception("notebooklm shim: %s failed", method)
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32000, "message": str(exc)},
        }


def _dispatch(
    method: str | None, params: dict[str, Any], backend: NotebookBackend
) -> dict[str, Any]:
    if method == "initialize":
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}, "resources": {}},
            "serverInfo": {"name": "notebooklm-shim", "version": "0.2.0"},
        }
    if method == "tools/list":
        return {"tools": _TOOL_DEFINITIONS}
    if method == "resources/list":
        # Default stub advertises a single placeholder resource.
        return {"resources": [_resource_for("default")]}
    if method == "resources/read":
        return _handle_resource_read(params, backend)
    if method == "tools/call":
        return _handle_tool_call(params, backend)
    raise ValueError(f"unknown method: {method!r}")


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------


def _require_nonempty_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} is required")
    return value


def _validate_source_path(source: str) -> None:
    """Reject obvious path traversal attempts.

    The real backend will perform a proper allowed-roots check; here we just
    catch the easy mistakes (``..`` segments) so the stub surface is safe.
    """
    if ".." in source.replace("\\", "/").split("/"):
        raise ValueError("rejecting path traversal in source path")


def _text_result(payload: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(payload)}]}


def _handle_tool_call(
    params: dict[str, Any], backend: NotebookBackend
) -> dict[str, Any]:
    name = params.get("name")
    arguments = params.get("arguments") or {}
    handler = _TOOL_HANDLERS.get(str(name))
    if handler is None:
        raise ValueError(f"unknown tool: {name!r}")
    return handler(arguments, backend)


# --- individual tool handlers ----------------------------------------------


def _tool_search_notebook(
    arguments: dict[str, Any], backend: NotebookBackend
) -> dict[str, Any]:
    notebook_id = _require_nonempty_str(arguments.get("notebook_id"), "notebook_id")
    query = _require_nonempty_str(arguments.get("query"), "query")
    hits = backend.search(notebook_id, query)
    return _text_result({"hits": hits})


def _tool_summarize_notebook(
    arguments: dict[str, Any], backend: NotebookBackend
) -> dict[str, Any]:
    notebook_id = _require_nonempty_str(arguments.get("notebook_id"), "notebook_id")
    summary = backend.summarize(notebook_id)
    return {"content": [{"type": "text", "text": summary}]}


def _tool_list_notebooks(
    arguments: dict[str, Any], backend: NotebookBackend
) -> dict[str, Any]:
    del arguments  # no inputs
    return _text_result({"notebooks": backend.list_notebooks()})


def _tool_create_notebook(
    arguments: dict[str, Any], backend: NotebookBackend
) -> dict[str, Any]:
    title = _require_nonempty_str(arguments.get("title"), "title")
    sources = arguments.get("sources", [])
    if not isinstance(sources, list) or not all(isinstance(s, str) for s in sources):
        raise ValueError("sources must be a list of strings")
    return _text_result(backend.create_notebook(title, sources))


def _tool_add_source_to_notebook(
    arguments: dict[str, Any], backend: NotebookBackend
) -> dict[str, Any]:
    notebook_id = _require_nonempty_str(arguments.get("notebook_id"), "notebook_id")
    source = _require_nonempty_str(
        arguments.get("source_url_or_path"), "source_url_or_path"
    )
    _validate_source_path(source)
    return _text_result(backend.add_source(notebook_id, source))


def _tool_generate_audio_overview(
    arguments: dict[str, Any], backend: NotebookBackend
) -> dict[str, Any]:
    notebook_id = _require_nonempty_str(arguments.get("notebook_id"), "notebook_id")
    voice = arguments.get("voice", "default")
    if not isinstance(voice, str) or not voice:
        voice = "default"
    return _text_result(backend.generate_audio_overview(notebook_id, voice))


def _tool_follow_citation(
    arguments: dict[str, Any], backend: NotebookBackend
) -> dict[str, Any]:
    notebook_id = _require_nonempty_str(arguments.get("notebook_id"), "notebook_id")
    citation_id = _require_nonempty_str(arguments.get("citation_id"), "citation_id")
    return _text_result(backend.follow_citation(notebook_id, citation_id))


def _tool_attach_to_chat(
    arguments: dict[str, Any], backend: NotebookBackend
) -> dict[str, Any]:
    notebook_id = _require_nonempty_str(arguments.get("notebook_id"), "notebook_id")
    return _text_result(backend.attach_to_chat(notebook_id))


_TOOL_HANDLERS: dict[str, Any] = {
    # value type: Callable[[dict, NotebookBackend], dict]
    "search_notebook": _tool_search_notebook,
    "summarize_notebook": _tool_summarize_notebook,
    "list_notebooks": _tool_list_notebooks,
    "create_notebook": _tool_create_notebook,
    "add_source_to_notebook": _tool_add_source_to_notebook,
    "generate_audio_overview": _tool_generate_audio_overview,
    "follow_citation": _tool_follow_citation,
    "attach_to_chat": _tool_attach_to_chat,
}


# ---------------------------------------------------------------------------
# Resource dispatch
# ---------------------------------------------------------------------------


def _handle_resource_read(
    params: dict[str, Any], backend: NotebookBackend
) -> dict[str, Any]:
    uri = params.get("uri")
    if not isinstance(uri, str) or not uri:
        raise ValueError("uri is required")
    notebook_id, source_id = _parse_notebook_uri(uri)
    if source_id is None:
        body = json.dumps(backend.metadata(notebook_id))
        mime = "application/json"
    else:
        body = backend.read_source(notebook_id, source_id)
        mime = "text/plain"
    return {
        "contents": [
            {"uri": uri, "mimeType": mime, "text": body},
        ]
    }


# ---------------------------------------------------------------------------
# Async stdio loop
# ---------------------------------------------------------------------------


async def _run_loop(
    backend: NotebookBackend,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    while True:
        line = await reader.readline()
        if not line:
            return
        try:
            request = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            _LOG.warning("notebooklm shim: malformed JSON-RPC frame; ignoring")
            continue
        response = handle_request(request, backend)
        writer.write((json.dumps(response) + "\n").encode("utf-8"))
        await writer.drain()


def main() -> None:  # pragma: no cover — integration entrypoint
    """Run the shim's stdin/stdout JSON-RPC loop."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    backend: NotebookBackend = StubNotebookBackend()

    async def _amain() -> None:
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        await loop.connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(reader), sys.stdin
        )
        writer_transport, writer_protocol = await loop.connect_write_pipe(
            asyncio.streams.FlowControlMixin, sys.stdout
        )
        writer = asyncio.StreamWriter(writer_transport, writer_protocol, None, loop)
        await _run_loop(backend, reader, writer)

    asyncio.run(_amain())


if __name__ == "__main__":  # pragma: no cover
    main()
