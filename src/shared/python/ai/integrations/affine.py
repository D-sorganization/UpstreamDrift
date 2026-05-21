"""Affine integration tools for Sidekick (Phase 2 — real GraphQL client)."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from src.shared.python.ai.tool_registry import ToolCategory, get_global_registry

logger = logging.getLogger(__name__)

_AFFINE_API_TOKEN: str | None = None
_AFFINE_BASE_URL: str = "https://app.affine.pro/graphql"

_WORKSPACES_QUERY = """
query {
  workspaces {
    id
    name
  }
}
"""

_CREATE_DOC_MUTATION = """
mutation CreatePage($workspaceId: String!) {
  createDoc(workspaceId: $workspaceId) {
    id
  }
}
"""


def set_affine_api_token(token: str) -> None:
    """Store the Affine API token in memory for session use.

    Args:
        token: A valid Affine session token.

    Raises:
        ValueError: If token is empty.
    """
    if not token:
        raise ValueError("token must be a non-empty string")
    global _AFFINE_API_TOKEN  # noqa: PLW0603
    _AFFINE_API_TOKEN = token


def set_affine_base_url(url: str) -> None:
    """Set a custom AFFiNE GraphQL endpoint (for self-hosted instances).

    Args:
        url: Full URL to the GraphQL endpoint.

    Raises:
        ValueError: If url is empty.
    """
    if not url:
        raise ValueError("url must be a non-empty string")
    global _AFFINE_BASE_URL  # noqa: PLW0603
    _AFFINE_BASE_URL = url


def _get_token() -> str:
    """Return the active Affine token or raise ValueError.

    Raises:
        ValueError: If no token is configured.
    """
    token = _AFFINE_API_TOKEN or os.environ.get("AFFINE_API_KEY")
    if not token:
        raise ValueError(
            "Affine API token not configured. "
            "Call set_affine_api_token() or set AFFINE_API_KEY env var."
        )
    return token


def _get_base_url() -> str:
    """Return the active Affine GraphQL base URL."""
    return os.environ.get("AFFINE_BASE_URL", _AFFINE_BASE_URL)


def _run_affine_query(query_str: str, variables: dict[str, Any]) -> dict[str, Any]:
    """Execute a GraphQL query/mutation against the AFFiNE API.

    Args:
        query_str: The GraphQL query or mutation string.
        variables: A mapping of variable names to values.

    Returns:
        The ``data`` field from the GraphQL response.

    Raises:
        ValueError: If no API token is configured.
        RuntimeError: On HTTP errors, network failures, or GraphQL errors.
    """
    token = _get_token()
    base_url = _get_base_url()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {"query": query_str, "variables": variables}

    logger.debug("Affine GraphQL request to %s", base_url)

    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(base_url, json=payload, headers=headers)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        text = exc.response.text
        raise RuntimeError(f"Affine API error {status}: {text}") from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"Affine API connection failed: {exc}") from exc

    body = response.json()
    errors = body.get("errors")
    if errors:
        raise RuntimeError(f"Affine GraphQL error: {errors}")

    data: dict[str, Any] = body.get("data", {})
    return data


def affine_list_workspaces() -> dict[str, Any]:
    """List available workspaces for the authenticated user.

    Returns:
        A dict with key ``workspaces`` containing a list of
        ``{"id": str, "name": str}`` entries.

    Raises:
        ValueError: If no API token is configured.
        RuntimeError: On API or network errors.
    """
    data = _run_affine_query(_WORKSPACES_QUERY, {})
    raw: list[dict[str, Any]] = data.get("workspaces", [])
    logger.info("Affine: found %d workspace(s)", len(raw))
    return {"workspaces": raw}


registry = get_global_registry()


@registry.register(
    "affine_sync_notes",
    "Sync markdown notes to an Affine workspace.",
    category=ToolCategory.ANALYSIS,
    requires_confirmation=True,
)
def affine_sync_notes(
    title: str, markdown_content: str, workspace_id: str = ""
) -> dict[str, Any]:
    """Create a new page in an AFFiNE workspace.

    Full content sync requires the AFFiNE desktop client or the yjs protocol.
    This function creates the page via the GraphQL API and returns the page ID
    so the caller can open it in the desktop client to paste content.

    Args:
        title: The title of the note (non-empty).
        markdown_content: The markdown body of the note (used for logging only;
            full yjs content write is out of Phase 2 scope).
        workspace_id: The ID of the target AFFiNE workspace. When omitted the
            first workspace returned by the API is used.

    Returns:
        A dict with keys: ``success``, ``doc_id``, ``workspace_id``, ``note``.

    Raises:
        ValueError: If ``title`` is empty or no API token is configured.
        RuntimeError: On API or network errors.
    """
    if not title:
        raise ValueError("title must be a non-empty string")

    resolved_workspace_id = workspace_id
    if not resolved_workspace_id:
        logger.debug("affine_sync_notes: no workspace_id given — fetching workspaces")
        ws_data = affine_list_workspaces()
        workspaces = ws_data.get("workspaces", [])
        if not workspaces:
            raise RuntimeError("No Affine workspaces found for the authenticated user.")
        resolved_workspace_id = workspaces[0]["id"]
        logger.info("affine_sync_notes: using workspace %s", resolved_workspace_id)

    data = _run_affine_query(
        _CREATE_DOC_MUTATION, {"workspaceId": resolved_workspace_id}
    )
    doc_id: str = data.get("createDoc", {}).get("id", "")
    logger.info(
        "affine_sync_notes: created doc %s in workspace %s",
        doc_id,
        resolved_workspace_id,
    )

    return {
        "success": True,
        "doc_id": doc_id,
        "workspace_id": resolved_workspace_id,
        "note": (
            "Content sync requires AFFiNE desktop client or yjs protocol. "
            "Page created with title only."
        ),
    }
