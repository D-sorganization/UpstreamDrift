"""Linear integration tools for Sidekick."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from src.shared.python.ai.tool_registry import ToolCategory, get_global_registry

logger = logging.getLogger(__name__)

_LINEAR_API_ENDPOINT = "https://api.linear.app/graphql"
_REQUEST_TIMEOUT = 30

_LINEAR_API_TOKEN: str | None = None


def set_linear_api_token(token: str) -> None:
    """Store the Linear API token in memory for session use."""
    global _LINEAR_API_TOKEN  # noqa: PLW0603
    _LINEAR_API_TOKEN = token


def _get_token() -> str:
    """Return the active Linear API token.

    Checks module-level token first, then LINEAR_API_KEY environment variable.

    Returns:
        The API token string.

    Raises:
        ValueError: When no token is configured.
    """
    token = _LINEAR_API_TOKEN or os.environ.get("LINEAR_API_KEY")
    if not token:
        raise ValueError(
            "Linear API token not configured. Call set_linear_api_token() or set"
            " LINEAR_API_KEY env var."
        )
    return token


def _run_linear_query(query_str: str, variables: dict[str, Any]) -> dict[str, Any]:
    """Execute a GraphQL query against the Linear API.

    Args:
        query_str: The GraphQL query or mutation string.
        variables: Variables to pass with the query.

    Returns:
        The parsed JSON response body.

    Raises:
        ValueError: When no API token is configured.
        RuntimeError: On HTTP error responses or network failures.
    """
    token = _get_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {"query": query_str, "variables": variables}

    try:
        with httpx.Client(timeout=_REQUEST_TIMEOUT) as client:
            response = client.post(
                _LINEAR_API_ENDPOINT,
                headers=headers,
                json=payload,
            )
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Linear API connection failed: {exc}") from exc

    if response.status_code != 200:
        raise RuntimeError(f"Linear API error {response.status_code}: {response.text}")

    return response.json()  # type: ignore[no-any-return]


registry = get_global_registry()

_ISSUES_QUERY = """
query Issues($filter: IssueFilter) {
  issues(filter: $filter) {
    nodes {
      id
      title
      description
      state { name }
      assignee { name }
      priority
      url
    }
  }
}
"""

_CREATE_ISSUE_MUTATION = """
mutation CreateIssue($input: IssueCreateInput!) {
  issueCreate(input: $input) {
    success
    issue {
      id
      title
      url
    }
  }
}
"""


@registry.register(
    "linear_query_issues",
    "Query Linear issues based on a search term or status.",
    category=ToolCategory.ANALYSIS,
)
def linear_query_issues(query: str, status: str = "open") -> dict[str, Any]:
    """Query Linear for issues matching a search term.

    Precondition: query must be a non-empty string.

    Args:
        query: The search term to find relevant issues.
        status: The status of issues to return (e.g. 'open', 'done').

    Returns:
        A dict with key 'issues' containing a list of issue dicts, each with
        keys: id, title, description, state, assignee, priority, url.

    Raises:
        ValueError: When query is empty or no API token is configured.
        RuntimeError: On HTTP error or network failure.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    issue_filter: dict[str, Any] = {"title": {"containsIgnoreCase": query}}
    if status and status.lower() != "open":
        issue_filter["state"] = {"name": {"eq": status}}

    variables = {"filter": issue_filter}
    logger.debug("Querying Linear issues with filter: %s", issue_filter)

    data = _run_linear_query(_ISSUES_QUERY, variables)
    nodes = data.get("data", {}).get("issues", {}).get("nodes", [])

    issues = []
    for node in nodes:
        issues.append(
            {
                "id": node.get("id"),
                "title": node.get("title"),
                "description": node.get("description"),
                "state": (
                    node.get("state", {}).get("name") if node.get("state") else None
                ),
                "assignee": (
                    node.get("assignee", {}).get("name")
                    if node.get("assignee")
                    else None
                ),
                "priority": node.get("priority"),
                "url": node.get("url"),
            }
        )

    logger.info("Linear query returned %d issues", len(issues))
    return {"issues": issues}


@registry.register(
    "linear_create_issue",
    "Create a new issue in Linear.",
    category=ToolCategory.ANALYSIS,
    requires_confirmation=True,
)
def linear_create_issue(
    title: str, description: str, team_id: str = ""
) -> dict[str, Any]:
    """Create a new Linear issue.

    Precondition: title must be a non-empty string.

    Args:
        title: The title of the new issue.
        description: Detailed description in Markdown.
        team_id: The team ID to create the issue under.

    Returns:
        A dict with 'success' (bool) and 'issue' containing id, title, url.

    Raises:
        ValueError: When title is empty or no API token is configured.
        RuntimeError: On HTTP error or network failure.
    """
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must be a non-empty string")

    issue_input: dict[str, Any] = {"title": title, "description": description}
    if team_id:
        issue_input["teamId"] = team_id

    variables = {"input": issue_input}
    logger.debug("Creating Linear issue: %s", title)

    data = _run_linear_query(_CREATE_ISSUE_MUTATION, variables)
    result = data.get("data", {}).get("issueCreate", {})

    issue_data = result.get("issue") or {}
    logger.info("Linear issue created: %s", issue_data.get("id"))
    return {
        "success": result.get("success", False),
        "issue": {
            "id": issue_data.get("id"),
            "title": issue_data.get("title"),
            "url": issue_data.get("url"),
        },
    }
