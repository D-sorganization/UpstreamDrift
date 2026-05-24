"""Embeddable-tool contract foundation for the launcher.

This package defines the public contract that allows tools to declare
themselves embeddable inside a host launcher window (e.g., as a tab or a
dock widget) instead of always opening as a standalone top-level window.

Public surface:

- :class:`EmbedCapabilities` — frozen dataclass describing how a tool wants
  to be embedded.
- :class:`EmbeddableTool` — runtime-checkable :class:`typing.Protocol` that
  tools implement to participate in embedding.
- :func:`register_embeddable_tool` / :func:`get_embeddable_tool` /
  :func:`is_embeddable` / :func:`unregister_embeddable_tool` — registry API.
- :data:`EMBEDDABLE_TOOL_REGISTRY` — the underlying registry mapping
  ``tool_id`` to :class:`EmbeddableTool`.

This module intentionally avoids importing PyQt6 at import time; PyQt6 is
optional fleet-wide. Widget types are typed as :class:`typing.Any` in the
protocol so that consumers do not need to import a Qt binding to satisfy
the contract.
"""

from .contract import EmbedCapabilities, EmbeddableTool
from .registry import (
    EMBEDDABLE_TOOL_REGISTRY,
    get_embeddable_tool,
    is_embeddable,
    register_embeddable_tool,
    unregister_embeddable_tool,
)

# Public API version (SemVer MAJOR.MINOR.PATCH).
#
# Bump rules (per issue #5917, ADR-0013):
# - MAJOR: breaking change to the ``EmbeddableTool`` protocol
#   (signatures, removed methods), to ``EmbedCapabilities`` fields,
#   or to registry semantics.
# - MINOR: backwards-compatible additions (new optional protocol
#   methods with default implementations, new capability fields with
#   defaults, new registry helpers).
# - PATCH: bug fixes that do not change the public surface.
__version__ = "1.0.0"

# Contract version exposed to embedded tools. Hosts may refuse tools
# declaring a higher major.
SCHEMA_VERSION = "1.0.0"

__all__ = [
    "EMBEDDABLE_TOOL_REGISTRY",
    "EmbedCapabilities",
    "EmbeddableTool",
    "SCHEMA_VERSION",
    "__version__",
    "get_embeddable_tool",
    "is_embeddable",
    "register_embeddable_tool",
    "unregister_embeddable_tool",
]
