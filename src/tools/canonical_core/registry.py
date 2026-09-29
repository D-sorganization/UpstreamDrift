"""Canonical-core tool descriptors shared by PyQt6 and React shells."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.shared.python.core.contracts import require

ShellSurface = Literal["pyqt6", "react"]


@dataclass(frozen=True, slots=True)
class CanonicalCoreTool:
    """Metadata for one canonical-core tool exposed through the app shell."""

    tool_id: str
    name: str
    description: str
    mode: Literal["estimation", "comparison"]
    capabilities: tuple[str, ...]
    web_route: str
    order: int
    category: str = "biomechanics"
    default_launch: str = "tab"
    shell_surfaces: tuple[ShellSurface, ...] = ("pyqt6", "react")
    pyqt_adapter: str = "src.tools.canonical_core._embed_adapter"

    def __post_init__(self) -> None:
        require(
            isinstance(self.tool_id, str) and bool(self.tool_id.strip()),
            "canonical-core tool_id must be non-empty",
            self.tool_id,
        )
        require(
            self.category == "biomechanics",
            "canonical-core tools must stay in the biomechanics category",
            self.category,
        )
        require(
            self.shell_surfaces == ("pyqt6", "react"),
            "canonical-core tools must expose both PyQt6 and React shells",
            self.shell_surfaces,
        )
        require(
            self.web_route.startswith("/tools/canonical-core/"),
            "canonical-core web_route must live under /tools/canonical-core",
            self.web_route,
        )


_CANONICAL_CORE_TOOLS: tuple[CanonicalCoreTool, ...] = (
    CanonicalCoreTool(
        tool_id="canonical_core_estimation",
        name="Canonical-Core Estimation",
        description=(
            "Workspace entry point for CC-19 estimation services and "
            "canonical-state fit handoff."
        ),
        mode="estimation",
        capabilities=(
            "canonical_core",
            "estimation",
            "services_layer",
            "biomechanics",
        ),
        web_route="/tools/canonical-core/estimation",
        order=36,
    ),
    CanonicalCoreTool(
        tool_id="canonical_core_comparison",
        name="Canonical-Core Comparison",
        description=(
            "Workspace entry point for CC-27 comparison services and "
            "cross-engine canonical-state review."
        ),
        mode="comparison",
        capabilities=(
            "canonical_core",
            "comparison",
            "services_layer",
            "biomechanics",
        ),
        web_route="/tools/canonical-core/comparison",
        order=37,
    ),
)


def canonical_core_tools() -> tuple[CanonicalCoreTool, ...]:
    """Return canonical-core tool descriptors in stable launcher order."""
    return _CANONICAL_CORE_TOOLS


def get_canonical_core_tool(tool_id: str) -> CanonicalCoreTool:
    """Return a canonical-core descriptor by registry id.

    Raises:
        ValueError: If ``tool_id`` is empty or unknown.
    """
    require(
        isinstance(tool_id, str) and bool(tool_id.strip()),
        "tool_id must be a non-empty string",
        tool_id,
    )
    for tool in _CANONICAL_CORE_TOOLS:
        if tool.tool_id == tool_id:
            return tool
    raise ValueError(f"Unknown canonical-core tool: {tool_id}")
