"""Catalog of all runnable UpstreamDrift tools with categories.

Every supported user-facing tool, launcher tile, panel, and utility
must have an entry here.  Hidden features must document their reason
and owner so coverage tests can enforce discoverability.

Design contracts
----------------
- :class:`UDToolEntry` validates ``tool_id``, ``category``, and
  ``hidden_reason`` on construction.
- :func:`get_ud_tool_catalog` is a lazy singleton — the same
  :class:`UDToolCatalog` instance is returned on subsequent calls.
- One category map feeds sidebar filters, launcher cards, and search
  (no duplicate category strings elsewhere).

Related issue: #5314.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Valid categories ──────────────────────────────────────────────────

VALID_CATEGORIES: frozenset[str] = frozenset(
    {
        "Physics Engines",
        "Biomechanics",
        "Simulation",
        "Motion Capture",
        "Motion Matching",
        "Analysis",
        "Visualization",
        "Signal Processing",
        "Robotics",
        "External Providers",
        "Documentation",
        "Developer Tools",
        "Media Processing",
    }
)


# ── Domain model ──────────────────────────────────────────────────────


@dataclass
class UDToolEntry:
    """A single UpstreamDrift tool entry in the launcher catalog.

    Attributes:
        tool_id:       Unique identifier (e.g. ``"mujoco_dashboard"``).
        title:         Human-readable tile label.
        category:      One of :data:`VALID_CATEGORIES`.
        description:   One-sentence description shown in the launcher.
        command:       Python module path or script to launch
                       (e.g. ``"src.launchers.mujoco_dashboard"``).
        is_hidden:     If ``True``, the tile is omitted from visible views.
        hidden_reason: Required when ``is_hidden=True``.  Must document
                       owner and unblock condition, e.g.
                       "Requires unreleased hardware (owner: eng, unblock: #1234)".
        icon:          Optional icon key for the tile image.

    Raises:
        ValueError: If ``tool_id`` is empty, ``category`` is not in
                    :data:`VALID_CATEGORIES`, or ``is_hidden=True`` but
                    ``hidden_reason`` is ``None`` or empty.
    """

    tool_id: str
    title: str
    category: str
    description: str
    command: str
    is_hidden: bool = False
    hidden_reason: str | None = None
    icon: str | None = None

    def __post_init__(self) -> None:
        if not self.tool_id or not self.tool_id.strip():
            raise ValueError("UDToolEntry.tool_id must be a non-empty string")
        if self.category not in VALID_CATEGORIES:
            raise ValueError(
                f"UDToolEntry.category '{self.category}' is not in VALID_CATEGORIES. "
                f"Use one of: {sorted(VALID_CATEGORIES)}"
            )
        if self.is_hidden and not self.hidden_reason:
            raise ValueError(
                f"Tool '{self.tool_id}' is hidden but has no hidden_reason. "
                "Document owner and unblock condition."
            )


# ── Catalog ───────────────────────────────────────────────────────────


class UDToolCatalog:
    """Immutable catalog of all UpstreamDrift tool entries.

    Provides category filtering, ID lookup, and visibility filtering.
    Construct once via :func:`get_ud_tool_catalog` (singleton factory).
    """

    def __init__(self) -> None:
        self._tools: list[UDToolEntry] = _build_catalog()

    def all_tools(self) -> list[UDToolEntry]:
        """Return all tools (visible and hidden), sorted by category + title.

        Postcondition: returned list is a copy — mutations do not affect
        the catalog.
        """
        return list(self._tools)

    def visible_tools(self) -> list[UDToolEntry]:
        """Return only non-hidden tools."""
        return [t for t in self._tools if not t.is_hidden]

    def by_category(self, category: str) -> list[UDToolEntry]:
        """Return all tools (including hidden) in ``category``.

        Args:
            category: Must be one of :data:`VALID_CATEGORIES`.

        Returns:
            List of :class:`UDToolEntry` in the given category.  Empty
            when the category has no registered tools.
        """
        return [t for t in self._tools if t.category == category]

    def get(self, tool_id: str) -> UDToolEntry | None:
        """Look up a tool by its unique ID.

        Args:
            tool_id: The tool identifier.

        Returns:
            :class:`UDToolEntry` or ``None`` if not found.
        """
        for tool in self._tools:
            if tool.tool_id == tool_id:
                return tool
        return None

    def list_categories(self) -> list[str]:
        """Return sorted list of categories that have at least one entry."""
        return sorted({t.category for t in self._tools})


# ── Singleton factory ─────────────────────────────────────────────────

_catalog_instance: UDToolCatalog | None = None


def get_ud_tool_catalog() -> UDToolCatalog:
    """Return the singleton :class:`UDToolCatalog`.

    Lazy-initialised on first call; subsequent calls return the same object.
    """
    global _catalog_instance
    if _catalog_instance is None:
        _catalog_instance = UDToolCatalog()
    return _catalog_instance


# ── Catalog definition ────────────────────────────────────────────────


def _build_catalog() -> list[UDToolEntry]:
    """Return the definitive list of all UpstreamDrift tool entries.

    Add new tools here.  Use ``is_hidden=True`` + ``hidden_reason`` for
    features that exist but are not yet ready to surface.
    """
    tools: list[UDToolEntry] = [
        # ── Physics Engines ───────────────────────────────────────────
        UDToolEntry(
            tool_id="mujoco_dashboard",
            title="MuJoCo Dashboard",
            category="Physics Engines",
            description="Launch the MuJoCo physics engine dashboard for swing analysis.",
            command="src.launchers.mujoco_dashboard",
            icon="mujoco",
        ),
        UDToolEntry(
            tool_id="drake_dashboard",
            title="Drake Dashboard",
            category="Physics Engines",
            description="Launch the Drake multi-body dynamics dashboard.",
            command="src.launchers.drake_dashboard",
            icon="drake",
        ),
        UDToolEntry(
            tool_id="pinocchio_dashboard",
            title="Pinocchio Dashboard",
            category="Physics Engines",
            description="Launch the Pinocchio rigid-body kinematics dashboard.",
            command="src.launchers.pinocchio_dashboard",
            icon="pinocchio",
        ),
        # ── Biomechanics ──────────────────────────────────────────────
        UDToolEntry(
            tool_id="cross_engine_dashboard",
            title="Cross-Engine Dashboard",
            category="Biomechanics",
            description="Compare swing dynamics across MuJoCo, Drake, and Pinocchio.",
            command="src.launchers.cross_engine_dashboard",
            icon="cross_engine",
        ),
        UDToolEntry(
            tool_id="exercise_dashboard",
            title="Exercise Dashboard",
            category="Biomechanics",
            description="Guided golf exercise and drill management.",
            command="src.launchers.exercise_dashboard",
            icon="exercise",
        ),
        # ── Simulation ────────────────────────────────────────────────
        UDToolEntry(
            tool_id="full_swing_ball_flight",
            title="Full Swing + Ball Flight",
            category="Simulation",
            description=(
                "Unified pipeline: physics engine swing → club-ball impact → "
                "aerodynamic ball flight visualization."
            ),
            command="src.launchers.shot_tracer",
            icon="shot_tracer",
        ),
        UDToolEntry(
            tool_id="pendulum_simulator",
            title="Pendulum Simulator",
            category="Simulation",
            description="Interactive pendulum dynamics (single, double, chaotic) "
            "with phase portraits and Poincaré maps.",
            command="src.shared.python.pendulum_simulator",
            icon="pendulum",
        ),
        # ── Motion Capture ────────────────────────────────────────────
        UDToolEntry(
            tool_id="motion_capture_launcher",
            title="Motion Capture",
            category="Motion Capture",
            description="Launch markerless motion capture recording and processing.",
            command="src.launchers.motion_capture_launcher",
            icon="mocap",
        ),
        # ── Motion Matching ───────────────────────────────────────────
        UDToolEntry(
            tool_id="pose_studio",
            title="Pose Studio",
            category="Motion Matching",
            description="Interactive cross-engine pose editor and matching tool.",
            command="src.tools.pose_studio",
            icon="pose_studio",
        ),
        # ── Analysis ─────────────────────────────────────────────────
        UDToolEntry(
            tool_id="launcher_diagnostics",
            title="Diagnostics",
            category="Analysis",
            description="Launcher health diagnostics and engine availability checks.",
            command="src.launchers.launcher_diagnostics",
            icon="diagnostics",
        ),
        # ── Visualization ─────────────────────────────────────────────
        UDToolEntry(
            tool_id="shot_tracer_viewer",
            title="Shot Tracer",
            category="Visualization",
            description="Interactive 3-D ball-flight trajectory viewer.",
            command="src.launchers.shot_tracer",
            icon="shot_tracer",
        ),
        # ── External Providers ────────────────────────────────────────
        UDToolEntry(
            tool_id="matlab_launcher",
            title="MATLAB Suite",
            category="External Providers",
            description="Launch MATLAB golf modeling tools and simulations.",
            command="src.launchers.matlab_launcher_unified",
            icon="matlab",
        ),
        # ── Developer Tools ───────────────────────────────────────────
        UDToolEntry(
            tool_id="onboarding",
            title="Getting Started",
            category="Developer Tools",
            description="Onboarding dialog to configure engines and API keys.",
            command="src.launchers.onboarding_dialog",
            icon="onboarding",
        ),
        UDToolEntry(
            tool_id="mcp_servers_config",
            title="MCP Servers",
            category="Developer Tools",
            description="Configure MCP server integrations for AI assistant tools.",
            command="src.launchers.mcp_servers_preferences",
            icon="mcp",
        ),
        UDToolEntry(
            tool_id="integrations_health",
            title="Integrations Health",
            category="Developer Tools",
            description="View live status of external integration dependencies.",
            command="src.launchers.integrations_health_window",
            icon="health",
        ),
    ]

    _validate_catalog(tools)
    return sorted(tools, key=lambda t: (t.category, t.title))


def _validate_catalog(tools: list[UDToolEntry]) -> None:
    """Assert catalog invariants at startup.

    Raises:
        AssertionError: If any invariant is violated.
    """
    ids = [t.tool_id for t in tools]
    if len(ids) != len(set(ids)):
        seen: set[str] = set()
        dupes = {i for i in ids if i in seen or seen.add(i)}  # type: ignore[func-returns-value]
        raise AssertionError(f"Duplicate tool_id(s) in catalog: {dupes}")
