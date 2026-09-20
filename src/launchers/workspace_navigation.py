"""Task-oriented desktop navigation and workspace definitions (ORG-05, #10515).

Implements the five primary task workspaces and secondary utilities:
1. Capture & Analyze
2. Model & Match
3. Shot & Course Lab
4. Optimize & Train
5. Results & Compare

Along with:
- Global utilities (All Tools, Favorites, History, Developer & Research).
- Identity migration through the authoritative ALIAS_MAP.
- Single-instance tool reuse policy.
- Guarded dirty-state tab management.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QTabWidget,
    QToolButton,
    QWidget,
)

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

#: Authoritative mapping of legacy/retired tile aliases to canonical identities
ALIAS_MAP: dict[str, str] = {
    "putting_green_gui": "putting_green",
    "starting_pose_matcher": "motion_target_preview",
    "cross_engine": "cross_engine_dashboard",
    "matlab_unified": "matlab_suite",
    "movement_optimizer": "tools_movement_optimizer",
}


@dataclass(frozen=True)
class WorkspaceInfo:
    """Descriptor for a launcher workspace destination."""

    id: str
    title: str
    description: str
    icon_name: str
    member_tool_ids: tuple[str, ...]
    is_primary: bool = True


#: The five primary task workspace definitions
PRIMARY_WORKSPACES: dict[str, WorkspaceInfo] = {
    "capture_analyze": WorkspaceInfo(
        id="capture_analyze",
        title="Capture & Analyze",
        description="Motion capture, video analysis, pose tracking, and time-series data",
        icon_name="camera",
        member_tool_ids=(
            "motion_capture",
            "video_analyzer",
            "c3d_viewer",
            "openpose_analysis",
            "mediapipe_analysis",
            "data_explorer",
            "data_processor",
            "rate_of_closure",
            "capture_rig",
        ),
    ),
    "model_match": WorkspaceInfo(
        id="model_match",
        title="Model & Match",
        description="Physics engines, biomechanical models, model exploration, and tour matching",
        icon_name="computer",
        member_tool_ids=(
            "mujoco_unified",
            "drake_golf",
            "pinocchio_golf",
            "opensim_golf",
            "myosim_suite",
            "model_explorer",
            "motion_target_preview",
            "tour_matching_viewer",
            "starting_pose_matcher",
        ),
    ),
    "shot_course_lab": WorkspaceInfo(
        id="shot_course_lab",
        title="Shot & Course Lab",
        description="Putting greens, launch monitors, bunker shots, impact, and course terrain",
        icon_name="sports_golf",
        member_tool_ids=(
            "putting_green",
            "golf_simulator",
            "bunkershot3d",
            "impact_explorer",
            "terrain",
            "swingset",
            "putting_green_gui",
        ),
    ),
    "optimize_train": WorkspaceInfo(
        id="optimize_train",
        title="Optimize & Train",
        description="Trajectory optimization, objective exploration, neural retargeting, and control",
        icon_name="build",
        member_tool_ids=(
            "tools_movement_optimizer",
            "movement_optimizer",
            "swing_objective_lab",
            "training",
            "pid_generator",
            "pendulum_simulator",
            "sg_optimizer",
        ),
    ),
    "results_compare": WorkspaceInfo(
        id="results_compare",
        title="Results & Compare",
        description="Cross-engine comparative dashboards, canonical comparison, and synthetic datasets",
        icon_name="assessment",
        member_tool_ids=(
            "cross_engine_dashboard",
            "cross_engine",
            "canonical_core_comparison",
            "canonical_core_estimation",
            "analysis_tools_api",
            "dataset_generator",
            "matlab_suite",
            "matlab_unified",
        ),
    ),
}

PRIMARY_WORKSPACE_IDS: tuple[str, ...] = tuple(PRIMARY_WORKSPACES.keys())

#: Secondary utilities navigation
SECONDARY_UTILITY_IDS: tuple[str, ...] = (
    "all_tools",
    "favorites",
    "history",
    "dev_research",
)

SECONDARY_WORKSPACES: dict[str, WorkspaceInfo] = {
    "all_tools": WorkspaceInfo(
        id="all_tools",
        title="All Tools",
        description="Searchable catalog of all available tools and engines",
        icon_name="grid_view",
        member_tool_ids=(),
        is_primary=False,
    ),
    "favorites": WorkspaceInfo(
        id="favorites",
        title="Favorites",
        description="User-starred favorites",
        icon_name="star",
        member_tool_ids=(),
        is_primary=False,
    ),
    "history": WorkspaceInfo(
        id="history",
        title="History",
        description="Recently and frequently launched tools",
        icon_name="history",
        member_tool_ids=(),
        is_primary=False,
    ),
    "dev_research": WorkspaceInfo(
        id="dev_research",
        title="Developer & Research",
        description="Developer utilities, research models, and headless integration services",
        icon_name="code",
        member_tool_ids=(
            "character_builder",
            "aip",
            "realtime_ws",
            "actuator_controls",
            "robotics_module",
            "unreal_integration",
            "perturbation_analysis",
            "force_overlays",
            "motion_pipeline",
        ),
        is_primary=False,
    ),
}


class WorkspaceDestination:
    """Constants for workspace navigation routes."""

    HOME = "home"
    CAPTURE_ANALYZE = "capture_analyze"
    MODEL_MATCH = "model_match"
    SHOT_COURSE_LAB = "shot_course_lab"
    OPTIMIZE_TRAIN = "optimize_train"
    RESULTS_COMPARE = "results_compare"
    ALL_TOOLS = "all_tools"
    FAVORITES = "favorites"
    HISTORY = "history"
    DEV_RESEARCH = "dev_research"


def is_primary_workspace(workspace_id: str) -> bool:
    """Return True if the workspace ID is one of the five primary task destinations."""
    return workspace_id in PRIMARY_WORKSPACES


def get_workspace_for_tool(tool_id: str) -> str | None:
    """Resolve which workspace a tool belongs to."""
    canonical_id = ALIAS_MAP.get(tool_id, tool_id)
    # Check primary workspaces first
    for ws_id, ws in PRIMARY_WORKSPACES.items():
        if canonical_id in ws.member_tool_ids or tool_id in ws.member_tool_ids:
            return ws_id
    # Check secondary developer/research tools
    if (
        canonical_id in SECONDARY_WORKSPACES["dev_research"].member_tool_ids
        or tool_id in SECONDARY_WORKSPACES["dev_research"].member_tool_ids
    ):
        return "dev_research"
    return None


def get_workspace_tools(workspace_id: str) -> tuple[str, ...]:
    """Return the tuple of member tool IDs for a workspace."""
    if workspace_id in PRIMARY_WORKSPACES:
        return PRIMARY_WORKSPACES[workspace_id].member_tool_ids
    if workspace_id in SECONDARY_WORKSPACES:
        return SECONDARY_WORKSPACES[workspace_id].member_tool_ids
    return ()


def migrate_model_order(order: list[str]) -> list[str]:
    """Migrate model order through ALIAS_MAP and deduplicate."""
    migrated: list[str] = []
    seen: set[str] = set()
    for mid in order:
        canonical = ALIAS_MAP.get(mid, mid)
        if canonical not in seen:
            seen.add(canonical)
            migrated.append(canonical)
    return migrated


def migrate_favorites(favorites: list[str]) -> list[str]:
    """Migrate favorites through ALIAS_MAP and deduplicate."""
    migrated: list[str] = []
    seen: set[str] = set()
    for fid in favorites:
        canonical = ALIAS_MAP.get(fid, fid)
        if canonical not in seen:
            seen.add(canonical)
            migrated.append(canonical)
    return migrated


def migrate_saved_layout(layout_data: dict[str, Any]) -> dict[str, Any]:
    """Migrate an entire saved layout configuration dictionary.

    Preserves user customizations (tile_scale, view_mode, dock_state)
    while upgrading model_order, favorites, and launch_stats to canonical identities.
    """
    result = dict(layout_data)
    if "model_order" in result and isinstance(result["model_order"], list):
        result["model_order"] = migrate_model_order(result["model_order"])
    if "favorites" in result and isinstance(result["favorites"], list):
        result["favorites"] = migrate_favorites(result["favorites"])
    if "launch_stats" in result and isinstance(result["launch_stats"], dict):
        new_stats: dict[str, Any] = {}
        for mid, stats in result["launch_stats"].items():
            canonical = ALIAS_MAP.get(mid, mid)
            if canonical in new_stats:
                # Merge count and pick latest launch time
                existing_count = new_stats[canonical].get("count", 0)
                added_count = stats.get("count", 0) if isinstance(stats, dict) else 0
                new_stats[canonical]["count"] = existing_count + added_count
            else:
                new_stats[canonical] = dict(stats) if isinstance(stats, dict) else stats
        result["launch_stats"] = new_stats
    return result


def find_existing_tool_tab(tab_widget: QTabWidget, tool_id: str) -> int | None:
    """Find the index of an already-open tab for the given tool ID."""
    canonical_id = ALIAS_MAP.get(tool_id, tool_id)
    for i in range(tab_widget.count()):
        widget = tab_widget.widget(i)
        if widget is not None:
            w_tool_id = widget.property("tool_id")
            if w_tool_id:
                if ALIAS_MAP.get(str(w_tool_id), str(w_tool_id)) == canonical_id:
                    return i
    return None


def focus_or_open_tool_tab(
    tab_widget: QTabWidget,
    tool_id: str,
    launcher_factory: Callable[[str], QWidget],
    title: str = "",
) -> int:
    """Single-instance policy: focus existing tab or launch a new one.

    If a tab hosting ``tool_id`` is already open, raises and focuses it.
    Otherwise, invokes ``launcher_factory`` to instantiate the tool widget,
    sets the ``tool_id`` property, and appends it to ``tab_widget``.
    """
    existing_idx = find_existing_tool_tab(tab_widget, tool_id)
    if existing_idx is not None:
        tab_widget.setCurrentIndex(existing_idx)
        return existing_idx

    new_widget = launcher_factory(tool_id)
    if new_widget.property("tool_id") is None:
        new_widget.setProperty("tool_id", ALIAS_MAP.get(tool_id, tool_id))

    display_title = title or tool_id.replace("_", " ").title()
    idx = tab_widget.addTab(new_widget, display_title)
    tab_widget.setCurrentIndex(idx)
    return idx


def close_tool_tab_guarded(
    tab_widget: QTabWidget,
    index: int,
    parent_widget: QWidget | None = None,
) -> bool:
    """Close a tab safely, prompting the user if the tool has unsaved dirty state.

    Returns:
        True if the tab was closed; False if close was cancelled by the user.
    """
    if index < 0 or index >= tab_widget.count():
        return False

    widget = tab_widget.widget(index)
    if widget is not None:
        is_dirty = widget.property("is_dirty")
        dirty_fn = getattr(widget, "is_dirty", None)
        if callable(dirty_fn):
            is_dirty = dirty_fn()

        if is_dirty:
            reply = QMessageBox.question(
                parent_widget or tab_widget,
                "Unsaved Changes",
                "This tool contains unsaved work. Are you sure you want to close it?",
                QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if reply != QMessageBox.StandardButton.Discard:
                return False

    tab_widget.removeTab(index)
    if widget is not None:
        widget.deleteLater()
    return True


class WorkspaceBreadcrumbBar(QFrame):
    """Accessible return-to-workspace breadcrumb bar (ORG-05, #10515).

    Provides return-to-workspace navigation with visible focus, accessible name,
    and styling adhering to the shared theme tokens.
    """

    def __init__(
        self,
        workspace_id: str,
        tool_name: str,
        on_return: Callable[[str], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.workspace_id = workspace_id
        self.tool_name = tool_name
        self.on_return = on_return
        self._init_ui()

    def _init_ui(self) -> None:
        from src.shared.python.theme.palette import DARK_THEME, get_current_colors

        try:
            colors = get_current_colors()
        except Exception:  # noqa: BLE001
            colors = DARK_THEME

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(8)

        ws_info = PRIMARY_WORKSPACES.get(self.workspace_id) or SECONDARY_WORKSPACES.get(
            self.workspace_id
        )
        ws_title = ws_info.title if ws_info else "Home"

        self.btn_workspace = QToolButton(self)
        self.btn_workspace.setText(f"← {ws_title}")
        self.btn_workspace.setToolTip(f"Return to {ws_title} workspace")
        self.btn_workspace.setAccessibleName(f"Return to {ws_title} workspace")
        self.btn_workspace.setAccessibleDescription(
            f"Navigate back to {ws_title} workspace in the launcher"
        )
        self.btn_workspace.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.btn_workspace.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_workspace.clicked.connect(lambda: self.on_return(self.workspace_id))

        self.lbl_separator = QLabel("/", self)
        self.lbl_tool = QLabel(self.tool_name, self)

        layout.addWidget(self.btn_workspace)
        layout.addWidget(self.lbl_separator)
        layout.addWidget(self.lbl_tool)
        layout.addStretch()

        self.setStyleSheet(f"""
            QFrame {{
                background-color: {colors.bg_elevated};
                border-bottom: 1px solid {colors.border_default};
            }}
            QToolButton {{
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 4px;
                color: {colors.primary};
                font-weight: bold;
                padding: 4px 8px;
            }}
            QToolButton:hover {{
                background-color: {colors.bg_highlight};
            }}
            QToolButton:focus {{
                border: 2px solid {colors.primary};
            }}
            QLabel {{
                color: {colors.text_secondary};
            }}
        """)


def explain_tool_status(
    tool_id: str,
    available_models: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Provide actionable explanation for tool capability status (ORG-05, #10515).

    Returns a dict with:
    - tool_id: canonical tool ID
    - available: bool indicating whether tool is ready to launch
    - status: 'ready', 'not_installed', 'external_dependency', 'retired', 'unregistered'
    - explanation: human-readable explanation and remediation steps
    """
    canonical_id = ALIAS_MAP.get(tool_id, tool_id)
    if available_models is not None:
        model = available_models.get(canonical_id) or available_models.get(tool_id)
        if model is not None:
            launcher_meta = getattr(model, "launcher", None)
            status = getattr(launcher_meta, "status", None) if launcher_meta else None
            if status == "ready":
                return {
                    "tool_id": canonical_id,
                    "available": True,
                    "status": "ready",
                    "explanation": f"{model.name} is ready to launch.",
                }
            if status in ("unlaunchable", "coming_soon"):
                return {
                    "tool_id": canonical_id,
                    "available": False,
                    "status": "coming_soon",
                    "explanation": f"{model.name} is currently non-launchable or under development.",
                }

    known_explanations: dict[str, tuple[str, str]] = {
        "matlab_suite": (
            "external_dependency",
            "MATLAB Suite requires a local MATLAB installation (R2024b+ with Simscape Multibody).",
        ),
        "opensim_golf": (
            "external_dependency",
            "OpenSim requires python OpenSim bindings (conda install -c opensim-org opensim).",
        ),
        "openpose_analysis": (
            "not_installed",
            "OpenPose analysis requires OpenPose binaries and CUDA GPU environment.",
        ),
        "unreal_integration": (
            "not_installed",
            "Unreal Engine integration requires Unreal Engine 5.4+ with UpstreamDrift LiveLink plugin.",
        ),
    }

    if canonical_id in known_explanations:
        stat, exp = known_explanations[canonical_id]
        return {
            "tool_id": canonical_id,
            "available": False,
            "status": stat,
            "explanation": exp,
        }

    return {
        "tool_id": canonical_id,
        "available": False,
        "status": "unregistered",
        "explanation": f"Capability '{canonical_id}' is not currently configured in the launcher catalog.",
    }
