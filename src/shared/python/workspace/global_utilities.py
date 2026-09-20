"""Global Workspace Utilities Coordinator (ORG-19, #10528).

Unifies Sidekick assistant, Setup Wizard, Contextual Help, and Project Library under
a shared workspace utilities facade preserving identity aliases, assistant context,
conversation history, onboarding preferences, and platform execution boundaries.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final
import uuid

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

__all__ = [
    "AssistantContextSnapshot",
    "CANONICAL_UTILITY_IDS",
    "DESKTOP_ONLY_ACTIONS",
    "GlobalWorkspaceUtilitiesCoordinator",
    "NativeActionUnavailableError",
    "OnboardingPreferences",
    "PlatformExecutionEnvironment",
    "UnknownUtilityError",
    "UtilityViewState",
    "WorkspaceContextualHelp",
]


class UnknownUtilityError(KeyError):
    """Raised when an unrecognized utility identifier is requested."""


class NativeActionUnavailableError(RuntimeError):
    """Raised when a browser environment attempts to execute a native-only desktop action."""


class PlatformExecutionEnvironment(str, Enum):
    """Execution platform environment for workspace utilities."""

    DESKTOP = "desktop"
    BROWSER = "browser"
    HEADLESS = "headless"


CANONICAL_UTILITY_IDS: Final[tuple[str, ...]] = (
    "sidekick",
    "setup_wizard",
    "help",
    "library",
)

UTILITY_ALIASES: Final[dict[str, str]] = {
    # Sidekick aliases
    "sidekick": "sidekick",
    "chat_assistant": "sidekick",
    "assistant": "sidekick",
    "ai_assistant": "sidekick",
    # Setup Wizard aliases
    "setup_wizard": "setup_wizard",
    "config_setup_wizard": "setup_wizard",
    "setup": "setup_wizard",
    # Library aliases
    "library": "library",
    "project_library": "library",
    "project_map": "library",
    # Help aliases
    "help": "help",
    "contextual_help": "help",
    "workspace_help": "help",
}

DESKTOP_ONLY_ACTIONS: Final[frozenset[str]] = frozenset(
    {
        "os_terminal",
        "jupyter",
        "mcp_server",
        "matlab",
        "system_file_picker",
        "native_subprocess",
    }
)


@dataclass(frozen=True)
class AssistantContextSnapshot:
    """Immutable snapshot of the active assistant context across workspaces.

    Invariants:
        - workspace_id is non-empty.
        - source is non-empty and identifies the origin of the context update.
        - timestamp_utc is an ISO 8601 formatted timestamp string.
    """

    workspace_id: str
    project_id: str | None = None
    active_run_id: str | None = None
    source: str = "workspace_navigation"
    timestamp_utc: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.workspace_id or not self.workspace_id.strip():
            raise ValueError("workspace_id must be a non-empty string")
        if not self.source or not self.source.strip():
            raise ValueError("source must be a non-empty string")


@dataclass(frozen=True)
class WorkspaceContextualHelp:
    """Contextual guidance for a workspace landing surface."""

    workspace_id: str
    title: str
    purpose: str
    required_inputs: tuple[str, ...]
    produced_artifacts: tuple[str, ...]
    next_compatible_actions: tuple[str, ...]
    limitations: tuple[str, ...]


@dataclass
class OnboardingPreferences:
    """User onboarding and first-run preferences.

    Invariants:
        - If onboarding_dismissed is True, it remains dismissed across saves and migrations.
    """

    first_run_completed: bool = False
    onboarding_dismissed: bool = False
    setup_wizard_completed: bool = False
    migrated_from_legacy: bool = False
    dismissed_at_utc: str | None = None


@dataclass
class UtilityViewState:
    """Current UI presentation state of a global utility."""

    is_open: bool = False
    active_utility_id: str | None = None
    previous_focus_widget_id: str | None = None


_WORKSPACE_HELP_CATALOG: Final[dict[str, WorkspaceContextualHelp]] = {
    "capture": WorkspaceContextualHelp(
        workspace_id="capture",
        title="Capture Rig & Optical Ingestion",
        purpose="Ingests multi-camera video, camera calibrations, and C3D optical marker trajectories.",
        required_inputs=(
            "Synchronized multi-camera video files",
            "Camera intrinsics/extrinsics calibration",
        ),
        produced_artifacts=(
            "Raw synchronized video streams",
            "C3D marker observations",
            "FreeMoCap session metadata",
        ),
        next_compatible_actions=("Open in Pose Inspection", "Inspect Optical Targets"),
        limitations=(
            "Requires pre-calibrated camera geometry for metric 3D reconstruction",
        ),
    ),
    "inspection": WorkspaceContextualHelp(
        workspace_id="inspection",
        title="Pose Inspection & Target Fitting",
        purpose="Inspects 2D/3D pose detections, FreeMoCap marker fits, and estimator confidence across frames.",
        required_inputs=(
            "Observation sets from MediaPipe or OpenPose",
            "Optical marker coordinate streams",
        ),
        produced_artifacts=(
            "Validated keypoint tracks",
            "Confidence score distributions",
            "C3D target observations",
        ),
        next_compatible_actions=(
            "Open in Model Calibration",
            "Configure Kinematic Subject",
        ),
        limitations=(
            "2D pixel coordinates cannot directly substitute for metric 3D coordinates without camera projection",
        ),
    ),
    "model_calibration": WorkspaceContextualHelp(
        workspace_id="model_calibration",
        title="Model Calibration & Subject Scaling",
        purpose="Calibrates subject anthropometrics, segment masses, joint centers, and club specifications.",
        required_inputs=(
            "Keypoint observations",
            "Subject height and weight",
            "Club geometry and mass properties",
        ),
        produced_artifacts=(
            "Scaled subject model specification",
            "Calibrated segment frames",
            "Club attachment spec",
        ),
        next_compatible_actions=("Open in Model Fitting", "Run Kinematic Fit"),
        limitations=("Segment scaling requires reliable landmark coordinates",),
    ),
    "fitting": WorkspaceContextualHelp(
        workspace_id="fitting",
        title="Kinematic Pose Fitting",
        purpose="Fits multi-body kinematic model poses to observed marker and keypoint trajectories.",
        required_inputs=("Calibrated subject model", "Tracked 3D target coordinates"),
        produced_artifacts=(
            "Generalized coordinate trajectories (q)",
            "Marker tracking residual reports",
        ),
        next_compatible_actions=("Open in Dynamics", "Run Dynamic Matching"),
        limitations=(
            "Pure kinematic fits do not verify joint torque feasibility or ground reaction consistency",
        ),
    ),
    "dynamics": WorkspaceContextualHelp(
        workspace_id="dynamics",
        title="Multi-Engine Dynamics",
        purpose="Solves inverse and forward multi-body dynamics, joint moments, and contact mechanics.",
        required_inputs=(
            "Kinematic trajectories (q, dq, ddq)",
            "Inertial model parameters",
            "Ground reaction forces",
        ),
        produced_artifacts=(
            "Joint torques (tau)",
            "Actuator effort metrics",
            "Mechanical power profiles",
        ),
        next_compatible_actions=(
            "Compare Flight Models",
            "Export Trajectory to Course",
        ),
        limitations=(
            "Requires verified physics engine backend availability (MuJoCo, Drake, Pinocchio, OpenSim)",
        ),
    ),
    "course": WorkspaceContextualHelp(
        workspace_id="course",
        title="Terrain, Course & Delivery",
        purpose="Simulates ground interactions, putting surfaces, bunker sand mechanics, and third-party simulator delivery.",
        required_inputs=(
            "Ball launch conditions",
            "Terrain mesh elevation and firmness",
        ),
        produced_artifacts=(
            "Ground roll trajectory",
            "Bunker contact records",
            "Simulator delivery receipts",
        ),
        next_compatible_actions=(
            "Open in Results Browser",
            "Replay in 3D Scene Viewer",
        ),
        limitations=(
            "Scene view provides visual review only and does not compute predictive physics",
        ),
    ),
    "results": WorkspaceContextualHelp(
        workspace_id="results",
        title="Results Browser & Comparison",
        purpose="Browses, filters, audits, and compares biomechanical runs and aerodynamic simulation artifacts.",
        required_inputs=("Registered result artifact files", "Run metadata manifests"),
        produced_artifacts=(
            "Comparison metric tables",
            "Exported CSV/JSON datasets with provenance headers",
        ),
        next_compatible_actions=("Open in Impact Explorer", "Export Evidence Package"),
        limitations=(
            "Cross-engine comparisons require matching coordinate frames and SI units",
        ),
    ),
    "optimization": WorkspaceContextualHelp(
        workspace_id="optimization",
        title="Optimization & Training Jobs",
        purpose="Executes bounded trajectory optimization, objective exploration, and reinforcement learning training.",
        required_inputs=(
            "Optimization objective weights",
            "Boundary constraints",
            "Initial reference trajectory",
        ),
        produced_artifacts=(
            "Optimized swing trajectories",
            "Training iteration checkpoints",
            "Objective Pareto metrics",
        ),
        next_compatible_actions=(
            "Evaluate in Results Browser",
            "Deploy Policy to Simulator",
        ),
        limitations=(
            "Requires solver backend installed and non-empty positive objective weights",
        ),
    ),
}


class GlobalWorkspaceUtilitiesCoordinator:
    """Unified coordinator for Sidekick, Setup, Help, and Library global utilities."""

    def __init__(self, session_id: str | None = None) -> None:
        self._session_id: str = session_id or str(uuid.uuid4())
        self._active_context: AssistantContextSnapshot = AssistantContextSnapshot(
            workspace_id="launcher",
            source="initialization",
        )
        self._chat_history: list[dict[str, Any]] = []
        self._onboarding_preferences: OnboardingPreferences = OnboardingPreferences()
        self._view_state: UtilityViewState = UtilityViewState()

    @property
    def session_id(self) -> str:
        """The stable session ID for this coordinator instance."""
        return self._session_id

    @property
    def onboarding_preferences(self) -> OnboardingPreferences:
        """User onboarding preferences."""
        return self._onboarding_preferences

    def resolve_utility_id(self, raw_id: str) -> str:
        """Resolve a raw or legacy utility identifier to its canonical identity."""
        cleaned = raw_id.strip().lower()
        canonical = UTILITY_ALIASES.get(cleaned)
        if canonical is None:
            raise UnknownUtilityError(
                f"Unknown utility '{raw_id}'. Expected one of: "
                f"{sorted(set(UTILITY_ALIASES.keys()))}"
            )
        return canonical

    def update_workspace_context(
        self,
        workspace_id: str,
        project_id: str | None = None,
        active_run_id: str | None = None,
        source: str = "workspace_navigation",
    ) -> AssistantContextSnapshot:
        """Update active assistant context for a newly selected workspace or project.

        Invariants:
            - Reuses the existing session ID (does NOT mint duplicate sessions).
            - Discards stale run references when transitioning without an explicit run.
            - Retains explicit source tracking.
        """
        cleaned_ws = workspace_id.strip()
        if not cleaned_ws:
            raise ValueError("workspace_id cannot be empty")

        # If switching workspaces and no new run_id is supplied, clear stale run_id
        resolved_run_id = active_run_id if active_run_id is not None else None

        self._active_context = AssistantContextSnapshot(
            workspace_id=cleaned_ws,
            project_id=project_id,
            active_run_id=resolved_run_id,
            source=source,
        )
        logger.debug(
            "Updated assistant workspace context: ws=%s, project=%s, run=%s, source=%s",
            cleaned_ws,
            project_id,
            resolved_run_id,
            source,
        )
        return self._active_context

    def get_active_context(self) -> AssistantContextSnapshot:
        """Return the current active assistant context snapshot."""
        return self._active_context

    def add_chat_message(self, role: str, content: str) -> None:
        """Record a chat message into conversation history.

        Invariants:
            - Stored in persistent sequence for this session.
            - Survives workspace navigation without reset.
        """
        msg = {
            "role": role,
            "content": content,
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "workspace_id": self._active_context.workspace_id,
        }
        self._chat_history.append(msg)

    def get_chat_history(self) -> list[dict[str, Any]]:
        """Return full conversation history for the current session."""
        return list(self._chat_history)

    def should_show_first_run_setup(self) -> bool:
        """Determine whether the first-run Setup Wizard should be displayed."""
        if self._onboarding_preferences.onboarding_dismissed:
            return False
        return not self._onboarding_preferences.first_run_completed

    def dismiss_onboarding(self) -> None:
        """Dismiss onboarding; stays dismissed permanently across future sessions."""
        self._onboarding_preferences.onboarding_dismissed = True
        self._onboarding_preferences.dismissed_at_utc = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()

    def complete_first_run(self) -> None:
        """Mark first run as completed."""
        self._onboarding_preferences.first_run_completed = True
        self._onboarding_preferences.setup_wizard_completed = True

    def migrate_onboarding_preferences(
        self, legacy_preferences: dict[str, Any]
    ) -> OnboardingPreferences:
        """Migrate legacy onboarding preferences without resetting user dismissal."""
        dismissed = bool(
            legacy_preferences.get("skip_onboarding")
            or legacy_preferences.get("dismissed")
            or legacy_preferences.get("onboarding_dismissed")
        )
        first_run = bool(
            legacy_preferences.get("first_run_done")
            or legacy_preferences.get("first_run_completed")
        )
        setup_done = bool(
            legacy_preferences.get("setup_complete")
            or legacy_preferences.get("setup_wizard_completed")
        )

        self._onboarding_preferences = OnboardingPreferences(
            first_run_completed=first_run,
            onboarding_dismissed=dismissed
            or self._onboarding_preferences.onboarding_dismissed,
            setup_wizard_completed=setup_done,
            migrated_from_legacy=True,
        )
        return self._onboarding_preferences

    def open_utility(
        self,
        utility_id: str,
        previous_focus_widget_id: str | None = None,
    ) -> UtilityViewState:
        """Open a global utility and record the previous focus target."""
        canonical = self.resolve_utility_id(utility_id)
        self._view_state = UtilityViewState(
            is_open=True,
            active_utility_id=canonical,
            previous_focus_widget_id=previous_focus_widget_id
            or self._view_state.previous_focus_widget_id,
        )
        return self._view_state

    def close_utility(self, utility_id: str | None = None) -> str | None:
        """Close utility and return the widget ID that should regain keyboard focus."""
        if utility_id is not None:
            canonical = self.resolve_utility_id(utility_id)
            if self._view_state.active_utility_id != canonical:
                return self._view_state.previous_focus_widget_id

        restore_widget_id = self._view_state.previous_focus_widget_id
        self._view_state = UtilityViewState(
            is_open=False,
            active_utility_id=None,
            previous_focus_widget_id=restore_widget_id,
        )
        return restore_widget_id

    def toggle_utility(
        self,
        utility_id: str,
        current_focus_widget_id: str | None = None,
    ) -> UtilityViewState:
        """Toggle utility visibility via keyboard shortcut (e.g. F1 or Ctrl+Shift+S)."""
        canonical = self.resolve_utility_id(utility_id)
        if self._view_state.is_open and self._view_state.active_utility_id == canonical:
            self.close_utility(canonical)
            return self._view_state
        return self.open_utility(canonical, current_focus_widget_id)

    def get_focus_restore_target(self) -> str | None:
        """Return the recorded focus restore target after a utility is closed."""
        return self._view_state.previous_focus_widget_id

    def get_workspace_help(self, workspace_id: str) -> WorkspaceContextualHelp:
        """Retrieve structured contextual help for a workspace."""
        cleaned = workspace_id.strip().lower()
        help_doc = _WORKSPACE_HELP_CATALOG.get(cleaned)
        if help_doc is None:
            raise KeyError(
                f"No contextual help configured for workspace '{workspace_id}'. "
                f"Available: {sorted(_WORKSPACE_HELP_CATALOG.keys())}"
            )
        return help_doc

    def execute_action(
        self,
        action_id: str,
        platform: PlatformExecutionEnvironment
        | str = PlatformExecutionEnvironment.DESKTOP,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute or route a utility action with platform boundary enforcement."""
        cleaned_action = action_id.strip().lower()
        platform_enum = (
            platform
            if isinstance(platform, PlatformExecutionEnvironment)
            else PlatformExecutionEnvironment(platform)
        )

        if platform_enum == PlatformExecutionEnvironment.BROWSER:
            if cleaned_action in DESKTOP_ONLY_ACTIONS:
                raise NativeActionUnavailableError(
                    f"Action '{action_id}' cannot be executed in browser mode; "
                    "requires desktop runtime environment."
                )

        # Execute safe action
        return {
            "action_id": cleaned_action,
            "status": "ok",
            "executed": True,
            "platform": platform_enum.value,
        }
