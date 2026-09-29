"""JaxSim engine dashboard for the cross-engine exercise selector (issue #6658).

This widget surfaces the JaxSim backend inside the exercise dashboard's engine
selector. It is intentionally capability-driven: rather than hard-coding which
features are available, it reads the backend's declared ``EngineCapabilities``
(the same taxonomy used by the API and parity layers) and greys out any
feature whose support level is not ``FULL``.

The feature rows are *read-only capability indicators*, not actionable
controls (#6901). JaxSim is Linux-only and is never imported here, so there is
no GUI-side action to invoke; the dashboard therefore reports support levels
via non-interactive labels instead of buttons that would otherwise advertise
"Supported" while doing nothing.

The parameter-sensitivity panel entry is a *stub*: it references the gated
issue #6656 (ZTCF parameter-gradient semantics) but performs no gradient
compute. Only the non-gated indicator (the disabled entry plus an explanatory
tooltip) ships here.

JaxSim itself is Linux-only and is never imported by this module: the
capability report is a pure dataclass produced by ``JaxSimBackend`` without a
loaded model, so this dashboard constructs and renders on any platform.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QWidget,
)

from src.engines.physics_engines.jaxsim import JaxSimBackend
from src.shared.python.engine_core.capabilities import (
    CapabilityLevel,
    EngineCapabilities,
)

#: Issue tracking the gated parameter-gradient semantics decision. The panel
#: entry is wired but disabled until this issue's ZTCF-semantics owner signs off.
PARAMETER_SENSITIVITY_GATED_ISSUE = 6656

#: Feature rows surfaced in the dashboard, mapped to the capability attribute
#: that governs whether the control is enabled. Keeping this as data (rather
#: than per-feature branches) keeps the enable/disable logic DRY.
_FEATURE_ROWS: tuple[tuple[str, str], ...] = (
    ("Forward simulation", "forward_sim"),
    ("Mass matrix M(q)", "mass_matrix"),
    ("Inverse dynamics", "inverse_dynamics"),
    ("Spatial Jacobian", "jacobian"),
    ("Contact forces", "contact_forces"),
    ("Drift / ZTCF acceleration", "drift_acceleration"),
    ("Dataset export", "dataset_export"),
)


def _capability_is_enabled(level: CapabilityLevel) -> bool:
    """Return whether a capability level should highlight its indicator.

    Only ``FULL`` support marks a feature as available; ``PARTIAL`` and
    ``NONE`` grey it out so the UI never advertises an incomplete capability
    as ready.
    """

    return level == CapabilityLevel.FULL


def _capability_marker(level: CapabilityLevel) -> str:
    """Glyph prefix conveying a capability's support level at a glance."""

    if level == CapabilityLevel.FULL:
        return "✓"  # check mark
    if level == CapabilityLevel.PARTIAL:
        return "–"  # en dash (partial)
    return "✗"  # ballot x (none)


def _capability_tooltip(level: CapabilityLevel) -> str:
    """Human-readable explanation of a read-only capability indicator.

    These indicators are *informational only* — they report the backend's
    declared support level and are not actionable controls (#6901). The
    wording therefore never promises a clickable action.
    """

    if level == CapabilityLevel.FULL:
        return (
            "Capability indicator (read-only): fully supported by the JaxSim backend."
        )
    if level == CapabilityLevel.PARTIAL:
        return (
            "Capability indicator (read-only): partial support — not yet fully covered."
        )
    return "Capability indicator (read-only): not implemented by the JaxSim backend."


class JaxSimDashboard(QWidget):
    """Capability-driven JaxSim dashboard surfaced in the engine selector.

    Args:
        exercise_filter: Name of the biomechanics exercise this dashboard is
            scoped to. Stored for parity with the other engine dashboards; the
            JaxSim panel is currently informational and capability-driven.
        capabilities: Optional injected capability report (test seam). Defaults
            to ``JaxSimBackend().get_capabilities()``.
        parent: Optional Qt parent widget.
    """

    def __init__(
        self,
        *,
        exercise_filter: str = "gait",
        capabilities: EngineCapabilities | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if not exercise_filter.strip():
            raise ValueError("exercise_filter must be non-empty")
        self.exercise_filter = exercise_filter
        self._capabilities = capabilities or JaxSimBackend().get_capabilities()

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(f"JaxSim — {self.exercise_filter.title()} (Linux CI engine)")
        )

        #: Maps feature label -> read-only capability indicator, exposed for
        #: capability-driven tests. These are *not* actionable controls: they
        #: report the backend's declared support level only (#6901). Using a
        #: non-interactive ``QLabel`` (rather than a ``QPushButton``) means the
        #: UI never advertises a clickable "Supported" action that does
        #: nothing when there is no GUI-side JaxSim action to invoke.
        self.feature_controls: dict[str, QLabel] = {}
        for label, attr in _FEATURE_ROWS:
            level = getattr(self._capabilities, attr)
            indicator = self._build_capability_indicator(label, level)
            self.feature_controls[label] = indicator
            layout.addWidget(indicator)

        #: Stubbed parameter-sensitivity entry (gated on #6656). Read-only
        #: indicator; it performs no gradient compute.
        self.parameter_sensitivity_button = self._build_parameter_sensitivity_stub()
        layout.addWidget(self.parameter_sensitivity_button)

    @staticmethod
    def _build_capability_indicator(label: str, level: CapabilityLevel) -> QLabel:
        """Create a read-only capability indicator for a feature row (#6901).

        The indicator's ``enabled`` state mirrors the capability level (FULL ->
        enabled) so the data-driven contract remains testable, but it is never
        an interactive control.
        """

        indicator = QLabel(f"{_capability_marker(level)}  {label}")
        # Non-interactive: an indicator, not a clickable control.
        indicator.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        indicator.setEnabled(_capability_is_enabled(level))
        indicator.setToolTip(_capability_tooltip(level))
        return indicator

    def _build_parameter_sensitivity_stub(self) -> QLabel:
        """Create the read-only parameter-sensitivity panel entry (#6656 stub)."""

        button = QLabel("Parameter sensitivity (ZTCF)")
        button.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        button.setEnabled(False)
        button.setToolTip(
            "Gated on issue "
            f"#{PARAMETER_SENSITIVITY_GATED_ISSUE} (ZTCF parameter-gradient "
            "semantics). Wiring is present; gradient compute is intentionally "
            "not implemented here."
        )
        self.feature_controls["Parameter sensitivity (ZTCF)"] = button
        return button
