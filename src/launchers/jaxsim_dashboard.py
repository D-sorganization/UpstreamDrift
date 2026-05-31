"""JaxSim engine dashboard for the cross-engine exercise selector (issue #6658).

This widget surfaces the JaxSim backend inside the exercise dashboard's engine
selector. It is intentionally capability-driven: rather than hard-coding which
feature buttons are available, it reads the backend's declared
``EngineCapabilities`` (the same taxonomy used by the API and parity layers)
and greys out any feature whose support level is not ``FULL``.

The parameter-sensitivity panel entry is a *stub*: it references the gated
issue #6656 (ZTCF parameter-gradient semantics) but performs no gradient
compute. Only the non-gated wiring (the disabled control plus an explanatory
tooltip) ships here.

JaxSim itself is Linux-only and is never imported by this module: the
capability report is a pure dataclass produced by ``JaxSimBackend`` without a
loaded model, so this dashboard constructs and renders on any platform.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QLabel,
    QPushButton,
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
    """Return whether a capability level should enable its control.

    Only ``FULL`` support enables a feature; ``PARTIAL`` and ``NONE`` grey it
    out so the UI never advertises an incomplete capability as ready.
    """

    return level == CapabilityLevel.FULL


def _capability_tooltip(level: CapabilityLevel) -> str:
    """Human-readable explanation for a capability's enable/disable state."""

    if level == CapabilityLevel.FULL:
        return "Supported by the JaxSim backend."
    if level == CapabilityLevel.PARTIAL:
        return "Partial support — disabled until full coverage lands."
    return "Not implemented by the JaxSim backend."


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

        #: Maps feature label -> control, exposed for capability-driven tests.
        self.feature_controls: dict[str, QPushButton] = {}
        for label, attr in _FEATURE_ROWS:
            level = getattr(self._capabilities, attr)
            button = QPushButton(label)
            button.setEnabled(_capability_is_enabled(level))
            button.setToolTip(_capability_tooltip(level))
            self.feature_controls[label] = button
            layout.addWidget(button)

        #: Stubbed parameter-sensitivity entry (gated on #6656). Wired but
        #: disabled; it performs no gradient compute.
        self.parameter_sensitivity_button = self._build_parameter_sensitivity_stub()
        layout.addWidget(self.parameter_sensitivity_button)

    def _build_parameter_sensitivity_stub(self) -> QPushButton:
        """Create the disabled parameter-sensitivity panel entry (#6656 stub)."""

        button = QPushButton("Parameter sensitivity (ZTCF)")
        button.setEnabled(False)
        button.setToolTip(
            "Gated on issue "
            f"#{PARAMETER_SENSITIVITY_GATED_ISSUE} (ZTCF parameter-gradient "
            "semantics). Wiring is present; gradient compute is intentionally "
            "not implemented here."
        )
        self.feature_controls["Parameter sensitivity (ZTCF)"] = button
        return button
