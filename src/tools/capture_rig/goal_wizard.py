"""Standard, modeless Qt wizard for the capability-driven capture workflow."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
    QWizard,
    QWizardPage,
)

from .goal_catalog import StepDescription, step_descriptions
from .goal_planner import (
    CaptureGoalCatalog,
    CaptureRoute,
    CaptureStep,
    Readiness,
    resolve,
)

ACTION_LABELS = {
    "library": "Open Capture Library",
    "edit": "Open Swing Editor",
    "draw": "Open Drawing Tools",
    "references": "Open Expert Library",
    "compare_reference": "Open Expert Comparison",
    "my_clubs": "Open My Clubs",
    "calibration": "Review or Repeat Calibration",
    "workflow": "Open Step Controls",
}


def _label(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


def _scroll(layout: QVBoxLayout, content: QWidget) -> None:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(content)
    layout.addWidget(area, 1)


class OutcomePage(QWizardPage):
    def __init__(self, owner: CaptureWizard) -> None:
        super().__init__(owner)
        self.owner = owner
        self.setTitle("What Would You Like to Do?")
        self.setSubTitle("Choose one or more outcomes. Shared steps appear once.")
        layout = QVBoxLayout(self)
        content = QWidget()
        choices = QVBoxLayout(content)
        self.checks: dict[str, QCheckBox] = {}
        for goal in owner.catalog.goals:
            check = QCheckBox(goal.title)
            check.toggled.connect(self.completeChanged)
            self.checks[goal.id] = check
            choices.addWidget(check)
        choices.addStretch()
        _scroll(layout, content)
        load = QPushButton("Open a Plan from the Capability Map…")
        load.clicked.connect(owner.plan_requested)
        layout.addWidget(load)
        resume = QPushButton("Resume This Capture’s Saved Workflow")
        resume.clicked.connect(owner.resume_requested)
        layout.addWidget(resume)
        self.feedback = _label(
            "Import an existing video through Library; a camera connection is optional."
        )
        layout.addWidget(self.feedback)

    def isComplete(self) -> bool:  # noqa: N802
        return any(check.isChecked() for check in self.checks.values())

    def validatePage(self) -> bool:  # noqa: N802
        try:
            self.owner.build_route(
                key for key, check in self.checks.items() if check.isChecked()
            )
        except ValueError as exc:
            self.feedback.setText(str(exc))
            return False
        self.feedback.setText(
            "Your route is ready. Review each step’s status before continuing."
        )
        return True

    def nextId(self) -> int:  # noqa: N802
        return 1


class CaptureStepPage(QWizardPage):
    def __init__(
        self,
        owner: CaptureWizard,
        step: CaptureStep,
        description: StepDescription,
        position: int,
        total: int,
    ) -> None:
        super().__init__(owner)
        self.owner, self.step = owner, step
        self.state = Readiness("blocked", "Refresh status to inspect this capture")
        self.setTitle(description.title)
        self.setSubTitle(f"Step {position} of {total} · {description.purpose}")
        layout = QVBoxLayout(self)
        self.identity = _label()
        self.identity.setAccessibleName("Wizard Capture Identity")
        layout.addWidget(self.identity)
        self.status = _label(self.state.reason)
        self.status.setAccessibleName("Wizard Step Status")
        layout.addWidget(self.status)
        content = QWidget()
        instructions = QVBoxLayout(content)
        for text in description.instructions:
            instructions.addWidget(_label(f"• {text}"))
        instructions.addStretch()
        _scroll(layout, content)
        self.open_button = QPushButton(ACTION_LABELS[step.action])
        self.open_button.clicked.connect(lambda: owner.action_requested.emit(step.id))
        layout.addWidget(self.open_button)
        refresh = QPushButton("Refresh Status")
        refresh.clicked.connect(owner.refresh_requested)
        layout.addWidget(refresh)
        self.skip: QCheckBox | None = None
        if step.optional:
            self.skip = QCheckBox("Skip This Optional Step for Now")
            self.skip.toggled.connect(
                lambda checked: owner.skip_requested.emit(step.id, checked)
            )
            layout.addWidget(self.skip)
        layout.addWidget(
            _label(
                "Open the controls to review or run this step. Return with Capture Wizard. "
                "Back and Cancel preserve your recordings and saved work."
            )
        )

    def isComplete(self) -> bool:  # noqa: N802
        return self.state.status in {"done", "skipped"}

    def apply_state(self, identity: str, state: Readiness, busy: bool) -> None:
        self.state = (
            Readiness("blocked", "An action is running; wait for its result.")
            if busy
            else state
        )
        self.identity.setText(identity)
        labels = {
            "done": "Available / Reviewed",
            "ready": "Ready for Your Action",
            "blocked": "Needs Attention",
            "skipped": "Skipped",
        }
        self.status.setText(f"{labels[self.state.status]}\n{self.state.reason}")
        self.open_button.setEnabled(not busy)
        self.completeChanged.emit()


class CaptureWizard(QWizard):
    action_requested = pyqtSignal(str)
    refresh_requested = pyqtSignal()
    skip_requested = pyqtSignal(str, bool)
    save_requested = pyqtSignal()
    resume_requested = pyqtSignal()
    plan_requested = pyqtSignal()

    def __init__(
        self, catalog: CaptureGoalCatalog, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.catalog = catalog
        self.route: CaptureRoute | None = None
        self.step_pages: dict[str, CaptureStepPage] = {}
        self.descriptions = step_descriptions()
        self.setWindowTitle("Capture Wizard")
        self.setWizardStyle(QWizard.WizardStyle.ClassicStyle)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setTitleFormat(Qt.TextFormat.PlainText)
        self.setSubTitleFormat(Qt.TextFormat.PlainText)
        self.setOption(QWizard.WizardOption.IndependentPages)
        self.setOption(QWizard.WizardOption.HaveHelpButton)
        self.setOption(QWizard.WizardOption.HaveCustomButton1)
        self.setButtonText(QWizard.WizardButton.CustomButton1, "Save and Close")
        self.customButtonClicked.connect(lambda _button: self.save_requested.emit())
        self.choices = OutcomePage(self)
        self.setPage(0, self.choices)
        self.resize(760, 610)
        self.currentIdChanged.connect(lambda _id: self.refresh_requested.emit())

    @property
    def current_step(self) -> str | None:
        page = self.currentPage()
        return page.step.id if isinstance(page, CaptureStepPage) else None

    def choose(self, goals: Iterable[str]) -> None:
        selected = set(goals)
        checks = self.choices.checks
        if not selected.issubset(checks):
            raise ValueError("Plan contains an unknown capture outcome")
        for key, check in checks.items():
            check.setChecked(key in selected)

    def build_route(self, goals: Iterable[str]) -> None:
        route = resolve(self.catalog, goals)
        if route == self.route:
            return
        for page_id in self.pageIds():
            if page_id != 0:
                page = self.page(page_id)
                self.removePage(page_id)
                if page is not None:
                    page.deleteLater()
        self.route = route
        self.step_pages.clear()
        for index, step in enumerate(route.steps, 1):
            page = CaptureStepPage(
                self,
                step,
                self.descriptions[step.node_id or step.id],
                index,
                len(route.steps),
            )
            self.setPage(index, page)
            self.step_pages[step.id] = page

    def navigate(self, step_id: str) -> None:
        route = self.route
        if route is None or step_id not in self.step_pages:
            raise ValueError("Step is not in the selected route")
        self.setCurrentId(route.step_ids.index(step_id) + 1)

    def set_feedback(self, message: str) -> None:
        feedback = self.choices.feedback
        feedback.setText(message)
        page = self.currentPage()
        if isinstance(page, CaptureStepPage):
            page.status.setText(message)

    def apply_skips(self, skipped: Iterable[str]) -> None:
        selected = set(skipped)
        optional = {key for key, page in self.step_pages.items() if page.step.optional}
        if not selected.issubset(optional):
            raise ValueError("Only optional steps can be skipped")
        for key, page in self.step_pages.items():
            check = page.skip
            if check is not None:
                check.blockSignals(True)
                check.setChecked(key in selected)
                check.blockSignals(False)

    def update_evidence(
        self, identity: str, evidence: Mapping[str, Readiness], *, busy: bool = False
    ) -> None:
        for key, page in self.step_pages.items():
            page.apply_state(
                identity,
                evidence.get(
                    key,
                    Readiness(
                        "blocked", "Refresh or open this step to review its inputs"
                    ),
                ),
                busy,
            )
