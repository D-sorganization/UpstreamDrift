"""Help and layout management dialogs for the launcher.

Provides HelpDialog, LayoutManagerDialog, and ContextHelpDock.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.ui.tile_help import help_relpath_for
from src.shared.python.logging_pkg.logging_config import get_logger

from .startup import ASSETS_DIR, REPOS_ROOT

logger = get_logger(__name__)


class HelpDialog(QDialog):
    """Dialog to display help documentation."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("UpstreamDrift - Help")
        self.resize(800, 600)
        layout = QVBoxLayout(self)
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        layout.addWidget(self.text_area)

        help_path = ASSETS_DIR / "help.md"
        if help_path.exists():
            self.text_area.setMarkdown(help_path.read_text(encoding="utf-8"))
        else:
            self.text_area.setText("Help file not found.")

        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)


class LayoutManagerDialog(QDialog):
    """Dialog allowing users to add or remove launcher tiles."""

    def __init__(
        self,
        available_models: dict[str, Any],
        active_models: list[str],
        parent: QWidget | None,
    ) -> None:
        if available_models is None:
            raise ValueError("available_models must be provided")
        super().__init__(parent)
        self.setWindowTitle("Customize Launcher Tiles")
        self.resize(520, 520)

        layout = QVBoxLayout(self)

        description = QLabel(
            "Select which applications should appear on the launcher grid. "
            "Checked items will be visible while unchecked items will be hidden."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        self.list_widget = QListWidget()

        sorted_models = sorted(
            available_models.values(),
            key=lambda model: getattr(model, "name", "").lower(),
        )

        for model in sorted_models:
            item = QListWidgetItem(f"{model.name} — {model.description}")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if model.id in active_models
                else Qt.CheckState.Unchecked
            )
            item.setData(Qt.ItemDataRole.UserRole, model.id)
            self.list_widget.addItem(item)

        layout.addWidget(self.list_widget)

        from PyQt6.QtWidgets import QHBoxLayout

        selection_layout = QHBoxLayout()
        btn_select_all = QPushButton("Select All")
        btn_deselect_all = QPushButton("Deselect All")

        btn_select_all.clicked.connect(
            lambda: self._set_all_states(Qt.CheckState.Checked)
        )
        btn_deselect_all.clicked.connect(
            lambda: self._set_all_states(Qt.CheckState.Unchecked)
        )

        selection_layout.addWidget(btn_select_all)
        selection_layout.addWidget(btn_deselect_all)
        selection_layout.addStretch()
        layout.addLayout(selection_layout)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _set_all_states(self, state: Qt.CheckState) -> None:
        """Set the check state for all items."""
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item:
                item.setCheckState(state)

    def selected_ids(self) -> list[str]:
        """Return IDs of all checked models."""
        selections: list[str] = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item and item.checkState() == Qt.CheckState.Checked:
                model_id = item.data(Qt.ItemDataRole.UserRole)
                if model_id:
                    selections.append(str(model_id))
        return selections


class ContextHelpDock(QDockWidget):
    """Context-aware help drawer."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Quick Help", parent)
        self.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
        )

        # Content
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.setStyleSheet(
            "background-color: #252526; color: #cccccc; border: none; padding: 10px;"
        )
        self.setWidget(self.text_area)

        # Default content
        self.update_context(None)

    def update_context(self, model_id: str | None) -> None:
        """Update help content based on selected model."""
        if not model_id:
            self.text_area.setMarkdown(
                "### Context Aware Help\n\nSelect a model to view its "
                "documentation and quick start guide."
            )
            return

        doc_file = self._get_doc_file(model_id)
        if doc_file is not None:
            try:
                self.text_area.setMarkdown(doc_file.read_text(encoding="utf-8"))
            except (RuntimeError, ValueError, OSError) as e:
                self.text_area.setText(f"Failed to load documentation: {e}")
            return

        self.text_area.setMarkdown(self._no_documentation_markdown(model_id))

    def _no_documentation_markdown(self, model_id: str) -> str:
        """Explain *why* there is no help, naming the locations searched.

        The dock used to claim "No specific documentation available" for 11 of
        the 22 mapped tiles while pointing at doc files that had never been
        written (issue #7986). Naming the searched paths makes the gap
        actionable instead of looking like a lookup miss.
        """
        candidates = self._doc_candidates(model_id)
        lines = [f"### {model_id}", ""]
        if candidates:
            lines.append(
                "No documentation file has been written for this tile yet. "
                "The following locations were searched:"
            )
            lines.append("")
            lines.extend(f"- `{self._relative_to_repo(p)}`" for p in candidates)
        else:
            lines.append(
                "This tile has no documentation mapping yet — no help file is "
                "associated with it."
            )
        lines.append("")
        lines.append(
            "General documentation: `docs/user_guide/user_manual.md`, "
            "`docs/architecture/PROJECT_MAP.md`, `docs/troubleshooting/FAQ.md`."
        )
        return "\n".join(lines)

    @staticmethod
    def _relative_to_repo(path: Path) -> str:
        """Render *path* relative to the repo root when it lives inside it."""
        try:
            return path.relative_to(REPOS_ROOT).as_posix()
        except ValueError:
            return str(path)

    def _doc_candidates(self, model_id: str) -> list[Path]:
        """Return every documentation path that could serve ``model_id``.

        Candidates are ordered most-specific first and are **not** filtered by
        existence — :meth:`_get_doc_file` does that, and
        :meth:`_no_documentation_markdown` needs the full list to report what
        was searched.
        """
        if model_id is None:
            raise ValueError("model_id must be provided")

        registry_page = help_relpath_for(model_id)
        if registry_page:
            # The tile registry (models.yaml `help:`) is authoritative; the
            # rule table below is only a legacy fallback for ids that predate
            # it (issue #9413 / #8843).
            return [REPOS_ROOT / registry_page]

        lowered = model_id.lower()
        docs = REPOS_ROOT / "docs"
        docs_engines = docs / "engines"
        docs_tutorials = docs / "tutorials" / "content"

        # (id fragments, candidate docs). First matching rule wins.
        rules: tuple[tuple[tuple[str, ...], tuple[Path, ...]], ...] = (
            (("mujoco",), (docs_engines / "mujoco.md",)),
            (("drake",), (docs_engines / "drake.md",)),
            (("pinocchio",), (docs_engines / "pinocchio.md",)),
            (("opensim",), (docs_engines / "opensim.md",)),
            (("myosim", "myosuite"), (docs_engines / "myosim.md",)),
            (("matlab",), (docs_engines / "matlab.md",)),
            (("simscape",), (docs_engines / "simscape.md",)),
            (("urdf",), (docs / "architecture" / "URDF_SUBSYSTEM_BOUNDARY.md",)),
            (
                ("c3d", "motion_capture"),
                (
                    docs / "motion_pipeline" / "compat.md",
                    docs / "help" / "motion_capture.md",
                ),
            ),
            (("openpose",), (docs_engines / "openpose.md",)),
            (
                ("mediapipe", "video_analyzer", "video_processor", "shot_tracer"),
                (docs_tutorials / "04_video_analysis.md",),
            ),
            (
                ("model_explorer",),
                (REPOS_ROOT / "src" / "tools" / "model_explorer" / "README.md",),
            ),
            (
                ("data_explorer", "data_processor"),
                (docs / "user_guide" / "user_manual.md",),
            ),
            (("putting_green", "pendulum_putter"), (docs_engines / "pendulum.md",)),
            (
                ("project_map",),
                (
                    docs / "architecture" / "PROJECT_MAP.md",
                    docs / "governance" / "PROJECT_MAP.md",
                ),
            ),
            (
                ("movement_optimizer",),
                (REPOS_ROOT.parent / "Movement_Optimizer" / "README.md",),
            ),
        )

        for fragments, candidates in rules:
            if any(fragment in lowered for fragment in fragments):
                return list(candidates)

        # Fallback: engine-specific README files.
        engines_root = REPOS_ROOT / "src" / "engines" / "physics_engines"
        for engine in ("mujoco", "drake", "pinocchio", "opensim", "myosuite"):
            if engine in lowered:
                return [engines_root / engine / "README.md"]

        return []

    def _get_doc_file(self, model_id: str) -> Path | None:
        """Return the first documentation file that actually exists."""
        for candidate in self._doc_candidates(model_id):
            if candidate.exists():
                return candidate
        return None
