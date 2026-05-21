"""Agentic Reporting and Summarization Engine tab for Sidekick.

Wires the sidebar context (workspace variables, project root, chat history)
into the shared ``ReportGenerator`` backend to produce LLM-driven session
reports instead of placeholder data.

Fixes: https://github.com/D-sorganization/Tools/issues/2743
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from . import design_tokens as theme
from .qt_compat import QtCore, QtWidgets, Signal

logger = logging.getLogger(__name__)

# Layout defaults from the design-token system
_MARGINS = getattr(theme, "SIDEBAR_LAYOUT_MARGINS", (8, 8, 8, 8))
_SPACING = getattr(theme, "SIDEBAR_LAYOUT_SPACING", 8)

# Sub-tabs whose runtime state belongs in a session report.  The UI label
# on the Reporting tab promises that the report aggregates "workspace
# context, chat interactions, and terminal history" — these IDs cover that
# promise plus the calculator and data-processor surfaces that share the
# same sidebar.  A widget is only included when it exposes a public
# ``get_context_snapshot()`` method that returns a JSON-serializable dict.
_SNAPSHOTTABLE_SUBTAB_IDS: tuple[str, ...] = (
    "chat",
    "terminal",
    "calculator",
    "data_processor",
)


def _gather_subtab_snapshots(sidebar: Any) -> dict[str, Any]:
    """Collect ``get_context_snapshot()`` payloads from known sub-tabs.

    The method is opt-in: a sub-tab is only included when its widget
    exposes a callable ``get_context_snapshot`` attribute.  Missing tabs
    and snapshot exceptions are swallowed so that an unhealthy sub-tab
    cannot block the rest of the report.

    Args:
        sidebar: The parent ``UnifiedToolsSidebar`` instance.  Must expose
            ``_tab_widgets`` (a ``dict[str, QWidget]``); sidebars without
            the attribute return an empty mapping.

    Returns:
        Mapping of tab id to snapshot dict.  Tabs without a snapshot
        method, or whose snapshot raises, are omitted entirely (no
        ``None`` pollution).
    """
    tab_widgets = getattr(sidebar, "_tab_widgets", None)
    if not isinstance(tab_widgets, dict):
        return {}

    snapshots: dict[str, Any] = {}
    for tab_id in _SNAPSHOTTABLE_SUBTAB_IDS:
        widget = tab_widgets.get(tab_id)
        if widget is None:
            continue
        snapshot_fn = getattr(widget, "get_context_snapshot", None)
        if not callable(snapshot_fn):
            continue
        try:
            snapshot = snapshot_fn()
        except Exception as exc:  # noqa: BLE001 - per-tab fault isolation
            logger.warning("Sub-tab %r get_context_snapshot raised: %s", tab_id, exc)
            continue
        snapshots[tab_id] = snapshot
    return snapshots


def _gather_session_context(sidebar: Any) -> dict[str, Any]:
    """Collect workspace state that the report generator can summarize.

    Args:
        sidebar: The parent ``UnifiedToolsSidebar`` instance.

    Returns:
        Dictionary of context data for report generation.  Always
        contains ``workspace_variables`` and ``project_root``; sub-tab
        snapshots (``chat``, ``terminal``, ``calculator``,
        ``data_processor``) are merged in when the corresponding widget
        exposes a ``get_context_snapshot()`` method.
    """
    variables: list[str] = []
    try:
        variables = [v.name for v in sidebar.registry.variables()]
    except Exception as exc:  # noqa: BLE001 - registry may not be ready
        logger.warning("Could not read workspace variables: %s", exc)

    project_root = str(getattr(sidebar, "project_root", "."))

    context: dict[str, Any] = {
        "workspace_variables": variables,
        "project_root": project_root,
    }
    context.update(_gather_subtab_snapshots(sidebar))
    return context


def _try_import_report_generator() -> Any | None:
    """Lazy-import the shared ``ReportGenerator``.

    Returns:
        A ``ReportGenerator`` class, or ``None`` when the reporting
        package is not installed.
    """
    try:
        from reporting.generator import ReportGenerator

        return ReportGenerator
    except Exception:  # noqa: BLE001 - optional dependency
        logger.debug("reporting.generator is not available; using local fallback")
        return None


def _format_local_report(context: dict[str, Any]) -> str:
    """Build a basic Markdown report without an LLM backend.

    This is the offline fallback when no ``InsightsProvider`` or
    ``ReportGenerator`` is available.

    Args:
        context: Context dictionary from ``_gather_session_context``.

    Returns:
        A Markdown-formatted report string.
    """
    variables = context.get("workspace_variables", [])
    project_root = context.get("project_root", "")
    var_list = "\n".join(f"- `{v}`" for v in variables) if variables else "_(none)_"
    return (
        "# Session Report\n\n"
        f"## Project\n{project_root}\n\n"
        f"## Workspace Variables\n{var_list}\n\n"
        "---\n"
    )


class _ReportWorker(QtCore.QThread):
    """Background worker that runs the async report generator in an isolated loop.

    Using a dedicated ``QThread`` avoids mixing ``asyncio`` with the Qt event
    loop and replaces the deprecated ``asyncio.get_event_loop()`` call.

    Signals:
        finished_with_report: Emitted with the completed report string on success.
        failed: Emitted with the error message string on failure.
    """

    finished_with_report = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        generator: Any,
        context: dict[str, Any],
        parent: QtCore.QObject | None = None,
    ) -> None:
        if generator is None:
            raise ValueError("generator must be provided")
        super().__init__(parent)
        self._generator = generator
        self._context = context

    def run(self) -> None:
        """Execute the async generator in an isolated event loop."""
        try:
            loop = asyncio.new_event_loop()
            try:
                report = loop.run_until_complete(
                    self._generator.generate_agentic_insights(self._context)
                )
                self.finished_with_report.emit(report)
            finally:
                loop.close()
        except Exception as exc:  # noqa: BLE001 - surface all errors via signal
            self.failed.emit(str(exc))


class SidekickReportingWidget(QtWidgets.QWidget):
    """Widget for generating agentic session reports.

    When the ``reporting.generator`` package is available and an
    ``InsightsProvider`` has been configured, the widget sends the
    gathered session context to the LLM backend for a comprehensive
    analysis.  Otherwise it produces a structured local report.
    """

    def __init__(self, sidebar: Any, parent: QtWidgets.QWidget | None = None) -> None:
        if sidebar is None:
            raise ValueError("sidebar must be provided")
        super().__init__(parent)
        self.sidebar = sidebar
        self.setObjectName("SidekickReportingWidget")
        self._report_generator: Any | None = None
        self._worker: _ReportWorker | None = None
        self._init_generator()
        self._build_ui()

    # ── Private helpers ──────────────────────────────────────────────

    def _init_generator(self) -> None:
        """Attempt to initialise the shared report generator."""
        gen_cls = _try_import_report_generator()
        if gen_cls is not None:
            try:
                self._report_generator = gen_cls()
            except Exception:  # noqa: BLE001
                logger.debug("ReportGenerator instantiation failed")

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(_MARGINS[0], _MARGINS[1], _MARGINS[2], _MARGINS[3])
        layout.setSpacing(_SPACING)

        # Info label
        self._info_label = QtWidgets.QLabel(
            "Generate a comprehensive report of your current session. "
            "This will aggregate workspace context, chat interactions, "
            "and terminal history."
        )
        self._info_label.setWordWrap(True)
        layout.addWidget(self._info_label)

        # Generate Button
        self._generate_btn = QtWidgets.QPushButton("Generate Session Report")
        self._generate_btn.setObjectName("SidekickReportingGenerate")
        self._generate_btn.clicked.connect(self._on_generate_clicked)
        layout.addWidget(self._generate_btn)

        # Preview area
        self._report_preview = QtWidgets.QTextEdit()
        self._report_preview.setObjectName("SidekickReportingPreview")
        self._report_preview.setReadOnly(True)
        self._report_preview.setPlaceholderText("Report preview will appear here...")
        layout.addWidget(self._report_preview, stretch=1)

        # Save Button
        self._save_btn = QtWidgets.QPushButton("Save Report")
        self._save_btn.setObjectName("SidekickReportingSave")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._on_save_clicked)
        layout.addWidget(self._save_btn)

    # ── Slots ────────────────────────────────────────────────────────

    def _on_generate_clicked(self) -> None:
        """Gather context and trigger report generation."""
        self._report_preview.setPlainText("Gathering context and generating report...")
        self._generate_btn.setEnabled(False)

        context = _gather_session_context(self.sidebar)

        if self._report_generator is not None:
            # Attempt async agentic report generation via QThread worker
            self._run_async_report(context)
        else:
            # Synchronous local fallback
            report = _format_local_report(context)
            self._on_report_generated(report)

    def _run_async_report(self, context: dict[str, Any]) -> None:
        """Run the async report generator in a background QThread.

        Creates a ``_ReportWorker``, connects its signals, and starts the
        thread.  The worker uses an isolated ``asyncio`` event loop so it
        never interferes with the Qt event loop.

        Args:
            context: Session context dictionary.
        """
        assert self._report_generator is not None  # guarded by caller
        self._worker = _ReportWorker(
            generator=self._report_generator, context=context, parent=self
        )
        self._worker.finished_with_report.connect(self._on_worker_report)
        self._worker.failed.connect(self._on_report_failed)
        self._worker.start()

    def _on_worker_report(self, insights: Any) -> None:
        """Handle successful agentic insights from the worker thread.

        Args:
            insights: Raw insights object returned by the report generator.
        """
        context = _gather_session_context(self.sidebar)
        report = _format_local_report(context)
        report += f"\n## Agentic Insights\n{insights}\n"
        self._on_report_generated(report)

    def _on_report_failed(self, error_message: str) -> None:
        """Handle a failure reported by the background worker.

        Args:
            error_message: Human-readable description of the failure.
        """
        logger.error("Async report generation failed: %s", error_message)
        context = _gather_session_context(self.sidebar)
        report = _format_local_report(context)
        report += f"\n## Insights Error\n{error_message}\n"
        self._on_report_generated(report)

    def _on_report_generated(self, report: str) -> None:
        """Display the generated report in the preview area.

        Args:
            report: The Markdown report string.
        """
        self._report_preview.setPlainText(report)
        self._generate_btn.setEnabled(True)
        self._save_btn.setEnabled(True)

    def _on_save_clicked(self) -> None:
        """Save the generated report to disk."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Session Report",
            "session_report.md",
            "Markdown Files (*.md);;All Files (*)",
        )
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self._report_preview.toPlainText())
                logger.info("Report saved to %s", path)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to save report: %s", exc)


def build_reporting_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build the Reporting tab for the Sidekick sidebar.

    Args:
        sidebar: The parent ``UnifiedToolsSidebar`` instance.

    Returns:
        The reporting widget.
    """
    widget = SidekickReportingWidget(sidebar=sidebar, parent=sidebar)
    widget.setToolTip("Generate an agentic summary and report of the session.")
    return widget
