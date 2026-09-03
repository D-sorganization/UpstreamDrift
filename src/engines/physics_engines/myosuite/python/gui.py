"""GUI shell for the MyoSuite dashboard.

MyoSuite is a Gym-environment wrapper around MuJoCo's musculoskeletal
models. Historically the engine has shipped without a Qt GUI — the
standalone entry point in :mod:`myosuite_physics_engine` simply pointed
the user at the MyoSuite web docs. As part of Subtask 5 / #4998 of EPIC
#4993 every launcher tile is required to expose an embeddable
``MainWidget`` so the launcher host can place the tool in a tab or dock
rather than always popping a top-level window.

This module provides:

- :class:`MainWidget` — a ``QWidget`` factory that the embed adapter
  hands to the launcher host. It renders a lightweight dashboard
  showing engine availability, the configured environment id (when
  one is loaded), and a hint describing how to drive the engine from
  Python or from the suite-wide tooling.
- :class:`MainWindow` — a thin :class:`QMainWindow` shell for the
  legacy standalone-launch path. It just wraps :class:`MainWidget` as
  its central widget so ``python -m myosuite`` keeps working alongside
  the embedded path.

The widget intentionally does **not** instantiate
:class:`MyoSuitePhysicsEngine` on construction. Building a MyoSuite env
requires the optional ``myosuite`` wheel and triggers a slow Gym /
MuJoCo import chain; we want the embedded widget to construct
synchronously even when MyoSuite is unavailable. Engine work is
triggered explicitly by user actions on the dashboard.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.ui.tile_help import attach_tile_help
from src.shared.python.logging_pkg.logging_config import get_logger

if TYPE_CHECKING:  # pragma: no cover - type hints only
    pass

logger = get_logger(__name__)

__all__ = ["MainWidget", "MainWindow"]


_DASHBOARD_INTRO = (
    "MyoSuite is a Gym-based suite of musculoskeletal environments built "
    "on MuJoCo. The launcher tile exposes engine status and a thin "
    "control surface; full training / rollout workflows live in the "
    "MyoSuite Python API and the shared muscle-analysis tooling."
)

_USAGE_SNIPPET = (
    "from src.engines.physics_engines.myosuite import Engine\n"
    "engine = Engine()\n"
    "engine.load_model('myoElbowPose1D6MRandom-v0')\n"
    "# ... drive the engine via the PhysicsEngine protocol ...\n"
)


class MainWidget(QWidget):
    """Embeddable MyoSuite dashboard widget.

    Constructs synchronously without importing the optional ``myosuite``
    wheel; clicking *Probe engine* triggers the lazy import path so a
    failure surfaces in the dashboard rather than at widget-construction
    time. This matches the pattern established for the other engine
    dashboards in Subtask 5 / #4998 of EPIC #4993.

    The widget owns no background timers or threads — :meth:`cleanup`
    is therefore a no-op beyond logging, but is provided so the host's
    embed contract works uniformly across tools.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("MyoSuite Dashboard")
        self._build_ui()
        logger.info("MyoSuite MainWidget initialized")

    # ---- UI construction ------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        title = QLabel("MyoSuite")
        title_font = title.font()
        title_font.setPointSize(max(title_font.pointSize() + 4, 14))
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        subtitle = QLabel("Muscle-actuated simulation environments (MuJoCo + Gym)")
        subtitle.setStyleSheet("color: gray;")
        layout.addWidget(subtitle)

        intro = QLabel(_DASHBOARD_INTRO)
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # Status row -----------------------------------------------------
        status_row = QHBoxLayout()
        self._status_label = QLabel("Engine status: not probed")
        self._status_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        status_row.addWidget(self._status_label)

        self._probe_button = QPushButton("Probe engine")
        self._probe_button.setToolTip(
            "Attempt to import the MyoSuite wheel and instantiate the engine."
        )
        self._probe_button.clicked.connect(self._on_probe_clicked)
        status_row.addWidget(self._probe_button)
        layout.addLayout(status_row)

        # Usage snippet --------------------------------------------------
        usage_label = QLabel("Programmatic usage:")
        layout.addWidget(usage_label)

        self._usage_view = QTextEdit()
        self._usage_view.setReadOnly(True)
        self._usage_view.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self._usage_view.setPlainText(_USAGE_SNIPPET)
        self._usage_view.setMinimumHeight(120)
        layout.addWidget(self._usage_view)

        layout.addStretch(1)

    # ---- handlers -------------------------------------------------------

    def _on_probe_clicked(self) -> None:
        """Lazy-probe the optional MyoSuite engine and report status."""
        try:
            # Lazy import: the engine module pulls in MyoSuite/Gym/MuJoCo.
            from .myosuite_physics_engine import MyoSuitePhysicsEngine

            # ``MyoSuitePhysicsEngine`` mixes in protocol attributes that
            # mypy flags as "abstract because empty body" — at runtime
            # the wrapper instantiates fine, and the probe is purely a
            # smoke test. Suppress the abstract-class diagnostic.
            engine = MyoSuitePhysicsEngine()  # type: ignore[abstract]
        except (
            Exception  # noqa: BLE001 - environment-dependent third-party import
        ) as exc:  # pragma: no cover - environment-dependent
            logger.warning("MyoSuite engine probe failed: %s", exc)
            self._status_label.setText(f"Engine status: unavailable ({exc!s})")
            return
        # We don't keep a handle to the engine here — the dashboard is a
        # status surface, not a runtime owner. Discarding the reference
        # is intentional.
        del engine
        self._status_label.setText("Engine status: available")

    # ---- embed-contract surface ----------------------------------------

    def cleanup(self) -> None:
        """Release any resources held by the widget.

        The dashboard owns no timers, sockets, or worker threads, so this
        is effectively a no-op. Provided for contract uniformity with
        the other engine dashboards.
        """
        logger.debug("MyoSuite MainWidget cleanup")


class MainWindow(QMainWindow):
    """Thin :class:`QMainWindow` shell for standalone launch.

    Exists purely so the legacy ``python -m`` entry point can show the
    dashboard outside the launcher. All real content lives in
    :class:`MainWidget`; this class just hosts it as the central widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("MyoSuite Dashboard")
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window)
        self._main_widget = MainWidget(self)
        self.setCentralWidget(self._main_widget)
        self.resize(1000, 700)
        attach_tile_help(self, "myosim_suite")

    @property
    def main_widget(self) -> MainWidget:
        """Return the wrapped :class:`MainWidget`."""
        return self._main_widget

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        """Forward close to :meth:`MainWidget.cleanup` for parity with embed."""
        try:
            self._main_widget.cleanup()
        finally:
            super().closeEvent(event)


def get_dockable_ui() -> MainWindow:
    """Return the main window instance for docking in the unified launcher."""
    return MainWindow()
