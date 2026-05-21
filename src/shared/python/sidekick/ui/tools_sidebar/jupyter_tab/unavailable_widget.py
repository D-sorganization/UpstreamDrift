"""Placeholder shown when Jupyter dependencies are missing."""

from __future__ import annotations

from ..qt_compat import QtCore, QtWidgets


class JupyterUnavailableWidget(QtWidgets.QWidget):
    """Compact placeholder explaining how to install the Jupyter extras.

    Renders the install hint plus a copy-to-clipboard button so a user
    can quickly paste the install command into a terminal.
    """

    def __init__(
        self,
        install_hint: str,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("SidekickJupyterUnavailableWidget")
        self._install_hint = install_hint
        self._build_layout()

    def install_hint(self) -> str:
        """Return the install-hint text displayed by this widget."""
        return self._install_hint

    def _build_layout(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel("Jupyter notebook viewer", self)
        title.setObjectName("SidekickJupyterUnavailableTitle")
        title.setWordWrap(True)
        layout.addWidget(title)

        hint = QtWidgets.QLabel(self._install_hint, self)
        hint.setObjectName("SidekickJupyterInstallHint")
        hint.setWordWrap(True)
        hint.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(hint)

        copy_button = QtWidgets.QPushButton("Copy install command", self)
        copy_button.setObjectName("SidekickJupyterInstallCopy")
        copy_button.clicked.connect(self._copy_install_hint)
        layout.addWidget(copy_button)

        layout.addStretch(1)

    def _copy_install_hint(self) -> None:
        clipboard = QtWidgets.QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self._install_hint)
