"""Overview tab — capture-metadata summary plus a full parameter tree."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from ...core.models import C3DDataModel

_ELEVATED_GROUPS_PRIORITY = (
    "POINT",
    "ANALOG",
    "FORCE_PLATFORM",
    "TRIAL",
    "MANUFACTURER",
)
_PROVENANCE_PATTERNS = (
    "CAPTURE_ID",
    "EXPORTED_AT",
    "PLAYER_ID",
    "SUB_TYPE",
    "CREATED_AT",
)


def _is_metadata_internal_key(key: str) -> bool:
    return key.startswith("__") and key.endswith("__")


def _scalar_value(node: Any) -> Any:
    """Extract the user-facing value from an ezc3d-style ``{"value": ...}``."""
    if isinstance(node, dict) and "value" in node:
        node = node["value"]
    if isinstance(node, np.ndarray):
        if node.ndim == 0:
            return node.item()
        if node.size == 1:
            return node.flat[0]
        return node
    if isinstance(node, list) and len(node) == 1:
        return node[0]
    return node


def _format_value(node: Any) -> str:
    val = _scalar_value(node)
    if isinstance(val, np.ndarray):
        flat = val.ravel()
        head = ", ".join(repr(x) for x in flat[:5].tolist())
        suffix = ", …" if flat.size > 5 else ""
        return f"shape={val.shape} [{head}{suffix}]"
    if isinstance(val, list):
        if len(val) > 5:
            head = ", ".join(repr(x) for x in val[:5])
            return f"len={len(val)} [{head}, …]"
        return repr(val)
    if isinstance(val, (bytes, bytearray)):
        return val.decode("latin-1", errors="replace")
    if isinstance(val, str):
        return val
    return repr(val)


def _summarize_capture(model: C3DDataModel) -> list[tuple[str, str]]:
    """Build the elevated capture-metadata summary rows."""
    out: list[tuple[str, str]] = []
    out.append(("File", os.path.basename(model.filepath) if model.filepath else ""))
    out.append(("Point rate (Hz)", f"{model.point_rate:.3f}"))
    out.append(
        ("Analog rate (Hz)", f"{model.analog_rate:.3f}" if model.analog_rate else "N/A")
    )
    n_frames = len(model.point_time) if model.point_time is not None else 0
    out.append(("Frames", str(n_frames)))
    duration = (n_frames / model.point_rate) if model.point_rate > 0 else 0.0
    out.append(("Duration", f"{duration:.3f} s"))
    raw = model.raw_parameters or {}
    point = raw.get("POINT", {}) if isinstance(raw, dict) else {}
    units = _scalar_value(point.get("UNITS")) if isinstance(point, dict) else None
    if units:
        out.append(("Units (POINT)", str(units)))
    if isinstance(point, dict) and "X_SCREEN" in point and "Y_SCREEN" in point:
        out.append(
            (
                "Axis convention",
                f"X={_scalar_value(point.get('X_SCREEN'))!s}  "
                f"Y={_scalar_value(point.get('Y_SCREEN'))!s}",
            )
        )
    fp = raw.get("FORCE_PLATFORM", {}) if isinstance(raw, dict) else {}
    if isinstance(fp, dict) and "USED" in fp:
        out.append(("Force-plate count", f"{_scalar_value(fp.get('USED'))!s}"))
    analog = raw.get("ANALOG", {}) if isinstance(raw, dict) else {}
    if isinstance(analog, dict) and "USED" in analog:
        out.append(("Analog channels", f"{_scalar_value(analog.get('USED'))!s}"))
    mfg = raw.get("MANUFACTURER", {}) if isinstance(raw, dict) else {}
    if isinstance(mfg, dict):
        sw = _scalar_value(mfg.get("SOFTWARE")) if "SOFTWARE" in mfg else None
        ver = _scalar_value(mfg.get("VERSION")) if "VERSION" in mfg else None
        if sw or ver:
            out.append(("Software", f"{sw or ''} {ver or ''}".strip()))
    return out


def _collect_provenance(model: C3DDataModel) -> list[tuple[str, str]]:
    """Walk the param tree gathering provenance-pattern keys."""
    raw = model.raw_parameters or {}
    if not isinstance(raw, dict):
        return []
    rows: list[tuple[str, str]] = []
    for group_name, group in raw.items():
        if not isinstance(group, dict):
            continue
        for key, node in group.items():
            if _is_metadata_internal_key(key):
                continue
            if any(p in key.upper() for p in _PROVENANCE_PATTERNS):
                rows.append((f"{group_name}.{key}", _format_value(node)))
    return rows


class OverviewTab(QtWidgets.QWidget):
    """Overview tab with elevated summary and full C3D parameter tree."""

    def __init__(self) -> None:
        super().__init__()
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        self.label_file = QtWidgets.QLabel("No file loaded")
        self.label_file.setWordWrap(True)
        self.label_file.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.label_file)

        # Capture metadata summary (top, always visible).
        summary_group = QtWidgets.QGroupBox("Capture metadata")
        summary_layout = QtWidgets.QFormLayout(summary_group)
        self._summary_rows: dict[str, QtWidgets.QLabel] = {}
        self._summary_form = summary_layout
        self._summary_group = summary_group
        layout.addWidget(summary_group)

        # Provenance subsection.
        self._prov_group = QtWidgets.QGroupBox("Provenance")
        self._prov_form = QtWidgets.QFormLayout(self._prov_group)
        layout.addWidget(self._prov_group)
        self._prov_group.setVisible(False)

        # Full parameter tree.
        layout.addWidget(QtWidgets.QLabel("Full parameter tree:"))
        self.tree = QtWidgets.QTreeView()
        self.tree.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.tree.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_context_menu)
        self._tree_model = QtGui.QStandardItemModel()
        self._tree_model.setHorizontalHeaderLabels(["Key", "Value"])
        self.tree.setModel(self._tree_model)
        layout.addWidget(self.tree, 1)

    # ----------------------------------------------------------- Public API

    def update_from_model(self, model: C3DDataModel | None) -> None:
        """Update UI with data from the model."""
        # Reset summary form.
        while self._summary_form.rowCount() > 0:
            self._summary_form.removeRow(0)
        # Reset provenance form.
        while self._prov_form.rowCount() > 0:
            self._prov_form.removeRow(0)
        self._tree_model.removeRows(0, self._tree_model.rowCount())

        if model is None:
            self.label_file.setText("No file loaded")
            self._prov_group.setVisible(False)
            return

        self.label_file.setText(f"Loaded file: {model.filepath}")

        for key, value in _summarize_capture(model):
            label = QtWidgets.QLabel(value)
            label.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self._summary_form.addRow(QtWidgets.QLabel(key + ":"), label)

        prov = _collect_provenance(model)
        if prov:
            for key, value in prov:
                label = QtWidgets.QLabel(value)
                label.setTextInteractionFlags(
                    QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
                )
                self._prov_form.addRow(QtWidgets.QLabel(key + ":"), label)
            self._prov_group.setVisible(True)
        else:
            label = QtWidgets.QLabel("no provenance available")
            label.setStyleSheet("color: gray; font-style: italic;")
            self._prov_form.addRow(label)
            self._prov_group.setVisible(True)

        self._populate_tree(model)

    @property
    def tree_node_count(self) -> int:
        """Total number of nodes (groups + their children) in the tree."""
        count = 0

        def _walk(parent: QtGui.QStandardItem) -> None:
            nonlocal count
            for r in range(parent.rowCount()):
                count += 1
                _walk(parent.child(r))

        _walk(self._tree_model.invisibleRootItem())
        return count

    @property
    def tree_group_count(self) -> int:
        """Number of top-level group nodes in the tree."""
        return self._tree_model.invisibleRootItem().rowCount()

    # -------------------------------------------------------------- Internal

    def _populate_tree(self, model: C3DDataModel) -> None:
        raw = model.raw_parameters
        # Fall back to the flat metadata dict if raw parameters are absent.
        if not isinstance(raw, dict) or not raw:
            for key, value in model.metadata.items():
                row = [QtGui.QStandardItem(str(key)), QtGui.QStandardItem(str(value))]
                for item in row:
                    item.setEditable(False)
                self._tree_model.invisibleRootItem().appendRow(row)
            return

        # Elevated groups first, then everything else alphabetically.
        priority = list(_ELEVATED_GROUPS_PRIORITY)
        ordered = [g for g in priority if g in raw]
        ordered += sorted(g for g in raw if g not in priority)

        for group_name in ordered:
            group = raw.get(group_name)
            if not isinstance(group, dict):
                continue
            group_item = QtGui.QStandardItem(str(group_name))
            group_item.setEditable(False)
            group_value = QtGui.QStandardItem("")
            group_value.setEditable(False)
            self._tree_model.invisibleRootItem().appendRow([group_item, group_value])
            for key in sorted(group.keys()):
                if _is_metadata_internal_key(key):
                    continue
                node = group[key]
                key_item = QtGui.QStandardItem(str(key))
                val_item = QtGui.QStandardItem(_format_value(node))
                key_item.setEditable(False)
                val_item.setEditable(False)
                # If the value is a list with multiple items, attach children.
                raw_val = _scalar_value(node)
                if isinstance(raw_val, np.ndarray) and raw_val.ndim >= 1:
                    flat = raw_val.ravel().tolist()
                    for i, v in enumerate(flat):
                        ck = QtGui.QStandardItem(f"[{i}]")
                        cv = QtGui.QStandardItem(repr(v))
                        ck.setEditable(False)
                        cv.setEditable(False)
                        key_item.appendRow([ck, cv])
                elif isinstance(raw_val, list) and len(raw_val) > 1:
                    for i, v in enumerate(raw_val):
                        ck = QtGui.QStandardItem(f"[{i}]")
                        cv = QtGui.QStandardItem(repr(v))
                        ck.setEditable(False)
                        cv.setEditable(False)
                        key_item.appendRow([ck, cv])
                group_item.appendRow([key_item, val_item])

    # -------------------------------------------------------- Context menu

    def _on_context_menu(self, point: QtCore.QPoint) -> None:
        index = self.tree.indexAt(point)
        if not index.isValid():
            return
        row = index.row()
        parent = index.parent()
        key_item = self._tree_model.itemFromIndex(
            self._tree_model.index(row, 0, parent)
        )
        val_item = self._tree_model.itemFromIndex(
            self._tree_model.index(row, 1, parent)
        )
        key = key_item.text() if key_item is not None else ""
        value = val_item.text() if val_item is not None else ""
        menu = QtWidgets.QMenu(self.tree)
        act_value = menu.addAction("Copy value")
        act_pair = menu.addAction("Copy key=value")
        viewport = self.tree.viewport()
        if viewport is None:
            return
        chosen = menu.exec(viewport.mapToGlobal(point))
        clipboard = QtWidgets.QApplication.clipboard()
        if clipboard is None or chosen is None:
            return
        if chosen is act_value:
            clipboard.setText(value)
        elif chosen is act_pair:
            clipboard.setText(f"{key}={value}")


__all__ = ["OverviewTab"]
