"""Qt widgets generated from the request schema (#15150 G6).

One collapsible group box per schema group; the widget type of each option
comes from :func:`dialog_core.widget_for`. Each group returns readers that
yield a widget-neutral value for :func:`dialog_core.collect_options`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.tools.native_viewer_export import dialog_core as core

Reader = Callable[[], Any]


def _spin(spec: core.OptionSpec) -> tuple[QWidget, Reader]:
    s = core.spin_settings()
    box: QSpinBox | QDoubleSpinBox
    if spec.widget == "int_spin":
        box = QSpinBox()
        box.setRange(int(s["spin_minimum"]), int(s["spin_maximum"]))
        start: float = box.minimum() if spec.default is None else int(spec.default)
        box.setValue(int(start))
    else:
        box = QDoubleSpinBox()
        box.setDecimals(s["spin_decimals"])
        box.setRange(float(s["spin_minimum"]), float(s["spin_maximum"]))
        start = box.minimum() if spec.default is None else float(spec.default)
        box.setValue(start)
    box.setSpecialValueText(core.load_dialog_settings()["default_label"])
    return box, lambda: None if box.value() == box.minimum() else box.value()


def _combo(spec: core.OptionSpec) -> tuple[QWidget, Reader]:
    combo = QComboBox()
    if spec.default is None:
        combo.addItem(core.load_dialog_settings()["default_label"], None)
    for choice in spec.choices:
        combo.addItem(str(choice), choice)
    combo.setCurrentIndex(max(0, combo.findData(spec.default)))
    return combo, combo.currentData


def _line(spec: core.OptionSpec) -> tuple[QWidget, Reader]:
    edit = QLineEdit("" if spec.default is None else str(spec.default))
    if spec.multi:
        edit.setPlaceholderText("comma-separated: " + ", ".join(spec.choices))
    if spec.widget != "path":
        return edit, edit.text
    row, button = QWidget(), QPushButton("Browse...")
    button.clicked.connect(
        lambda: edit.setText(
            QFileDialog.getOpenFileName(row, spec.name)[0] or edit.text()
        )
    )
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(edit, 1)
    layout.addWidget(button)
    return row, edit.text


def _checkbox(spec: core.OptionSpec) -> tuple[QWidget, Reader]:
    check = QCheckBox()
    check.setChecked(bool(spec.default))
    return check, check.isChecked


_FACTORIES = {
    "checkbox": _checkbox,
    "combo": _combo,
    "int_spin": _spin,
    "float_spin": _spin,
    "path": _line,
    "line": _line,
}


def build_group(
    name: str, specs: list[core.OptionSpec], readers: dict[str, Reader]
) -> QGroupBox:
    """A collapsed, checkable group box holding one widget per spec.

    Fills ``readers`` (option name to value reader) and sets each widget's
    object name to ``option_<name>`` so tests and tooltips can find it.
    """
    box, body, form = QGroupBox(name), QWidget(), QFormLayout()
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    for spec in specs:
        widget, readers[spec.name] = _FACTORIES[spec.widget](spec)
        widget.setObjectName(f"option_{spec.name}")
        widget.setToolTip(spec.help)
        form.addRow(spec.name.replace("_", " "), widget)
    body.setLayout(form)
    box.setCheckable(True)
    box.setChecked(False)
    body.setVisible(False)
    box.setFlat(True)
    box.toggled.connect(body.setVisible)
    box.toggled.connect(lambda on: box.setFlat(not on))
    outer = QVBoxLayout(box)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.addWidget(body)
    return box
