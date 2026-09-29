from collections.abc import Callable

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..psa_model import DEFAULT_COMPONENTS, ComponentData

__all__ = [
    "InputPanel",
    "create_slider",
]


def create_slider(
    min_value: int,
    max_value: int,
    default_value: int,
    orientation: Qt.Orientation,
    value_changed_callback: Callable[[int], None] | None = None,
) -> QSlider:
    slider = QSlider(orientation)
    slider.setRange(min_value, max_value)
    slider.setValue(default_value)
    if value_changed_callback:
        slider.valueChanged.connect(value_changed_callback)
    return slider


class InputPanel(QWidget):
    """Panel for PSA model input parameters."""

    input_changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Operating Parameters Group
        op_group = QGroupBox("Operating Parameters")
        op_layout = QGridLayout()

        # Total Feed
        op_layout.addWidget(QLabel("Total Feed (SCFM):"), 0, 0)
        self.feed_input = QLineEdit("1100")
        self.feed_input.setValidator(QDoubleValidator(0, 100000, 2))
        op_layout.addWidget(self.feed_input, 0, 1)

        # S2 Tail Recycle
        op_layout.addWidget(QLabel("S2 Tail Recycle (%):"), 1, 0)
        self.s2_recycle_slider = create_slider(
            min_value=0,
            max_value=100,
            default_value=100,
            orientation=Qt.Orientation.Horizontal,
            value_changed_callback=lambda v: self.s2_recycle_label.setText(f"{v}%"),
        )
        self.s2_recycle_label = QLabel("100%")
        op_layout.addWidget(self.s2_recycle_slider, 1, 1)
        op_layout.addWidget(self.s2_recycle_label, 1, 2)

        # Product Recycle
        op_layout.addWidget(QLabel("Product Recycle (%):"), 2, 0)
        self.prod_recycle_slider = create_slider(
            min_value=0,
            max_value=100,
            default_value=0,
            orientation=Qt.Orientation.Horizontal,
            value_changed_callback=lambda v: self.prod_recycle_label.setText(f"{v}%"),
        )
        self.prod_recycle_label = QLabel("0%")
        op_layout.addWidget(self.prod_recycle_slider, 2, 1)
        op_layout.addWidget(self.prod_recycle_label, 2, 2)

        op_group.setLayout(op_layout)
        layout.addWidget(op_group)

        # Component Data Group
        comp_group = QGroupBox("Component Data (Feed % | S1 Removal % | S2 Removal %)")
        comp_layout = QVBoxLayout()

        self.component_table = QTableWidget(7, 4)
        self.component_table.setHorizontalHeaderLabels(
            ["Component", "Feed %", "S1 Removal %", "S2 Removal %"]
        )
        header = self.component_table.verticalHeader()
        if header is not None:
            header.setVisible(False)

        for i, comp in enumerate(DEFAULT_COMPONENTS):
            self.component_table.setItem(i, 0, QTableWidgetItem(comp["name"]))
            self.component_table.setItem(i, 1, QTableWidgetItem(str(comp["feed_pct"])))
            self.component_table.setItem(
                i, 2, QTableWidgetItem(str(comp["stage1_removal_pct"]))
            )
            self.component_table.setItem(
                i, 3, QTableWidgetItem(str(comp["stage2_removal_pct"]))
            )

        self.component_table.resizeColumnsToContents()
        comp_layout.addWidget(self.component_table)

        comp_group.setLayout(comp_layout)
        layout.addWidget(comp_group)

        # Reset Button
        self.reset_button = QPushButton("Reset to Defaults")
        layout.addWidget(self.reset_button)
        self.reset_button.clicked.connect(self._reset_defaults)

        layout.addStretch()

        # Connect input changes to the panel-level change contract.
        self.feed_input.textChanged.connect(self._on_input_change)
        self.s2_recycle_slider.valueChanged.connect(self._on_input_change)
        self.prod_recycle_slider.valueChanged.connect(self._on_input_change)
        self.component_table.cellChanged.connect(self._on_input_change)

    def _on_input_change(self) -> None:
        """Signal that inputs have changed - emitted for auto-calculate."""
        self.input_changed.emit()

    def _reset_defaults(self) -> None:
        """Reset all inputs to default values."""
        self.feed_input.setText("1100")
        self.s2_recycle_slider.setValue(100)
        self.prod_recycle_slider.setValue(0)

        for i, comp in enumerate(DEFAULT_COMPONENTS):
            self.component_table.setItem(i, 1, QTableWidgetItem(str(comp["feed_pct"])))
            self.component_table.setItem(
                i, 2, QTableWidgetItem(str(comp["stage1_removal_pct"]))
            )
            self.component_table.setItem(
                i, 3, QTableWidgetItem(str(comp["stage2_removal_pct"]))
            )

    def get_parameters(self) -> tuple[float, float, float, list[ComponentData]]:
        """Get current input parameters."""
        total_feed = float(self.feed_input.text())
        s2_recycle = self.s2_recycle_slider.value() / 100.0
        prod_recycle = self.prod_recycle_slider.value() / 100.0

        components: list[ComponentData] = []
        for i in range(7):
            name_item = self.component_table.item(i, 0)
            feed_item = self.component_table.item(i, 1)
            s1_item = self.component_table.item(i, 2)
            s2_item = self.component_table.item(i, 3)

            if name_item and feed_item and s1_item and s2_item:
                components.append(
                    {
                        "name": name_item.text(),
                        "feed_pct": float(feed_item.text()),
                        "stage1_removal_pct": float(s1_item.text()),
                        "stage2_removal_pct": float(s2_item.text()),
                    }
                )

        return total_feed, s2_recycle, prod_recycle, components
