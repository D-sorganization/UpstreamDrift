from __future__ import annotations

from typing import Any

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)


class AttachmentPointSelector(QDialog):
    """Dialog for selecting and configuring attachment point."""

    def __init__(
        self,
        available_links: list[str],
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the dialog."""
        if available_links is None:
            raise ValueError("available_links must be provided")
        super().__init__(parent)
        self.setWindowTitle("Select Attachment Point")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        layout.addWidget(self._create_attachment_config(available_links))
        layout.addWidget(self._create_position_offset())
        layout.addWidget(self._create_orientation())
        layout.addWidget(self._create_naming())

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _create_attachment_config(self, available_links: list[str]) -> QGroupBox:
        """Create attachment configuration group."""
        if available_links is None:
            raise ValueError("available_links must be provided")
        attach_group = QGroupBox("Attachment Configuration")
        attach_layout = QFormLayout(attach_group)

        self.link_combo = QComboBox()
        self.link_combo.addItems(available_links)
        attach_layout.addRow("Attach to link:", self.link_combo)

        self.joint_type_combo = QComboBox()
        self.joint_type_combo.addItems(["fixed", "revolute", "prismatic", "continuous"])
        attach_layout.addRow("Joint type:", self.joint_type_combo)

        return attach_group

    def _create_position_offset(self) -> QGroupBox:
        """Create position offset group."""
        offset_group = QGroupBox("Position Offset")
        offset_layout = QFormLayout(offset_group)

        self.offset_x = QDoubleSpinBox()
        self.offset_x.setRange(-10, 10)
        self.offset_x.setValue(0)
        self.offset_x.setSuffix(" m")
        offset_layout.addRow("X:", self.offset_x)

        self.offset_y = QDoubleSpinBox()
        self.offset_y.setRange(-10, 10)
        self.offset_y.setValue(0)
        self.offset_y.setSuffix(" m")
        offset_layout.addRow("Y:", self.offset_y)

        self.offset_z = QDoubleSpinBox()
        self.offset_z.setRange(-10, 10)
        self.offset_z.setValue(0.1)
        self.offset_z.setSuffix(" m")
        offset_layout.addRow("Z:", self.offset_z)

        return offset_group

    def _create_orientation(self) -> QGroupBox:
        """Create orientation group."""
        orient_group = QGroupBox("Orientation (RPY)")
        orient_layout = QFormLayout(orient_group)

        self.roll = QDoubleSpinBox()
        self.roll.setRange(-3.15, 3.15)
        self.roll.setValue(0)
        self.roll.setSuffix(" rad")
        orient_layout.addRow("Roll:", self.roll)

        self.pitch = QDoubleSpinBox()
        self.pitch.setRange(-3.15, 3.15)
        self.pitch.setValue(0)
        self.pitch.setSuffix(" rad")
        orient_layout.addRow("Pitch:", self.pitch)

        self.yaw = QDoubleSpinBox()
        self.yaw.setRange(-3.15, 3.15)
        self.yaw.setValue(0)
        self.yaw.setSuffix(" rad")
        orient_layout.addRow("Yaw:", self.yaw)

        return orient_group

    def _create_naming(self) -> QGroupBox:
        """Create naming group."""
        prefix_group = QGroupBox("Naming")
        prefix_layout = QFormLayout(prefix_group)

        self.prefix_edit = QLineEdit()
        self.prefix_edit.setPlaceholderText("optional prefix for link/joint names")
        prefix_layout.addRow("Name prefix:", self.prefix_edit)

        return prefix_group

    def get_configuration(self) -> dict[str, Any]:
        """Get the attachment configuration."""
        return {
            "parent_link": self.link_combo.currentText(),
            "joint_type": self.joint_type_combo.currentText(),
            "offset": (
                self.offset_x.value(),
                self.offset_y.value(),
                self.offset_z.value(),
            ),
            "orientation": (self.roll.value(), self.pitch.value(), self.yaw.value()),
            "name_prefix": self.prefix_edit.text(),
        }
