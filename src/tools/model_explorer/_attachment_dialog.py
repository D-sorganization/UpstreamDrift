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
        attachment_points: list[dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the dialog."""
        if available_links is None:
            raise ValueError("available_links must be provided")
        self._attachment_points = tuple(attachment_points or ())
        super().__init__(parent)
        self.setWindowTitle("Select Attachment Point")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        layout.addWidget(self._create_attachment_config(available_links))
        layout.addWidget(self._create_position_offset())
        layout.addWidget(self._create_orientation())
        layout.addWidget(self._create_naming())
        self._apply_selected_point()

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
        self._populate_link_combo(available_links)
        self.link_combo.currentIndexChanged.connect(self._apply_selected_point)
        attach_layout.addRow("Attach to link:", self.link_combo)

        self.joint_type_combo = QComboBox()
        self.joint_type_combo.addItems(["fixed", "revolute", "prismatic", "continuous"])
        attach_layout.addRow("Joint type:", self.joint_type_combo)

        return attach_group

    def _populate_link_combo(self, available_links: list[str]) -> None:
        declared_links = set()
        for point in self._attachment_points:
            link_name = str(point.get("link_name", "")).strip()
            if not link_name:
                continue
            declared_links.add(link_name)
            role = str(point.get("role", "")).strip()
            name = str(point.get("name", link_name)).strip()
            label = f"{name} ({role})" if role else name
            self.link_combo.addItem(label, point)
        for link_name in available_links:
            if link_name not in declared_links:
                self.link_combo.addItem(link_name, None)

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

    def _apply_selected_point(self) -> None:
        point = self.link_combo.currentData()
        if not isinstance(point, dict):
            return
        frame = point.get("interface_frame")
        if not isinstance(frame, dict):
            return
        xyz = _triple(frame.get("xyz"), (0.0, 0.0, 0.0))
        rpy = _triple(frame.get("rpy"), (0.0, 0.0, 0.0))
        self.offset_x.setValue(xyz[0])
        self.offset_y.setValue(xyz[1])
        self.offset_z.setValue(xyz[2])
        self.roll.setValue(rpy[0])
        self.pitch.setValue(rpy[1])
        self.yaw.setValue(rpy[2])

    def get_configuration(self) -> dict[str, Any]:
        """Get the attachment configuration."""
        point = self.link_combo.currentData()
        link_name = self.link_combo.currentText()
        attachment_point = None
        if isinstance(point, dict):
            link_name = str(point.get("link_name", link_name))
            attachment_point = str(point.get("name", ""))
        return {
            "parent_link": link_name,
            "attachment_point": attachment_point,
            "joint_type": self.joint_type_combo.currentText(),
            "offset": (
                self.offset_x.value(),
                self.offset_y.value(),
                self.offset_z.value(),
            ),
            "orientation": (self.roll.value(), self.pitch.value(), self.yaw.value()),
            "name_prefix": self.prefix_edit.text(),
        }


def declared_payload_warnings(
    attachment_points: list[dict[str, Any]],
    payload_kg: float,
) -> tuple[str, ...]:
    """Return warnings for declared mount points exceeded by a selected payload."""
    if attachment_points is None:
        raise ValueError("attachment_points must be provided")
    if payload_kg < 0:
        raise ValueError("payload_kg must be non-negative")
    warnings: list[str] = []
    for point in attachment_points:
        limit = point.get("max_payload_kg")
        if limit is None:
            continue
        try:
            max_payload_kg = float(limit)
        except (TypeError, ValueError):
            continue
        if payload_kg > max_payload_kg:
            name = point.get("name") or point.get("link_name") or "unnamed"
            warnings.append(
                f"Attachment point '{name}' payload limit is "
                f"{max_payload_kg:.1f} kg; selected payload is {payload_kg:.1f} kg."
            )
    return tuple(warnings)


def _triple(
    raw: object, default: tuple[float, float, float]
) -> tuple[float, float, float]:
    if not isinstance(raw, list | tuple) or len(raw) != 3:
        return default
    try:
        return (float(raw[0]), float(raw[1]), float(raw[2]))
    except (TypeError, ValueError):
        return default
