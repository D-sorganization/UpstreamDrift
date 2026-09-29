from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPixmap
from PyQt6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

__all__ = [
    "PFDWidget",
]


class PFDWidget(QWidget):
    """Widget for displaying the Process Flow Diagram."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Process Flow Diagram")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Image label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Try to load the PFD image using a robust path resolution
        try:
            from pathlib import Path

            # Use pathlib for cleaner path handling
            script_path = Path(__file__).resolve()
            # The image is actually in the parent directory of 'ui'
            image_path = script_path.parent.parent / "PSA System PFD.jpg"

            if image_path.exists():
                pixmap = QPixmap(str(image_path))
                if not pixmap.isNull():
                    scaled = pixmap.scaled(
                        800,
                        600,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                    self.image_label.setPixmap(scaled)
                else:
                    self.image_label.setText("PFD image could not be loaded")
            else:
                self.image_label.setText(f"PFD image not found at: {image_path}")
        except (PermissionError, OSError) as e:
            self.image_label.setText(f"Error loading PFD: {e}")

        layout.addWidget(self.image_label)

        # Stream legend
        legend_group = QGroupBox("Stream Legend")
        legend_layout = QGridLayout()

        streams = [
            ("1", "Fresh Feed (from gasifier)"),
            ("2", "Exhaust (PSA 1 tail)"),
            ("3G", "Gross Product (PSA 2 output)"),
            ("3N", "Net Product (final product)"),
            ("3R", "Product Recycle"),
            ("4", "Stage 2 Tail Recycle"),
            ("5A/5B", "Mixed Feed"),
            ("6", "Interstage (PSA 1 to PSA 2)"),
        ]

        for i, (num, desc) in enumerate(streams):
            row = i // 2
            col = (i % 2) * 2
            legend_layout.addWidget(QLabel(f"<b>{num}:</b>"), row, col)
            legend_layout.addWidget(QLabel(desc), row, col + 1)

        legend_group.setLayout(legend_layout)
        layout.addWidget(legend_group)
