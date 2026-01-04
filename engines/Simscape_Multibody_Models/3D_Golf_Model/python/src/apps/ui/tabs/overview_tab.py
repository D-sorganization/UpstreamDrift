from PyQt6 import QtWidgets
from ...core.models import C3DDataModel

class OverviewTab(QtWidgets.QWidget):
    """Overview tab showing file metadata."""

    def __init__(self) -> None:
        super().__init__()
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        # File info
        self.label_file = QtWidgets.QLabel("No file loaded")
        self.label_file.setWordWrap(True)
        self.label_file.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.label_file)

        # Basic info table
        self.table_metadata = QtWidgets.QTableWidget()
        self.table_metadata.setColumnCount(2)
        self.table_metadata.setHorizontalHeaderLabels(["Field", "Value"])
        header = self.table_metadata.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(
                0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
            )
            header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table_metadata)

    def update_from_model(self, model: C3DDataModel | None) -> None:
        """Update UI with data from the model."""
        if model is None:
            self.label_file.setText("No file loaded")
            self.table_metadata.setRowCount(0)
            return

        self.label_file.setText(f"Loaded file: {model.filepath}")
        
        self.table_metadata.setRowCount(0)
        for key, value in model.metadata.items():
            row = self.table_metadata.rowCount()
            self.table_metadata.insertRow(row)
            item_key = QtWidgets.QTableWidgetItem(key)
            item_value = QtWidgets.QTableWidgetItem(str(value))
            self.table_metadata.setItem(row, 0, item_key)
            self.table_metadata.setItem(row, 1, item_value)
