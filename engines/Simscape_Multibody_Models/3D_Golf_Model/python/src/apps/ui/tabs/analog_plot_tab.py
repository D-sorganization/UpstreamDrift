from PyQt6 import QtWidgets
from ...core.models import C3DDataModel
from ..widgets.mpl_canvas import MplCanvas

class AnalogPlotTab(QtWidgets.QWidget):
    """Analog channel plotting tab."""

    def __init__(self) -> None:
        super().__init__()
        self.model: C3DDataModel | None = None
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)

        left_panel = QtWidgets.QVBoxLayout()
        self.list_analog = QtWidgets.QListWidget()
        self.list_analog.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.list_analog.itemSelectionChanged.connect(self.update_plot)
        left_panel.addWidget(QtWidgets.QLabel("Analog channels:"))
        left_panel.addWidget(self.list_analog)

        layout.addLayout(left_panel, 1)

        right_panel = QtWidgets.QVBoxLayout()
        self.canvas_analog = MplCanvas(self, width=5, height=4, dpi=100)
        right_panel.addWidget(self.canvas_analog)

        layout.addLayout(right_panel, 3)

    def update_from_model(self, model: C3DDataModel | None) -> None:
        """Update UI with data from the model."""
        self.model = model
        self.list_analog.clear()
        
        if model is None:
            self.canvas_analog.clear_axes()
            return

        for name in model.analog_names():
            self.list_analog.addItem(name)

        if model.analog_names():
            self.list_analog.setCurrentRow(0)

    def update_plot(self) -> None:
        """Update the analog plot based on selected channel."""
        if self.model is None:
            return
            
        selected_items = self.list_analog.selectedItems()
        if not selected_items:
            self.canvas_analog.clear_axes()
            return

        name = selected_items[0].text()
        channel = self.model.analog.get(name)
        if channel is None or self.model.analog_time is None:
            self.canvas_analog.clear_axes()
            return

        t = self.model.analog_time
        values = channel.values

        self.canvas_analog.fig.clear()
        ax = self.canvas_analog.add_subplot(111)
        ax.plot(t, values, label=name)
        unit = f" ({channel.unit})" if channel.unit else ""
        ax.set_ylabel(f"Value{unit}")
        ax.set_xlabel("Time (s)")
        ax.set_title(f"Analog channel: {name}")
        ax.grid(True)
        ax.legend()

        self.canvas_analog.fig.tight_layout()
        self.canvas_analog.draw()  # type: ignore
