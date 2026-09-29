# ruff: noqa: E501
import os
import subprocess
import sys
import webbrowser

from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QTabWidget,
    QWidget,
)

from src.shared.python.theme.integration import ThemedWindowMixin

from ..psa_model import PSAModel
from .input_panel import InputPanel
from .pfd_widget import PFDWidget
from .results_panel import ResultsPanel
from .sensitivity_plot import SensitivityPlotWidget

__all__ = [
    "PSAMainWindow",
]


class PSAMainWindow(ThemedWindowMixin, QMainWindow):
    """Main window for PSA analysis application."""

    def __init__(self) -> None:
        super().__init__()
        self.setup_theme_support()
        self.setWindowTitle("Two-Stage PSA System Analysis")
        self.setMinimumSize(1400, 900)
        self._setup_menu()
        self._setup_ui()
        self._connect_signals()

    def _setup_menu(self) -> None:
        """Setup the menu bar with launch options."""
        menubar = self.menuBar()
        if menubar is None:
            return

        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        if tools_menu is None:
            return

        # Launch Jupyter Notebook
        notebook_action = QAction("Open Jupyter Notebook", self)
        notebook_action.setShortcut("Ctrl+J")
        notebook_action.triggered.connect(self._launch_jupyter)
        tools_menu.addAction(notebook_action)

        # Launch Colab
        colab_action = QAction("Open in Google Colab", self)
        colab_action.setShortcut("Ctrl+G")
        colab_action.triggered.connect(self._launch_colab)
        tools_menu.addAction(colab_action)

        tools_menu.addSeparator()

        # Launch Web App
        webapp_action = QAction("Launch Web App (Streamlit)", self)
        webapp_action.setShortcut("Ctrl+W")
        webapp_action.triggered.connect(self._launch_webapp)
        tools_menu.addAction(webapp_action)

        # Help menu
        help_menu = menubar.addMenu("Help")
        if help_menu is None:
            return

        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _launch_jupyter(self) -> None:
        """Launch the Jupyter notebook."""
        # Note: we need to go up one level since we are in ui/
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        notebook_path = os.path.join(script_dir, "psa_analysis.ipynb")

        if os.path.exists(notebook_path):
            try:
                if sys.platform == "win32":
                    subprocess.Popen(
                        ["jupyter", "notebook", notebook_path],
                        creationflags=subprocess.CREATE_NEW_CONSOLE,
                    )
                else:
                    subprocess.Popen(["jupyter", "notebook", notebook_path])
                QMessageBox.information(
                    self, "Jupyter Notebook", "Launching Jupyter Notebook..."
                )
            except FileNotFoundError:
                QMessageBox.warning(
                    self,
                    "Jupyter Not Found",
                    "Jupyter is not installed. Install with: pip install jupyter",
                )
        else:
            QMessageBox.warning(
                self, "File Not Found", f"Notebook not found: {notebook_path}"
            )

    def _launch_colab(self) -> None:
        """Open the Colab-compatible notebook in Google Colab."""
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_notebook = os.path.join(script_dir, "psa_analysis_colab.ipynb")

        msg = QMessageBox(self)
        msg.setWindowTitle("Open in Google Colab")
        msg.setText("To use Google Colab, you need to upload the notebook to GitHub.")
        msg.setInformativeText(
            f"Local notebook location:\n{local_notebook}\n\n"
            "Options:\n"
            "1. Upload to GitHub and update the Colab URL\n"
            "2. Upload directly to Google Drive and open in Colab\n"
            "3. Copy the notebook content manually"
        )
        msg.setStandardButtons(
            QMessageBox.StandardButton.Open | QMessageBox.StandardButton.Cancel
        )
        msg.setDefaultButton(QMessageBox.StandardButton.Open)

        if msg.exec() == QMessageBox.StandardButton.Open:
            webbrowser.open("https://colab.research.google.com/")

    def _launch_webapp(self) -> None:
        """Launch the Streamlit web app."""
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        webapp_path = os.path.join(script_dir, "psa_webapp.py")

        if os.path.exists(webapp_path):
            try:
                if sys.platform == "win32":
                    subprocess.Popen(
                        ["streamlit", "run", webapp_path],
                        creationflags=subprocess.CREATE_NEW_CONSOLE,
                    )
                else:
                    subprocess.Popen(["streamlit", "run", webapp_path])
                QMessageBox.information(
                    self,
                    "Web App",
                    "Launching Streamlit web app...\n\n"
                    "The app will open in your default browser.",
                )
            except FileNotFoundError:
                QMessageBox.warning(
                    self,
                    "Streamlit Not Found",
                    "Streamlit is not installed. Install with: pip install streamlit",
                )
        else:
            QMessageBox.warning(
                self, "File Not Found", f"Web app not found: {webapp_path}"
            )

    def _show_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About PSA System Analysis",
            "<h3>Two-Stage PSA System Analysis</h3>"
            "<p>Version 1.0</p>"
            "<p>A comprehensive tool for analyzing pressure swing adsorption systems.</p>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>Mass balance calculations</li>"
            "<li>Sensitivity analysis</li>"
            "<li>O2 safety analysis</li>"
            "<li>Interactive plots</li>"
            "</ul>"
            "<p>All calculations validated against Excel reference model.</p>",
        )

    def _setup_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # Left panel - Inputs
        self.input_panel = InputPanel()
        self.input_panel.setMaximumWidth(400)
        main_layout.addWidget(self.input_panel)

        # Right panel - Tabs for different views
        self.tab_widget = QTabWidget()

        # Results tab
        results_scroll = QScrollArea()
        results_scroll.setWidgetResizable(True)
        self.results_panel = ResultsPanel()
        results_scroll.setWidget(self.results_panel)
        self.tab_widget.addTab(results_scroll, "Results")

        # Sensitivity Analysis tab
        self.sensitivity_widget = SensitivityPlotWidget()
        self.tab_widget.addTab(self.sensitivity_widget, "Sensitivity Analysis")

        # PFD tab
        pfd_scroll = QScrollArea()
        pfd_scroll.setWidgetResizable(True)
        self.pfd_widget = PFDWidget()
        pfd_scroll.setWidget(self.pfd_widget)
        self.tab_widget.addTab(pfd_scroll, "Process Flow Diagram")

        main_layout.addWidget(self.tab_widget, stretch=1)

        # Run initial calculation
        self._calculate()

    def _connect_signals(self) -> None:
        """Connect UI signals to slots."""
        self.input_panel.input_changed.connect(self._on_input_change)

        # Tab change triggers plot pre-calculation
        self.tab_widget.currentChanged.connect(self._on_tab_change)

    def _on_input_change(self) -> None:
        """Handle any input value changes - auto-calculate."""
        self._calculate()

    def _on_tab_change(self, index: int) -> None:
        """Pre-calculate plots when switching to the sensitivity tab.

        Resolve the tab position dynamically (issue #3105 F2) — the previous
        hard-coded ``index == 1`` silently broke the trigger if tabs were
        reordered.
        """
        if index == self.tab_widget.indexOf(self.sensitivity_widget):
            self.sensitivity_widget._update_plot()

    def _calculate(self) -> None:
        """Run PSA calculation with current inputs."""
        try:
            total_feed, s2_recycle, prod_recycle, components = (
                self.input_panel.get_parameters()
            )

            model = PSAModel(
                total_feed_scfm=total_feed,
                s2_tail_recycle_frac=s2_recycle,
                product_recycle_frac=prod_recycle,
                components=components,
            )

            results = model.calculate()
            self.results_panel.update_results(results)

            # Update sensitivity widget with current components
            self.sensitivity_widget.set_components(components)

        except ValueError as e:
            QMessageBox.warning(self, "Input Error", f"Invalid input: {e}")
        except (RuntimeError, AttributeError) as e:
            QMessageBox.critical(self, "Calculation Error", f"Error: {e}")
