import sys
import logging
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
import pyvista as pv
from pyvistaqt import QtInteractor

from src.shared.python.physics.ball_enhanced_simulator import (
    EnhancedBallFlightSimulator,
)

# from src.shared.python.physics.terrain_engine import TerrainAwareEngine # Assuming this exists

logger = logging.getLogger(__name__)


class GolfSimulationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Golf Simulation Suite")
        self.setGeometry(100, 100, 800, 600)

        # Setup main widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Setup PyVista interactor
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)

        # Setup UI controls
        self.btn_simulate = QPushButton("Simulate Ball Flight")
        self.btn_simulate.clicked.connect(self.run_simulation)
        layout.addWidget(self.btn_simulate)

        self.btn_putting = QPushButton("Putting Green Mode")
        self.btn_putting.clicked.connect(self.run_putting_green)
        layout.addWidget(self.btn_putting)

        # Initialize simulators
        self.ball_sim = EnhancedBallFlightSimulator()

    def run_simulation(self):
        logger.info("Running golf ball simulation...")
        self.plotter.clear()

        # Dummy visualization for ball flight
        sphere = pv.Sphere(radius=0.02, center=(0, 0, 0))
        self.plotter.add_mesh(sphere, color="white")

        # Add a simple trajectory line
        points = [[0, 0, 0], [50, 0, 20], [100, 0, 0]]
        poly = pv.MultipleLines(points=points)
        self.plotter.add_mesh(poly, color="blue", line_width=2)

        self.plotter.add_axes()
        self.plotter.reset_camera()

    def run_putting_green(self):
        logger.info("Running putting green mode...")
        self.plotter.clear()

        # Dummy visualization for putting green
        plane = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=10, j_size=10)
        self.plotter.add_mesh(plane, color="green", show_edges=True)

        # Add hole and ball
        hole = pv.Cylinder(
            center=(3, 3, -0.05), direction=(0, 0, 1), radius=0.05, height=0.1
        )
        ball = pv.Sphere(radius=0.02, center=(-3, -3, 0.02))

        self.plotter.add_mesh(hole, color="black")
        self.plotter.add_mesh(ball, color="white")

        self.plotter.add_axes()
        self.plotter.reset_camera()


def get_dockable_ui() -> QMainWindow:
    """Return the main window instance for docking in the unified launcher."""
    return GolfSimulationWindow()


def main():
    logging.basicConfig(level=logging.INFO)
    app = QApplication.instance() or QApplication(sys.argv)
    window = GolfSimulationWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
