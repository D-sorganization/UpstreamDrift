"""Background worker for loading C3D files asynchronously."""

from PyQt6.QtCore import QThread, pyqtSignal

# C3DDataModel is used in type hinting potentially, but signal registration uses object
# We keep import if used in type hints, otherwise remove.
# Since it is not used in code body, we can remove it or use TYPE_CHECKING
from ..core.models import C3DDataModel  # noqa: F401
from .c3d_loader import load_c3d_file


class C3DLoaderThread(QThread):
    """
    Worker thread to load C3D files without blocking the GUI.

    Signals:
        loaded (C3DDataModel): Emitted when loading is successful.
        failed (str): Emitted with error message if loading fails.
    """

    loaded = pyqtSignal(object)  # C3DDataModel
    failed = pyqtSignal(str)

    def __init__(self, filepath: str) -> None:
        """Initialize the loader thread.

        Args:
            filepath: Absolute path to the C3D file.
        """
        super().__init__()
        self.filepath = filepath

    def run(self) -> None:
        """Execute the loading task."""
        try:
            model = load_c3d_file(self.filepath)
            self.loaded.emit(model)
        except Exception as e:
            self.failed.emit(str(e))
