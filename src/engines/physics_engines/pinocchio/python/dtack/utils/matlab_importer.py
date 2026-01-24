"""Python utility to import MATLAB data files (.mat, .c3d)."""

from __future__ import annotations

import typing
from pathlib import Path

from dtack.utils.gears_parser import GearsParser

from src.shared.python.logging_config import get_logger

if typing.TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

try:
    import scipy.io

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import ezc3d

    EZC3D_AVAILABLE = True
except ImportError:
    EZC3D_AVAILABLE = False


logger = get_logger(__name__)


class MATLABImporter:
    """Import MATLAB data files for use in Python workflows."""

    @staticmethod
    def load_mat(file_path: Path | str) -> dict[str, npt.NDArray[np.float64]]:
        """Load MATLAB .mat file.

        Args:
            file_path: Path to .mat file

        Returns:
            Dictionary of variable names to arrays

        Raises:
            ImportError: If scipy is not installed
            FileNotFoundError: If file does not exist
        """
        if not SCIPY_AVAILABLE:
            msg = "scipy is required for .mat import. Install with: pip install scipy"
            raise ImportError(msg)

        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            msg = f"File not found: {file_path}"
            raise FileNotFoundError(msg)

        data = scipy.io.loadmat(str(file_path_obj))
        # Remove MATLAB metadata
        cleaned = {k: v for k, v in data.items() if not k.startswith("__")}
        logger.info("Loaded %d variables from %s", len(cleaned), file_path_obj.name)
        return cleaned

    @staticmethod
    def load_c3d(file_path: Path | str) -> dict[str, npt.NDArray[np.float64]]:
        """Load C3D motion capture file.

        Args:
            file_path: Path to .c3d file

        Returns:
            Dictionary with 'markers', 'analog', 'parameters' keys

        Raises:
            ImportError: If ezc3d is not installed
            FileNotFoundError: If file does not exist
        """
        if not EZC3D_AVAILABLE:
            msg = "ezc3d is required for .c3d import. Install with: pip install ezc3d"
            raise ImportError(msg)

        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            msg = f"File not found: {file_path}"
            raise FileNotFoundError(msg)

        c3d = ezc3d.C3d(str(file_path_obj))

        # Extract marker data
        markers = {}
        for i, name in enumerate(c3d["parameters"]["POINT"]["LABELS"]["value"]):
            markers[name] = c3d["data"]["points"][:, i, :]  # [frames x 3]

        # Extract analog data if available
        analog = {}
        if "ANALOG" in c3d["parameters"]:
            for i, name in enumerate(c3d["parameters"]["ANALOG"]["LABELS"]["value"]):
                analog[name] = c3d["data"]["analogs"][i, :]  # [samples]

        result = {
            "markers": markers,
            "analog": analog,
            "frame_rate": c3d["parameters"]["POINT"]["RATE"]["value"][0],
            "first_frame": c3d["header"]["first_frame"],
            "last_frame": c3d["header"]["last_frame"],
        }

        logger.info(
            "Loaded C3D file: %d markers, %d analog channels, %.1f Hz",
            len(markers),
            len(analog),
            result["frame_rate"],
        )
        return result

    @staticmethod
    def load_gpcap(file_path: Path | str) -> dict[str, npt.NDArray[np.float64]]:
        """Load Gears capture file (.gpcap).

        Args:
            file_path: Path to .gpcap file

        Returns:
            Dictionary with capture data

        Raises:
            RuntimeError: Parser not yet implemented. File format requires reverse
                engineering.
        """
        result = GearsParser.load(file_path)
        return dict(result)  # type: ignore[arg-type]
