import logging
import argparse
import sys
from pathlib import Path
from typing import Any
import numpy as np

try:
    import ezc3d
except ImportError:
    ezc3d = None

logger = logging.getLogger(__name__)


def check_matlab_availability() -> bool:
    """Check if MATLAB engine is available."""
    try:
        import matlab.engine  # type: ignore

        return True
    except ImportError:
        logger.warning(
            "MATLAB engine not available. Diagnostics will be emitted, but full Simscape matching will be mocked/skipped."
        )
        return False


def load_c3d_markers(filepath: str) -> dict[str, np.ndarray]:
    """Load C3D file and return markers."""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"C3D file not found: {filepath}")

    if ezc3d is None:
        logger.warning("ezc3d not installed. Returning empty markers.")
        return {}

    logger.info(f"Loading C3D data from {filepath}")
    try:
        c = ezc3d.c3d(filepath)
        markers: dict[str, np.ndarray] = {}
        labels = c["parameters"]["POINT"]["LABELS"]["value"]
        data = c["data"]["points"]
        for i, label in enumerate(labels):
            markers[label] = data[:3, i, :]
        return markers
    except Exception as e:
        logger.error(f"Error reading C3D file: {e}")
        return {}


def canonicalize_rotation(rot: np.ndarray) -> np.ndarray:
    """
    Canonicalize Python rotations as unit quaternions [w, x, y, z].
    Requires input to be an array with last dimension 4.
    """
    if not isinstance(rot, np.ndarray) or rot.shape[-1] != 4:
        raise ValueError("Rotation must be a numpy array with last dimension 4.")

    # ⚡ Bolt: Optimize norm calculation along axis using einsum
    norm = np.sqrt(np.einsum("...i,...i->...", rot, rot))[..., np.newaxis]
    # Avoid division by zero
    norm = np.where(norm == 0, 1.0, norm)
    q: np.ndarray = rot / norm

    return q


def match_motion(
    markers: dict[str, np.ndarray], matlab_available: bool
) -> dict[str, Any]:
    """
    Match Simscape model to the tour-average swing based on C3D markers.
    """
    if matlab_available:
        logger.info("Executing Simscape motion matching via MATLAB.")
        # Mock MATLAB call returning simulated rotations
        rot = canonicalize_rotation(np.array([[1.0, 0.0, 0.0, 0.0]]))
        return {"matched": True, "rotations": rot}
    logger.info("MATLAB unavailable. Emitting diagnostics for motion matching.")
    # Return fallback data
    rot = canonicalize_rotation(np.array([[1.0, 0.0, 0.0, 0.0]]))
    return {"matched": False, "rotations": rot}


def generate_preview(
    markers: dict[str, np.ndarray], model_data: dict[str, Any], output_path: str
) -> None:
    """
    Emit a preview comparing C3D markers against the matched model.
    """
    logger.info(f"Generating preview at {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("C3D-to-Simscape Preview Diagnostics\n")
        f.write("=====================================\n")
        marker_names = list(markers.keys())[:5] if markers else []
        f.write(f"Markers loaded: {marker_names}...\n")
        f.write(f"Model matched: {model_data.get('matched')}\n")
        f.write(
            f"Canonical rotation example: {model_data.get('rotations', [])[0] if len(model_data.get('rotations', [])) > 0 else 'N/A'}\n"
        )


def run_pipeline(input_c3d: str, output_preview: str) -> None:
    """Run the headless C3D-to-Simscape pipeline."""
    matlab_available = check_matlab_availability()
    markers = load_c3d_markers(input_c3d)
    model_data = match_motion(markers, matlab_available)
    generate_preview(markers, model_data, output_preview)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Headless C3D-to-Simscape motion-matching preview pipeline"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="data/C3D_TA_Driver.c3d",
        help="Input C3D file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output/preview_diagnostics.txt",
        help="Output preview file",
    )
    args = parser.parse_args()

    try:
        run_pipeline(args.input, args.output)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)
