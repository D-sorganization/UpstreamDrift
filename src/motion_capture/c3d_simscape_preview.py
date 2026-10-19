import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.data_io.c3d_loader import c3d_reader_class

logger = logging.getLogger(__name__)

_MATCH_NOT_IMPLEMENTED_REASON = (
    "Simscape motion matching is not wired to the tour-average matcher yet."
)


@dataclass(frozen=True)
class C3DMarkerLoadResult:
    """Markers loaded from a C3D file, normalized to metres."""

    markers: dict[str, np.ndarray]
    n_frames: int
    source_units: str
    units: str


def check_matlab_availability() -> bool:
    """Check if MATLAB engine is available."""
    try:
        import matlab.engine  # type: ignore[import-not-found]  # noqa: F401

        return True
    except ImportError:
        logger.warning(
            "MATLAB engine not available. Diagnostics will be emitted, but full Simscape matching will be mocked/skipped."
        )
        return False


def load_c3d_markers(filepath: str) -> C3DMarkerLoadResult:
    """Load C3D marker trajectories via the shared :class:`C3DDataReader`."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"C3D file not found: {filepath}")

    logger.info("Loading C3D data from %s", filepath)
    reader_cls = c3d_reader_class()
    reader = reader_cls(path)
    metadata = reader.get_metadata()
    if metadata.marker_count == 0:
        raise ValueError(f"C3D file contains no markers: {filepath}")

    df = reader.points_dataframe(include_time=False, target_units="m")
    markers: dict[str, np.ndarray] = {}
    for label in metadata.marker_labels:
        subset = df.loc[df["marker"] == label].sort_values("frame")
        if len(subset) != metadata.frame_count:
            raise ValueError(
                f"Marker {label!r} has {len(subset)} frames; "
                f"expected {metadata.frame_count} in {filepath}"
            )
        markers[label] = np.stack(
            [
                subset["x"].to_numpy(),
                subset["y"].to_numpy(),
                subset["z"].to_numpy(),
            ],
            axis=0,
        )

    return C3DMarkerLoadResult(
        markers=markers,
        n_frames=int(metadata.frame_count),
        source_units=metadata.units,
        units="m",
    )


def canonicalize_rotation(rot: np.ndarray) -> np.ndarray:
    """
    Canonicalize Python rotations as unit quaternions [w, x, y, z].
    Requires input to be an array with last dimension 4.
    """
    if not isinstance(rot, np.ndarray) or rot.shape[-1] != 4:
        raise ValueError("Rotation must be a numpy array with last dimension 4.")

    rot_float = rot.astype(np.float64, copy=False)
    # ⚡ Bolt: Optimize norm calculation along axis using einsum
    norm = np.sqrt(np.einsum("...i,...i->...", rot_float, rot_float))[..., np.newaxis]
    # Avoid division by zero
    norm = np.where(norm == 0, 1.0, norm)
    q: np.ndarray = rot_float / norm

    return q


def match_motion(
    markers: dict[str, np.ndarray], matlab_available: bool
) -> dict[str, Any]:
    """
    Match Simscape model to the tour-average swing based on C3D markers.

    Postcondition: ``matched`` is True only after a real matcher runs (not yet wired).
    """
    if not markers:
        raise ValueError("markers must be non-empty to run motion matching preview")

    if matlab_available:
        logger.info(
            "MATLAB is available, but Simscape motion matching is not implemented yet."
        )
    else:
        logger.info("MATLAB unavailable. Emitting diagnostics for motion matching.")

    rot = canonicalize_rotation(np.array([[1.0, 0.0, 0.0, 0.0]]))
    return {
        "matched": False,
        "reason": _MATCH_NOT_IMPLEMENTED_REASON,
        "rotations": rot,
    }


def generate_preview(
    load_result: C3DMarkerLoadResult,
    model_data: dict[str, Any],
    output_path: str,
) -> None:
    """
    Emit a preview comparing C3D markers against the matched model.
    """
    logger.info("Generating preview at %s", output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    marker_names = list(load_result.markers.keys())[:5]
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("C3D-to-Simscape Preview Diagnostics\n")
        f.write("=====================================\n")
        f.write(f"Markers loaded ({len(load_result.markers)}): {marker_names}...\n")
        f.write(f"Frames: {load_result.n_frames}\n")
        f.write(
            f"Point units: {load_result.units} "
            f"(converted from {load_result.source_units})\n"
        )
        f.write(f"Model matched: {model_data.get('matched')}\n")
        if not model_data.get("matched"):
            f.write(f"Match status: {model_data.get('reason', 'not matched')}\n")
        rotations = model_data.get("rotations", [])
        example = rotations[0] if len(rotations) > 0 else "N/A"
        f.write(f"Canonical rotation example: {example}\n")


def run_pipeline(input_c3d: str, output_preview: str) -> None:
    """Run the headless C3D-to-Simscape pipeline."""
    matlab_available = check_matlab_availability()
    load_result = load_c3d_markers(input_c3d)
    model_data = match_motion(load_result.markers, matlab_available)
    generate_preview(load_result, model_data, output_preview)


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
    except Exception as e:  # noqa: BLE001
        logger.error("Pipeline failed: %s", e)
        sys.exit(1)
