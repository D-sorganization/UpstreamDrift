"""Export through existing frozen capture/TRC providers in a separate process."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.shared.python.motion_matching import tour_capture_contract


def export_capture(capture_path: Path, output: Path) -> dict[str, object]:
    """Retain original source identity and measure existing TRC rounding bounds."""
    capture = tour_capture_contract.load_tour_capture(capture_path)
    from src.engines.physics_engines.opensim.python.tour_matching.trc import (
        read_trc,
        write_trc,
    )

    write_trc(capture, output, rate_hz=capture.rate_hz)
    restored = read_trc(output)
    if restored.labels != capture.labels or not np.array_equal(
        restored.valid, capture.valid
    ):
        raise ValueError("TRC export changed label order or missingness")
    clock_error = float(np.max(np.abs(restored.time_s - capture.time_s)))
    position_error = float(
        np.max(
            np.abs(restored.points_m[capture.valid] - capture.points_m[capture.valid])
        )
    )
    if clock_error > 5.1e-10 or position_error > 5.1e-7:
        raise ValueError("TRC rounding exceeded its declared serialization precision")
    return {
        "capture_sha256": capture.source_sha256,
        "trc_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "loader_sha256": hashlib.sha256(
            Path(tour_capture_contract.__file__).read_bytes()
        ).hexdigest(),
        "max_clock_rounding_s": clock_error,
        "max_position_rounding_m": position_error,
        "source_clock_policy": "arange(frozen_frames)/frozen_rate_hz",
    }


def main() -> None:
    """Write only summary metadata to stdout; marker arrays stay in the TRC file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(export_capture(args.capture, args.output)))


if __name__ == "__main__":
    main()
