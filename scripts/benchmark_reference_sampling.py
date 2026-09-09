"""Compare per-frame reference sampling with a full-trajectory workload control.

Run as ``python3 -m scripts.benchmark_reference_sampling --output report.json``.
Timing is diagnostic, not a portable CI threshold or whole-application speedup.
"""

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from scripts.benchmark_capture_responsiveness import measure
from src.motion_capture.provenance import sha256_of
from src.motion_capture.reference import ReferenceMotion, ReferenceRegistration
from src.motion_capture.reference.registration import (
    sample_reference_motion,
    transform_reference_motion,
)


def _measure_sampling(
    asset: ReferenceMotion, registration: ReferenceRegistration, samples: int
) -> dict[str, object]:
    return {
        "sample_one_frame": measure(
            lambda i: sample_reference_motion(
                asset, registration, np.array([(i + 0.5) / 120])
            ),
            samples,
        ),
        "full_trajectory_control": measure(
            lambda _: transform_reference_motion(asset, registration), samples
        ),
    }


def benchmark(samples: int = 5) -> dict[str, object]:
    if not 1 <= samples <= 30:
        raise ValueError("Use between 1 and 30 benchmark samples")
    rows = []
    for frames in (120, 1200, 12000):
        joints = 17
        asset = ReferenceMotion.model_validate(
            {
                "title": "Synthetic sampling workload",
                "source": {
                    "path": "synthetic",
                    "sha256": "0" * 64,
                    "format": "body_target_json_v1",
                },
                "source_units": "m",
                "source_axes": ("+X", "+Y", "+Z"),
                "source_names": tuple(f"joint-{i}" for i in range(joints)),
                "joint_names": tuple(f"joint-{i}" for i in range(joints)),
                "time_s": tuple(i / 120 for i in range(frames)),
                "points_m": tuple(
                    tuple((i / 120, j / 10, 1) for j in range(joints))
                    for i in range(frames)
                ),
            }
        )
        registration = ReferenceRegistration(
            reference_id=asset.id, calibration_id="unavailable"
        )
        rows.append(
            {
                "frames": frames,
                "joints": joints,
                **_measure_sampling(asset, registration, samples),
            }
        )

    source = (
        Path(__file__).resolve().parents[1]
        / "src/motion_capture/reference/registration.py"
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "registration_sha256": sha256_of(source),
        "workloads": rows,
        "limit": "Synthetic CPU sampling only; excludes loading, decoding, drawing and display.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(benchmark(), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
