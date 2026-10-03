"""Extract historical footage through shared source and detector contracts."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

from src.shared.python.pose_estimation.mediapipe_estimator import MediaPipeEstimator
from src.shared.python.pose_estimation.mediapipe_models import (
    resolve_pose_model,
    sha256_of,
)
from src.shared.python.shadow_tracker.historical_capture import (
    CaptureWindow,
    export_capture,
)


def main() -> None:
    """Run one continuous candidate window; emit its unqualified receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    args = parser.parse_args()
    window = CaptureWindow(start_s=args.start, end_s=args.end)
    model = resolve_pose_model()
    estimator = MediaPipeEstimator(enable_temporal_smoothing=False)
    estimator.load_model(model)
    try:
        receipt = export_capture(
            args.source,
            args.destination,
            window,
            subject_id=args.subject,
            estimator=estimator,
            detector_identity={
                "name": "MediaPipeEstimator",
                "version": importlib.metadata.version("mediapipe"),
                "model_sha256": sha256_of(model),
                "smoothing": False,
                "min_detection_confidence": 0.5,
                "min_tracking_confidence": 0.5,
            },
        )
    finally:
        estimator.close()
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
