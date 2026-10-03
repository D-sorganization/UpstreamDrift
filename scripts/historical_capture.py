"""Extract historical footage through shared source and detector contracts."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from src.shared.python.pose_estimation.interface import PoseEstimationResult
from src.shared.python.pose_estimation.mediapipe_models import (
    resolve_pose_model,
    sha256_of,
)
from src.shared.python.pose_estimation.registry import (
    EstimatorInfo,
    create_estimator,
    estimator_availability,
    get_estimator_info,
)
from src.shared.python.shadow_tracker.historical_capture import (
    CaptureWindow,
    export_capture,
)


class _ImageEstimatorAdapter:
    """Adapter bridging PoseEstimator instances to ImageEstimator protocol."""

    def __init__(self, estimator: Any) -> None:
        self._estimator = estimator

    def estimate_from_image(
        self, image: np.ndarray, timestamp_ms: int
    ) -> PoseEstimationResult:
        try:
            return self._estimator.estimate_from_image(image, timestamp_ms)
        except TypeError:
            return self._estimator.estimate_from_image(image)

    def close(self) -> None:
        if hasattr(self._estimator, "close"):
            self._estimator.close()


def _parse_options(raw_options: list[str]) -> dict[str, Any]:
    """Parse KEY=VALUE pairs into typed primitive options."""
    options: dict[str, Any] = {}
    for item in raw_options:
        if "=" not in item:
            raise ValueError(
                f"Invalid --estimator-option format: {item!r} (expected KEY=VALUE)"
            )
        key, val = item.split("=", 1)
        key = key.strip()
        val_stripped = val.strip()
        lowered = val_stripped.lower()
        if lowered == "true":
            parsed: Any = True
        elif lowered == "false":
            parsed = False
        else:
            try:
                parsed = int(val_stripped)
            except ValueError:
                try:
                    parsed = float(val_stripped)
                except ValueError:
                    parsed = val_stripped
        options[key] = parsed
    return options


def _resolve_model_hash_and_load(
    estimator_name: str,
    estimator: Any,
    options: dict[str, Any],
) -> str | None:
    """Resolve model weights without implicit downloads and return their SHA-256."""
    if hasattr(estimator, "model_sha256"):
        return str(estimator.model_sha256)
    if hasattr(estimator, "weights_hash"):
        return str(estimator.weights_hash)
    if "model_sha256" in options:
        return str(options["model_sha256"])
    if estimator_name == "mediapipe":
        variant = str(options.get("variant", "full"))
        model = resolve_pose_model(variant=variant)
        if hasattr(estimator, "load_model"):
            estimator.load_model(model)
        return sha256_of(model)
    if estimator_name == "rtmpose_onnx":
        from src.shared.python.pose_estimation.model_files import sha256_of as mf_sha256
        from src.shared.python.pose_estimation.rtmpose_models import resolve_rtmpose

        keypoint_set = str(options.get("keypoint_set", "coco17"))
        model_path = resolve_rtmpose(keypoint_set)
        return mf_sha256(model_path)
    if estimator_name == "openpose_dnn":
        from src.shared.python.pose_estimation.model_files import sha256_of as mf_sha256
        from src.shared.python.pose_estimation.openpose_models import resolve_body25

        _, weights_path = resolve_body25()
        return mf_sha256(weights_path)
    return None


def _resolve_package_version(estimator: Any, info: EstimatorInfo) -> str:
    """Resolve package version from estimator instance or registered probe module."""
    if hasattr(estimator, "version"):
        return str(estimator.version)
    pkg_name = info.probe_module
    try:
        return importlib.metadata.version(pkg_name)
    except (importlib.metadata.PackageNotFoundError, ValueError):
        try:
            mod = importlib.import_module(pkg_name)
            return str(getattr(mod, "__version__", "unknown"))
        except (ImportError, AttributeError):
            return "unknown"


def _build_detector_identity(
    estimator_name: str,
    estimator: Any,
    info: EstimatorInfo,
    options: dict[str, Any],
) -> dict[str, Any]:
    """Assemble verified detector evidence block for receipt."""
    model_sha256 = _resolve_model_hash_and_load(estimator_name, estimator, options)
    version = _resolve_package_version(estimator, info)
    min_conf = float(
        options.get("min_confidence", options.get("min_detection_confidence", 0.5))
    )
    track_conf = float(
        options.get("min_confidence", options.get("min_tracking_confidence", 0.5))
    )
    smoothing = bool(options.get("enable_temporal_smoothing", False))
    name = "MediaPipeEstimator" if estimator_name == "mediapipe" else estimator_name
    return {
        "name": name,
        "registry_name": estimator_name,
        "version": version,
        "model_sha256": model_sha256,
        "smoothing": smoothing,
        "min_detection_confidence": min_conf,
        "min_tracking_confidence": track_conf,
    }


def main(argv: list[str] | None = None) -> int:
    """Run one continuous candidate window; emit its unqualified receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--estimator", default="mediapipe")
    parser.add_argument(
        "--estimator-option",
        action="append",
        default=[],
        dest="estimator_options",
        help="Option in KEY=VALUE form passed to estimator factory",
    )
    args = parser.parse_args(argv)

    if args.destination.exists():
        sys.stderr.write(
            f"Refusing to overwrite existing destination directory: {args.destination}\n"
        )
        raise SystemExit(1)

    try:
        info = get_estimator_info(args.estimator)
    except ValueError as e:
        sys.stderr.write(f"Unknown estimator {args.estimator!r}: {e}\n")
        raise SystemExit(1) from e

    available, reason = estimator_availability(args.estimator)
    if not available:
        sys.stderr.write(f"Estimator {args.estimator!r} is unavailable: {reason}\n")
        raise SystemExit(1)

    options = _parse_options(args.estimator_options)
    if args.estimator == "mediapipe":
        options.setdefault("enable_temporal_smoothing", False)

    estimator = create_estimator(args.estimator, **options)
    detector_identity = _build_detector_identity(
        args.estimator, estimator, info, options
    )

    window = CaptureWindow(start_s=args.start, end_s=args.end)
    adapted_estimator = _ImageEstimatorAdapter(estimator)
    try:
        receipt = export_capture(
            args.source,
            args.destination,
            window,
            subject_id=args.subject,
            estimator=adapted_estimator,
            detector_identity=detector_identity,
        )
    finally:
        adapted_estimator.close()
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
