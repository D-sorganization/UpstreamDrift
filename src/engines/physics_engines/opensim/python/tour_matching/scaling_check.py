"""OpenSim segment scaling check with metrics JSON and RUN manifest (MATCH #14444).

Runs ``estimate_segment_scales`` on a tour capture and gates each segment on the
#13991 length-plausibility acceptance metric: the scale factor must sit inside an
anthropometric range and the segment must stay rigid over the address frames.
Writes ``scaling_check_metrics.json`` and a ``run_manifest.json`` that validates
against ``src.sim_run.manifest`` (run-manifest-v1). The command-line entry point
is ``scripts/opensim_scaling_check.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.engines.physics_engines.opensim.python.tour_matching.scale import (
    estimate_segment_scales,
)
from src.shared.python.motion_matching.tour_capture_contract import TourCapture
from src.sim_run.manifest import (
    RUN_MANIFEST_SCHEMA_VERSION,
    SHA256_PATTERN,
    validate_run_manifest,
)

SCALING_CHECK_VERSION: Final = "1.0.0"
ACCEPTANCE_REFERENCE: Final = "#13991"
METRICS_NAME = "scaling_check_metrics.json"
MANIFEST_NAME = "run_manifest.json"
MODEL_ID = "opensim/rajagopal_nominal_segment_lengths"

# Adult stature spans roughly 0.88-1.14 of the 1.70 m Rajagopal subject; the
# extra margin allows for segment proportions. OS-9's humerus 1.46 falls outside.
SCALE_FACTOR_MIN = 0.85
SCALE_FACTOR_MAX = 1.15
# A rigid segment's marker distance should vary by under 1 cm during the address
# hold; more means the markers slide on skin or are mislabelled.
RIGID_RESIDUAL_MAX_M = 0.010


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ScalingBounds(_Strict):
    """Plausibility bounds applied to every segment."""

    scale_factor_min: float
    scale_factor_max: float
    rigid_residual_max_m: float


class SegmentCheck(_Strict):
    """Measured scale of one segment and whether it passes the gate."""

    scale_factor: float
    measured_length_m: float = Field(..., gt=0)
    nominal_length_m: float = Field(..., gt=0)
    rigid_residual_m: float = Field(..., ge=0)
    provenance: str
    plausible: bool
    reasons: list[str]


class ScalingCheckMetrics(_Strict):
    """Versioned output of one scaling check."""

    schema_version: Literal["1.0.0"]
    acceptance_reference: Literal["#13991"]
    capture_sha256: str = Field(..., pattern=SHA256_PATTERN)
    bounds: ScalingBounds
    segments: dict[str, SegmentCheck]
    verdict: Literal["pass", "fail"]
    failed_segments: list[str]


DEFAULT_BOUNDS = ScalingBounds(
    scale_factor_min=SCALE_FACTOR_MIN,
    scale_factor_max=SCALE_FACTOR_MAX,
    rigid_residual_max_m=RIGID_RESIDUAL_MAX_M,
)


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _reasons(scale: float, residual: float, bounds: ScalingBounds) -> list[str]:
    reasons = []
    if not bounds.scale_factor_min <= scale <= bounds.scale_factor_max:
        reasons.append(
            f"scale_factor {scale:.3f} outside "
            f"[{bounds.scale_factor_min}, {bounds.scale_factor_max}]"
        )
    if residual > bounds.rigid_residual_max_m:
        reasons.append(
            f"rigid_residual {residual:.4f} m exceeds {bounds.rigid_residual_max_m} m"
        )
    return reasons


def run_scaling_check(
    capture: TourCapture,
    *,
    capture_sha256: str,
    nominal_lengths_m: Mapping[str, float] | None = None,
    bounds: ScalingBounds = DEFAULT_BOUNDS,
) -> ScalingCheckMetrics:
    """Estimate segment scales and gate them on the #13991 plausibility bounds.

    Raises:
        ValueError: if ``capture_sha256`` is not a 64-char hex digest, or the
            capture is unusable for scale estimation.

    Postcondition: ``verdict == "pass"`` exactly when ``failed_segments`` is empty.
    """
    if not re.fullmatch(SHA256_PATTERN, capture_sha256):
        raise ValueError("capture_sha256 must be a 64-char lowercase hex digest")
    result = estimate_segment_scales(capture, nominal_lengths_m=nominal_lengths_m)
    segments = {}
    for name, scale in result.scale_factors.items():
        residual = result.rigid_residuals_m[name]
        reasons = _reasons(scale, residual, bounds)
        segments[name] = SegmentCheck(
            scale_factor=scale,
            measured_length_m=result.measured_lengths_m[name],
            nominal_length_m=result.nominal_lengths_m[name],
            rigid_residual_m=residual,
            provenance=result.provenance[name],
            plausible=not reasons,
            reasons=reasons,
        )
    failed = sorted(name for name, check in segments.items() if not check.plausible)
    return ScalingCheckMetrics(
        schema_version=SCALING_CHECK_VERSION,
        acceptance_reference=ACCEPTANCE_REFERENCE,
        capture_sha256=capture_sha256,
        bounds=bounds,
        segments=segments,
        verdict="fail" if failed else "pass",
        failed_segments=failed,
    )


def _nominal_revision(metrics: ScalingCheckMetrics) -> str:
    nominal = {name: s.nominal_length_m for name, s in metrics.segments.items()}
    payload = json.dumps(nominal, sort_keys=True).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def write_scaling_check(
    metrics: ScalingCheckMetrics,
    out_dir: Path,
    *,
    ud_commit: str,
    created_at: str | None = None,
) -> tuple[Path, Path]:
    """Write the metrics JSON and a validated RUN manifest; return both paths.

    Raises:
        RunManifestError: if ``ud_commit`` or ``created_at`` is invalid; nothing
            is left half-written because the manifest is validated first.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_text = json.dumps(metrics.model_dump(mode="json"), indent=2) + "\n"
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "model": {"model_id": MODEL_ID, "revision": _nominal_revision(metrics)},
        "parameters": {
            "check": "opensim_segment_scaling",
            "acceptance_reference": ACCEPTANCE_REFERENCE,
            "bounds": metrics.bounds.model_dump(),
            "verdict": metrics.verdict,
            "metrics_file": METRICS_NAME,
            "metrics_sha256": hashlib.sha256(metrics_text.encode("utf-8")).hexdigest(),
        },
        "input_data": {"sha256": metrics.capture_sha256, "label": "tour capture"},
        "seed": 0,
        "engine": {
            "name": "ud-segment-scale-estimator",
            "version": SCALING_CHECK_VERSION,
        },
        "ud_commit": ud_commit,
        "created_at": created_at
        or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    validate_run_manifest(manifest)
    metrics_path = out_dir / METRICS_NAME
    manifest_path = out_dir / MANIFEST_NAME
    metrics_path.write_text(metrics_text, encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return metrics_path, manifest_path
