"""Immutable report snapshots; resume from a candidate, not optimizer internals."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def read_prefix_candidate(path: Path, expected: Mapping[str, Any]) -> list[float]:
    """Read a candidate for the same capture, scales and declared fit identity.

    Legacy reports remain readable together. A declared native-state identity
    cannot be silently dropped or supplied to an unidentified legacy snapshot.
    """
    payload = json.loads(path.read_bytes())
    if payload.get("schema") != "simscape-first-prefix-checkpoint/1":
        raise ValueError("incompatible checkpoint schema")
    report = payload["report"]
    if report.get("fit_identity") != expected.get("fit_identity"):
        raise ValueError("incompatible checkpoint fit_identity")
    for key in ("source_sha256", "labels", "effort_scales"):
        if key not in expected or report.get(key) != expected[key]:
            raise ValueError(f"incompatible checkpoint {key}")
    values = payload["resume_parameters"]
    if len(values) != len(expected["effort_scales"]) or any(
        not math.isfinite(value) or not 0 <= value <= 2 for value in values
    ):
        raise ValueError("incompatible checkpoint parameter bounds")
    return [float(value) for value in values]


def snapshot_prefix_report(source: Path, destination: Path) -> Path:
    """Preserve a complete first-prefix report and its best warm-start parameters.

    Reject partial/invalid reports before writing. Identical reports are a no-op;
    changed reports create new files. Existing snapshots are never overwritten.
    The experiment maps effort=scale*(parameter-1); no optimizer state is claimed.
    """
    raw = source.read_bytes()
    report = json.loads(raw)
    if report.get("qualification") != "exploratory-first-prefix-only":
        raise ValueError("unsupported fit report qualification")
    scales = report.get("effort_scales", [])
    evaluations = report.get("evaluations", [])
    if (
        not scales
        or not evaluations
        or any(not math.isfinite(scale) or scale <= 0 for scale in scales)
    ):
        raise ValueError("checkpoint needs finite positive scales and evaluations")
    for evaluation in evaluations:
        efforts = evaluation.get("efforts", [])
        error = evaluation.get("rmse_m", math.nan)
        if (
            len(efforts) != len(scales)
            or any(not math.isfinite(value) for value in efforts)
            or not math.isfinite(error)
            or error < 0
        ):
            raise ValueError("checkpoint evaluation is incomplete or nonfinite")
    best = min(evaluations, key=lambda row: row["rmse_m"])
    digest = hashlib.sha256(raw).hexdigest()
    payload = {
        "schema": "simscape-first-prefix-checkpoint/1",
        "source_report_sha256": digest,
        "resume_semantics": "best candidate warm start; optimizer internals not saved",
        "resume_parameters": [
            1 + effort / scale
            for effort, scale in zip(best["efforts"], scales, strict=True)
        ],
        "best_evaluation": best,
        "report": report,
    }
    encoded = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"evaluation-{len(evaluations):05d}-{digest}.json"
    try:
        with path.open("xb") as stream:
            stream.write(encoded)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise ValueError("existing checkpoint content differs") from None
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    snapshot_prefix_report(args.source, args.destination)


if __name__ == "__main__":
    main()
