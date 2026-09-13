"""Inspect saved marker Jacobian conditioning without another dynamics solve."""

import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


def main() -> None:
    here = Path(__file__).resolve().parent
    archive = here.parent / "sensitivity_9967_72/raw-run.zip"
    with ZipFile(archive) as source:
        checkpoint = json.loads(source.read("inputs/1-best.json"))["candidate"]
        target = json.loads(source.read("inputs/2-driver_marker_payload_9967.json"))
        saved = np.load(io.BytesIO(source.read("output/sensitivity.npz")))
        jacobian = saved["marker_jacobian"].copy()
        clock = saved["time_s"].copy()
    indices = [target["labels"].index(name) for name in checkpoint["marker_labels"]]
    mask = np.asarray(target["time_s"]) <= checkpoint["duration_s"]
    np.testing.assert_array_equal(np.asarray(target["time_s"])[mask], clock)
    valid = np.asarray(target["valid"], dtype=bool)[mask][:, indices]
    points = np.asarray(target["points_world_m"])[mask][:, indices]
    valid &= np.isfinite(points).all(axis=2)
    # Match runner73's dimensionless controls and marker/terminal rows only.
    amplitude, terminal_weight = 10.0, 10.0
    rows = (
        np.concatenate(
            (
                jacobian[valid].reshape(-1, 81),
                terminal_weight * jacobian[-1, valid[-1]].reshape(-1, 81),
            )
        )
        * amplitude
    )
    if not np.isfinite(rows).all():
        raise ValueError("Nonfinite stored Jacobian")
    norms = np.linalg.norm(rows, axis=0)
    singular = np.linalg.svd(rows, compute_uv=False)
    threshold = max(rows.shape) * np.finfo(float).eps * singular[0]
    normalized = rows / np.where(norms > 0, norms, 1)
    normalized_singular = np.linalg.svd(normalized, compute_uv=False)
    labels = [
        f"{name}:B{control}"
        for name in checkpoint["coordinate_names"]
        for control in (4, 5, 6)
    ]
    report = {
        "qualification": "Local marker/terminal Jacobian conditioning only; excludes effort penalty, bounds, and nonlinear replay. No convergence or full derivative certification.",
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy_version": np.__version__,
        "rows_columns": list(rows.shape),
        "amplitude_scale": amplitude,
        "terminal_weight": terminal_weight,
        "singular_values": singular.tolist(),
        "numerical_rank": int(np.sum(singular > threshold)),
        "rank_threshold": float(threshold),
        "condition_number": float(singular[0] / singular[-1]),
        "column_normalized_condition_number": float(
            normalized_singular[0] / normalized_singular[-1]
        ),
        "column_norms": dict(zip(labels, norms.tolist(), strict=True)),
    }
    (here / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
