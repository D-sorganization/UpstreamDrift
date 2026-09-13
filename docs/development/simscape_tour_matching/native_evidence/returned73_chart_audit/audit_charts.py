"""Sample native rotation-chart conditioning on the archived run73 trajectory."""

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from src.shared.python.pose_interchange import (
    NativeJointStateAdapter,
    SerialRotationChart,
)


def main() -> None:
    here = Path(__file__).resolve().parent
    archive = here.parent / "regularized_fit_9967_73/raw-run.zip"
    samples = here.parent / "regularized_fit_9967_73/sampled-markers-state.npz"
    with ZipFile(archive) as source:
        raw = source.read("inputs/0-native_geometry_spec_9967.json")
        if samples.read_bytes() != source.read(
            "output/evidence-replay/sampled-markers-state.npz"
        ):
            raise ValueError("State samples differ from original run73 archive")
    adapter = NativeJointStateAdapter(json.loads(raw))
    with np.load(samples) as data:
        clock, states = data["time_s"].copy(), data["native_state"].copy()
        np.testing.assert_array_equal(
            data["coordinate_names"], adapter.coordinate_order
        )
    if states.shape != (len(clock), 2 * len(adapter.coordinate_order)):
        raise ValueError("Saved state shape differs from native inventory")
    groups = {}
    for group in adapter.groups:
        indices = [adapter.coordinate_order.index(name) for name in group.coordinates]
        chart = SerialRotationChart(group.axes)
        conditions = np.array([chart.condition_number(row[indices]) for row in states])
        peak = int(np.argmax(conditions))
        groups[group.name] = {
            "condition_at_each_sample": conditions.tolist(),
            "maximum": float(conditions[peak]),
            "maximum_time_s": float(clock[peak]),
        }
    report = {
        "scope": "Sampled rate-map conditioning only; between-sample extrema, inertia and full trajectory sensitivity are not bounded by this audit.",
        "specification_sha256": adapter.specification_sha256,
        "raw_model_sha256": hashlib.sha256(raw).hexdigest(),
        "input_hashes": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (archive, samples, Path(__file__))
        },
        "time_s": clock.tolist(),
        "groups": groups,
    }
    (here / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
