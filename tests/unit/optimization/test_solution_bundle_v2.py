"""Explicit no-impact and separate-grid Solution Bundle v2 contract."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.optimization.solution_bundle import SolutionBundle
from src.shared.python.optimization.solution_bundle_v2 import (
    load_bundle_v2,
    migrate_v1,
    save_bundle_v2,
)

pytestmark = pytest.mark.unit


def _v1() -> SolutionBundle:
    a = {
        "time": np.array([0.0, 0.1]),
        "q_ref": np.zeros((2, 1)),
        "qd_ref": np.zeros((2, 1)),
        "tau_ff_ref": np.zeros((2, 1)),
        "q_rollout": np.zeros((2, 1)),
        "qd_rollout": np.zeros((2, 1)),
        "tau_cmd": np.zeros((2, 1)),
        "tau_applied": np.zeros((2, 1)),
        "knots": np.array([0.0, 0.1]),
        "delta_tau_coeffs": np.zeros((1, 1)),
        "q0": np.zeros(1),
        "qd0": np.zeros(1),
        "controller_state0": np.zeros(0),
        "kp": np.ones(1),
        "kd": np.ones(1),
    }
    return SolutionBundle(
        "s",
        None,
        "a" * 40,
        "b" * 64,
        {},
        "world",
        ("j",),
        {},
        {},
        {},
        0,
        {"t": 0.05, "q": [0.0], "qd": [0.0]},
        {},
        {},
        (),
        a,
        schema_version=1,
    )


def test_v1_migration_preserves_existing_event_without_invention(
    tmp_path: Path,
) -> None:
    v2 = migrate_v1(_v1())
    assert v2.schema_version == 2
    assert v2.impact_event == _v1().impact_event
    np.testing.assert_array_equal(v2.arrays["reference_time"], _v1().arrays["time"])
    save_bundle_v2(v2, tmp_path / "bundle")
    assert load_bundle_v2(tmp_path / "bundle").impact_event == v2.impact_event


def test_no_impact_and_distinct_reference_grid_roundtrip(tmp_path: Path) -> None:
    v2 = migrate_v1(_v1())
    arrays = {**v2.arrays, "reference_time": np.array([0.0, 0.05, 0.1])}
    for key in ("q_ref", "qd_ref", "tau_ff_ref"):
        arrays[key] = np.zeros((3, 1))
    v2 = replace(v2, impact_event={"status": "none"}, arrays=arrays)
    save_bundle_v2(v2, tmp_path / "bundle")
    out = load_bundle_v2(tmp_path / "bundle")
    assert out.impact_event == {"status": "none"}
    assert len(out.arrays["reference_time"]) == 3
    assert len(out.arrays["time"]) == 2


@pytest.mark.parametrize(
    "event", [{}, {"status": "none", "t": 0.0}, {"status": "detected"}]
)
def test_reject_ambiguous_or_fabricated_event(tmp_path: Path, event: dict) -> None:
    with pytest.raises(ValueError):
        save_bundle_v2(
            replace(migrate_v1(_v1()), impact_event=event), tmp_path / "bundle"
        )


def test_reject_nonfinite_extra_array(tmp_path: Path) -> None:
    v2 = migrate_v1(_v1())
    with pytest.raises(ValueError, match="extra"):
        save_bundle_v2(
            replace(v2, arrays={**v2.arrays, "extra": np.array([np.nan])}),
            tmp_path / "bundle",
        )


def test_reject_duplicate_manifest_key(tmp_path: Path) -> None:
    path = save_bundle_v2(migrate_v1(_v1()), tmp_path / "bundle") / "manifest.json"
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text.replace('"schema_version": 2', '"schema_version": 2, "schema_version": 2'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_bundle_v2(path.parent)


def test_external_manifest_digest_detects_metadata_tampering(tmp_path: Path) -> None:
    path = save_bundle_v2(migrate_v1(_v1()), tmp_path / "bundle") / "manifest.json"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    load_bundle_v2(path.parent, expected_manifest_sha256=digest)
    content = json.loads(path.read_text(encoding="utf-8"))
    content["frame"] = "different"
    path.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest hash"):
        load_bundle_v2(path.parent, expected_manifest_sha256=digest)


@pytest.mark.parametrize("name", ["time", "reference_time"])
def test_reject_nonmonotonic_time_grid(tmp_path: Path, name: str) -> None:
    v2 = migrate_v1(_v1())
    with pytest.raises(ValueError, match="time grid"):
        save_bundle_v2(
            replace(v2, arrays={**v2.arrays, name: np.array([0.1, 0.0])}),
            tmp_path / "b",
        )


def test_reject_incompatible_reference_dimension_and_coverage(tmp_path: Path) -> None:
    v2 = migrate_v1(_v1())
    with pytest.raises(ValueError, match="dimension"):
        save_bundle_v2(
            replace(v2, arrays={**v2.arrays, "q_ref": np.zeros((3, 1))}), tmp_path / "a"
        )
    with pytest.raises(ValueError, match="cover"):
        save_bundle_v2(
            replace(v2, arrays={**v2.arrays, "reference_time": np.array([0.0, 0.05])}),
            tmp_path / "b",
        )


def test_reject_unsupported_version_and_model_mismatch(tmp_path: Path) -> None:
    v2 = migrate_v1(_v1())
    with pytest.raises(ValueError, match="version"):
        save_bundle_v2(replace(v2, schema_version=3), tmp_path / "a")
    save_bundle_v2(v2, tmp_path / "b")
    with pytest.raises(ValueError, match="model hash"):
        load_bundle_v2(tmp_path / "b", expected_model_hash="c" * 64)


@pytest.mark.parametrize(
    "field,value",
    [
        ("git_sha", "not-a-sha"),
        ("model_hash", "not-a-hash"),
        ("optimizer_seed", True),
        ("flight_metrics", {"carry": "not numeric"}),
    ],
)
def test_reject_malformed_identity_and_metrics(
    tmp_path: Path, field: str, value: object
) -> None:
    with pytest.raises(ValueError):
        save_bundle_v2(replace(migrate_v1(_v1()), **{field: value}), tmp_path / "b")
