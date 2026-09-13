"""Strict native-motion persistence and replacement-failure safety."""

import json
from pathlib import Path

import numpy as np
import pytest
from src.shared.python.pose_interchange import (
    NativeJointStateAdapter,
    NativeMotionSequence,
    export_native_motion,
    restore_native_motion,
)
from src.shared.python.pose_interchange.native_motion_io import (
    load_native_motion,
    save_native_motion,
)

pytestmark = pytest.mark.unit
Source = tuple[NativeJointStateAdapter, NativeMotionSequence, tuple[np.ndarray, ...]]


@pytest.fixture
def source() -> Source:
    path = (
        Path(__file__).resolve().parents[3]
        / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
    )
    adapter = NativeJointStateAdapter(json.loads(path.read_text()))
    q = np.full((2, len(adapter.coordinate_order)), 0.2)
    for group in adapter.groups:
        q[:, adapter.coordinate_order.index(group.coordinates[1])] = -2.1
        q[:, adapter.coordinate_order.index(group.coordinates[0])] = 7.1
    fields = (q, np.full_like(q, 0.3), np.full_like(q, 0.4), np.full_like(q, 2.0))
    return adapter, export_native_motion(adapter, [0, 0.1], *fields), fields


def test_full_roundtrip_and_separate_hash(source: Source, tmp_path: Path) -> None:
    adapter, motion, fields = source
    path = tmp_path / "motion.json"
    assert save_native_motion(motion, path, raw_model_sha256="a" * 64) == path
    document = load_native_motion(path)
    assert document.raw_model_sha256 == "a" * 64
    assert document.sequence.specification_sha256 == motion.specification_sha256
    assert document.sequence.native_reference == motion.native_reference
    assert document.sequence.rotation_groups == motion.rotation_groups
    for actual, expected in zip(
        restore_native_motion(adapter, document.sequence), fields, strict=True
    ):
        np.testing.assert_allclose(actual, expected, atol=1e-12)
    with pytest.raises(TypeError):
        document.sequence.states[0].scalars["bad"] = (0, 0, 0, 0)
    save_native_motion(document.sequence, path)
    assert load_native_motion(path).raw_model_sha256 is None


@pytest.mark.parametrize(
    "mutation",
    [
        "file_version",
        "sequence_version",
        "nonfinite",
        "inventory",
        "count",
        "unknown",
        "raw_hash",
    ],
)
def test_malformed_payload_rejected(
    source: Source, tmp_path: Path, mutation: str
) -> None:
    _, motion, _ = source
    path = tmp_path / "motion.json"
    save_native_motion(motion, path)
    payload = json.loads(path.read_text())
    if mutation == "file_version":
        payload["file_convention"] = "future"
    elif mutation == "sequence_version":
        payload["sequence"]["convention_tag"] = "future"
    elif mutation == "nonfinite":
        payload["sequence"]["times_s"][0] = float("nan")
    elif mutation == "inventory":
        payload["sequence"]["coordinate_order"][0] = "other"
    elif mutation == "count":
        payload["sequence"]["states"].pop()
    elif mutation == "unknown":
        payload["unexpected"] = "ignored?"
    else:
        payload["raw_model_sha256"] = "not-a-hash"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_native_motion(path)


def test_failed_replace_preserves_original_and_cleans_temp(
    source: Source, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.shared.python.pose_interchange.native_motion_io as provider

    _, motion, _ = source
    path = tmp_path / "motion.json"
    path.write_text("original content")

    def fail(*args: object) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(provider.os, "replace", fail)
    with pytest.raises(OSError, match="injected"):
        save_native_motion(motion, path)
    assert path.read_text() == "original content"
    assert list(tmp_path.iterdir()) == [path]


def test_invalid_hash_never_changes_existing_file(
    source: Source, tmp_path: Path
) -> None:
    _, motion, _ = source
    path = tmp_path / "motion.json"
    path.write_text("original")
    with pytest.raises(ValueError):
        save_native_motion(motion, path, raw_model_sha256="x")
    assert path.read_text() == "original"


def test_duplicate_json_keys_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        '{"file_convention":"native-motion-file-v1","file_convention":"future"}'
    )
    with pytest.raises(ValueError, match="Duplicate"):
        load_native_motion(path)


def test_public_file_api_is_lazy_and_discoverable() -> None:
    import src.shared.python.pose_interchange as public
    from src.shared.python.pose_interchange.native_motion_io import NativeMotionDocument

    for name, expected in (
        ("NativeMotionDocument", NativeMotionDocument),
        ("save_native_motion", save_native_motion),
        ("load_native_motion", load_native_motion),
    ):
        assert name in public.__all__
        assert getattr(public, name) is expected
