"""Revision registration and durable selection contracts for #10233."""

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import pytest

from shared.python.shadow_tracker._validation import (
    FRAME_SCHEMA_VERSION,
    MASK_SCHEMA_VERSION,
)
from shared.python.shadow_tracker.mask_records import MaskFrame
from shared.python.shadow_tracker.segmentation import ManualMaskProvider
from shared.python.shadow_tracker.source_records import FrameIdentity

pytestmark = pytest.mark.unit


@pytest.fixture
def mask() -> MaskFrame:
    return MaskFrame(
        schema_version=MASK_SCHEMA_VERSION,
        frame=FrameIdentity(
            schema_version=FRAME_SCHEMA_VERSION,
            asset_id="asset",
            shot_id="shot",
            swing_id="swing",
            camera_id="camera",
            frame_id="frame",
            pts_ticks=1,
            timebase_numerator=1,
            timebase_denominator=30,
            physical_time_s=None,
            physical_time_reason="unknown clock",
            frame_sha256="a" * 64,
        ),
        width_px=2,
        height_px=2,
        body=b"\1\0\0\0",
        club=b"\0\1\0\0",
        valid=b"\1\1\1\0",
        revision_id="root",
        parent_revision_id=None,
        producer_id="reviewer",
        correction_note="initial",
    )


def snapshot(provider: ManualMaskProvider, path: Path) -> bytes:
    provider.save(path)
    return path.read_bytes()


@pytest.mark.parametrize(
    "change",
    [
        {"frame_sha256": "b" * 64},
        {"pts_ticks": 2},
        {"timebase_denominator": 60},
        {"physical_time_s": 0.1},
        {"decoder_version": "other"},
        {"camera_id": "other"},
    ],
)
def test_parent_requires_complete_observation_identity(
    mask: MaskFrame, tmp_path: Path, change: dict[str, Any]
) -> None:
    provider = ManualMaskProvider()
    provider.register_mask(mask)
    before = snapshot(provider, tmp_path / "before.json")
    child = replace(
        mask,
        revision_id="child",
        parent_revision_id="root",
        frame=replace(mask.frame, **change),
    )
    with pytest.raises(ValueError):
        provider.register_mask(child)
    assert snapshot(provider, tmp_path / "after.json") == before
    assert provider.all_revisions() == (mask,)
    assert provider.get_revision_history("frame") == (mask,)
    assert provider.get_mask("frame") == mask


def test_parent_requires_same_pixel_grid(mask: MaskFrame) -> None:
    provider = ManualMaskProvider()
    provider.register_mask(mask)
    with pytest.raises(ValueError):
        provider.register_mask(
            replace(
                mask,
                revision_id="child",
                parent_revision_id="root",
                width_px=1,
                height_px=4,
            )
        )


@pytest.mark.parametrize(
    "field", ["asset_id", "shot_id", "swing_id", "camera_id", "frame_id"]
)
def test_collision_and_namespace_are_independent(
    mask: MaskFrame, field: str, tmp_path: Path
) -> None:
    provider = ManualMaskProvider()
    provider.register_mask(mask)
    other = replace(mask, frame=replace(mask.frame, **{field: "other"}))
    before = snapshot(provider, tmp_path / "before.json")
    with pytest.raises(ValueError):
        provider.register_mask(other)
    assert snapshot(provider, tmp_path / "after.json") == before
    provider.register_mask(replace(other, revision_id="other"))
    assert len(provider.all_revisions()) == 2
    path = tmp_path / "namespaced.json"
    provider.save(path)
    restored = ManualMaskProvider.load(path)
    assert restored.all_revisions() == provider.all_revisions()
    assert snapshot(restored, path) == snapshot(provider, path)


def test_selected_ancestor_survives_idempotent_registration_and_reopen(
    mask: MaskFrame, tmp_path: Path
) -> None:
    provider = ManualMaskProvider()
    provider.register_mask(mask)
    root_key = provider.get_cache_key("frame")
    child = replace(
        mask, revision_id="child", parent_revision_id="root", body=b"\0\0\1\0"
    )
    provider.register_mask(child)
    assert provider.get_cache_key("frame") != root_key
    provider.register_mask(replace(mask))
    assert provider.get_mask("frame") == child
    provider.select_revision("root")
    assert provider.get_cache_key("frame") == root_key
    provider.register_mask(replace(child))
    assert provider.get_mask("frame") == mask
    path = tmp_path / "revisions.json"
    provider.save(path)
    reopened = ManualMaskProvider.load(path)
    assert reopened.all_revisions() == (mask, child)
    assert reopened.get_revision_history("frame") == (mask, child)
    assert reopened.get_mask("frame") == mask
    assert reopened.get_cache_key("frame") == root_key
    before = snapshot(reopened, path)
    with pytest.raises(KeyError):
        reopened.select_revision("missing")
    assert snapshot(reopened, path) == before


@pytest.mark.parametrize(
    "corruption",
    [
        "version",
        "unknown",
        "selection",
        "duplicate_selection",
        "missing_selection",
        "duplicate_revision",
        "cycle",
        "orphan",
        "pixels",
    ],
)
def test_corrupt_snapshot_rejected(
    mask: MaskFrame, tmp_path: Path, corruption: str
) -> None:
    provider = ManualMaskProvider()
    provider.register_mask(mask)
    provider.register_mask(
        replace(mask, revision_id="child", parent_revision_id="root")
    )
    path = tmp_path / "revisions.json"
    provider.save(path)
    payload = json.loads(path.read_text())
    if corruption == "version":
        payload["schema_version"] = "999"
    elif corruption == "unknown":
        payload["unexpected"] = True
    elif corruption == "selection":
        payload["current_revision_ids"] = ["missing"]
    elif corruption == "duplicate_selection":
        payload["current_revision_ids"] = ["root", "child"]
    elif corruption == "missing_selection":
        payload["current_revision_ids"] = []
    elif corruption == "duplicate_revision":
        payload["revisions"].append(payload["revisions"][0])
    elif corruption == "cycle":
        payload["revisions"][0]["parent_revision_id"] = "child"
    elif corruption == "orphan":
        payload["revisions"][1]["parent_revision_id"] = "absent"
    else:
        payload["revisions"][0]["body"] = [2, 0, 0, 0]
    path.write_text(json.dumps(payload))
    with pytest.raises((ValueError, TypeError, KeyError)):
        ManualMaskProvider.load(path)


def test_legacy_snapshot_retains_last_registered_selection(
    mask: MaskFrame, tmp_path: Path
) -> None:
    child = replace(mask, revision_id="child", parent_revision_id="root")
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {"schema_version": "1.0.0", "revisions": [mask.to_dict(), child.to_dict()]}
        )
    )
    assert ManualMaskProvider.load(path).get_mask("frame") == child


def test_interrupted_save_preserves_previous_snapshot(
    mask: MaskFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = ManualMaskProvider()
    provider.register_mask(mask)
    path = tmp_path / "revisions.json"
    before = snapshot(provider, path)
    provider.register_mask(
        replace(mask, revision_id="child", parent_revision_id="root")
    )

    def fail_replace(self: Path, target: Path) -> None:
        raise OSError("interrupted replace")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted"):
        provider.save(path)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]
    assert ManualMaskProvider.load(path).all_revisions() == (mask,)


@pytest.mark.parametrize(
    "change",
    [
        {"body": b"\0\0\1\0"},
        {"club": b"\0\0\1\0"},
        {"valid": b"\1\1\0\0"},
        {"producer_id": "other"},
        {"correction_note": "other"},
        {"width_px": 1, "height_px": 4},
    ],
)
def test_conflicting_record_never_mutates_indexes(
    mask: MaskFrame, tmp_path: Path, change: dict[str, Any]
) -> None:
    provider = ManualMaskProvider()
    provider.register_mask(mask)
    before = snapshot(provider, tmp_path / "before.json")
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        provider.register_mask(replace(mask, **change))
    assert snapshot(provider, tmp_path / "after.json") == before
    assert provider.all_revisions() == (mask,)
    assert provider.get_revision_history("frame") == (mask,)
    assert provider.get_mask("frame") == mask


def test_cycle_attempt_leaves_indexes_unchanged(
    mask: MaskFrame, tmp_path: Path
) -> None:
    provider = ManualMaskProvider()
    provider.register_mask(mask)
    child = replace(mask, revision_id="child", parent_revision_id="root")
    provider.register_mask(child)
    before = snapshot(provider, tmp_path / "before.json")
    with pytest.raises(ValueError):
        provider.register_mask(replace(mask, parent_revision_id="child"))
    assert snapshot(provider, tmp_path / "after.json") == before
    assert provider.get_revision_history("frame") == (mask, child)
    assert provider.get_mask("frame") == child
