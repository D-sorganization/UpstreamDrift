"""Player bag persistence and capture-bound snapshots retain original evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.shared.python.club_data.catalog import (
    ClubIdentity,
    ClubRecord,
    PropertyClaim,
    SpecificationSource,
)
from src.shared.python.club_data.player_clubs import (
    CaptureClubSnapshot,
    PlayerBag,
    PlayerClub,
)
from src.tools.capture_rig.capture_library import CaptureNotes, NOTES_FILE
from src.tools.capture_rig.equipment import (
    load_bag,
    save_bag,
    load_capture_club,
    save_capture_club,
)
from src.motion_capture.rig.documents import write_document

pytestmark = pytest.mark.unit


def club() -> PlayerClub:
    return PlayerClub(
        label="My 7-Iron",
        base=ClubRecord(
            identity=ClubIdentity(model="Custom", club_type="iron", number="7")
        ),
    )


def measured_length(value: float) -> PropertyClaim:
    return PropertyClaim(
        property="length",
        component="assembled",
        value=value,
        unit="in",
        status="measured",
        source=SpecificationSource(
            kind="player",
            title="Player measurement",
            license="user-provided",
            method="Tape measure",
        ),
    )


def capture_root(tmp_path: Path) -> Path:
    root = tmp_path / "swing"
    root.mkdir()
    write_document(
        root / NOTES_FILE, CaptureNotes(title="First Swing").model_dump(mode="json")
    )
    return root


def test_unknown_custom_club_and_archived_entries_round_trip(tmp_path: Path) -> None:
    unknown = club()
    archived = club().model_copy(update={"archived": True})
    bag = PlayerBag(clubs=(unknown, archived))
    path = tmp_path / "player_clubs.json"
    assert load_bag(path).clubs == ()
    save_bag(path, bag)
    assert load_bag(path) == bag
    assert unknown.effective_record().physical_value("length", "assembled") is None


def test_duplicate_bag_ids_and_unattributed_overrides_are_rejected() -> None:
    item = club()
    with pytest.raises(ValidationError, match="duplicate"):
        PlayerBag(clubs=(item, item))
    source = SpecificationSource(kind="legacy", title="Old defaults", license="unknown")
    override = measured_length(37).model_copy(update={"source": source})
    with pytest.raises(ValidationError, match="player"):
        PlayerClub(label="Bad", base=item.base, overrides=(override,))


def test_capture_snapshot_is_unchanged_when_bag_entry_is_edited(tmp_path: Path) -> None:
    root = capture_root(tmp_path)
    item = club().model_copy(update={"overrides": (measured_length(37),)})
    save_capture_club(root, item)
    original = load_capture_club(root)
    updated = item.model_copy(
        update={"overrides": (measured_length(38),), "notes": "Reshafted"}
    )
    save_bag(tmp_path / "bag.json", PlayerBag(clubs=(updated,)))
    assert load_capture_club(root) == original
    assert original.club.effective_record().physical_value(
        "length", "assembled"
    ) == pytest.approx(0.9398)


def test_snapshot_rejects_changed_capture_identity_and_content(tmp_path: Path) -> None:
    root = capture_root(tmp_path)
    save_capture_club(root, club())
    path = root / "capture_club.json"
    original = json.loads(path.read_text(encoding="utf-8"))
    data = dict(original)
    data["capture_id"] = "another-take"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="capture"):
        load_capture_club(root)
    original["club"]["label"] = "Changed without updating its revision"
    with pytest.raises(ValidationError, match="revision"):
        CaptureClubSnapshot.model_validate(original)


def test_legacy_capture_has_no_invented_club(tmp_path: Path) -> None:
    assert load_capture_club(tmp_path) is None
    with pytest.raises((ValueError, OSError)):
        save_capture_club(tmp_path, club())
    assert not (tmp_path / "capture_club.json").exists()


def test_corrupt_bag_is_not_silently_replaced(tmp_path: Path) -> None:
    path = tmp_path / "bag.json"
    path.write_text("broken", encoding="utf-8")
    with pytest.raises(ValueError):
        load_bag(path)
    assert path.read_text(encoding="utf-8") == "broken"


def test_snapshot_save_refuses_archived_club(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="archived"):
        save_capture_club(
            capture_root(tmp_path), club().model_copy(update={"archived": True})
        )


def test_stale_bag_save_cannot_overwrite_an_intervening_edit(tmp_path: Path) -> None:
    path = tmp_path / "bag.json"
    original = PlayerBag(clubs=(club(),))
    save_bag(path, original)
    changed = PlayerBag(clubs=(club(),))
    save_bag(path, changed, expected_revision=original.revision)
    with pytest.raises(ValueError, match="changed"):
        save_bag(path, original, expected_revision=original.revision)
    assert load_bag(path) == changed


def test_reselecting_a_club_retains_both_capture_revisions(tmp_path: Path) -> None:
    root = capture_root(tmp_path)
    original = save_capture_club(root, club())
    second = save_capture_club(root, club())
    assert load_capture_club(root) == second
    revisions = root / "equipment_revisions"
    saved = CaptureClubSnapshot.model_validate_json(
        (revisions / f"{original.club_revision}.json").read_text(encoding="utf-8")
    )
    assert saved == original
