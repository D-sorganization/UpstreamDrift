"""Atomic player-bag storage and capture-owned equipment revisions."""

from __future__ import annotations

import json
from pathlib import Path

from src.motion_capture.rig.documents import write_document
from src.motion_capture.rig.equipment import (
    CAPTURE_CLUB_FILE as CAPTURE_CLUB_FILE,
    load_capture_club as load_capture_club,
)
from src.shared.python.club_data.catalog_io import MAX_EXCHANGE_BYTES
from src.shared.python.club_data.player_clubs import (
    CaptureClubSnapshot,
    PlayerBag,
    PlayerClub,
)

from .capture_library import read_notes

BAG_FILE = "player_clubs.json"
REVISION_DIRECTORY = "equipment_revisions"


def _read(path: Path) -> str:
    if path.stat().st_size > MAX_EXCHANGE_BYTES:
        raise ValueError("Equipment document exceeds the 8 MiB limit")
    return path.read_text(encoding="utf-8")


def load_bag(path: Path) -> PlayerBag:
    """Return an empty new bag; report corrupt existing records for recovery."""
    return PlayerBag.model_validate_json(_read(path)) if path.exists() else PlayerBag()


def save_bag(
    path: Path, bag: PlayerBag, *, expected_revision: str | None = None
) -> None:
    """Replace only a validated bag and reject an observed intervening edit.

    The revision check protects stale dialogs; atomic replacement protects the
    file on write failure. This is not a multi-writer database transaction.
    """
    actual = load_bag(path)
    if path.exists() and expected_revision is None:
        raise ValueError(
            "Existing bag requires its expected revision; reload before saving"
        )
    if expected_revision is not None and actual.revision != expected_revision:
        raise ValueError("Player bag changed; reload it before saving")
    payload = bag.model_dump(mode="json")
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    if len(encoded.encode("utf-8")) > MAX_EXCHANGE_BYTES:
        raise ValueError("Equipment document exceeds the 8 MiB limit")
    write_document(path, payload)


def save_capture_club(root: Path, club: PlayerClub) -> CaptureClubSnapshot:
    """Save a capture-owned snapshot, retaining earlier selections by revision."""
    notes = read_notes(root)
    snapshot = CaptureClubSnapshot(
        capture_id=notes.capture_id, club=club, club_revision=club.revision
    )
    # A damaged existing selection must not be silently replaced by a UI action.
    load_capture_club(root)
    revisions = root / REVISION_DIRECTORY
    revisions.mkdir(exist_ok=True)
    archive = revisions / f"{snapshot.club_revision}.json"
    if archive.exists():
        saved = CaptureClubSnapshot.model_validate_json(_read(archive))
        if saved.capture_id != snapshot.capture_id or saved.club != club:
            raise ValueError("Archived equipment revision conflicts with this capture")
        snapshot = saved
    else:
        write_document(archive, snapshot.model_dump(mode="json"))
    write_document(root / CAPTURE_CLUB_FILE, snapshot.model_dump(mode="json"))
    return snapshot
