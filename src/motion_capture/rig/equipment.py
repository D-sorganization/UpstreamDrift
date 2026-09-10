"""Read capture-bound equipment without loading Capture Rig or Qt."""

from pathlib import Path
from typing import Any

from src.shared.python.club_data.catalog_io import MAX_EXCHANGE_BYTES
from src.shared.python.club_data.catalog import Component, Property
from src.shared.python.club_data.player_clubs import CaptureClubSnapshot

from .capture_notes import read_notes

CAPTURE_CLUB_FILE = "capture_club.json"


def load_capture_club(root: Path) -> CaptureClubSnapshot | None:
    """Legacy sessions remain unassigned; changed capture IDs invalidate bindings."""
    path = root / CAPTURE_CLUB_FILE
    if not path.exists():
        return None
    if path.stat().st_size > MAX_EXCHANGE_BYTES:
        raise ValueError("Equipment document exceeds the 8 MiB limit")
    snapshot = CaptureClubSnapshot.model_validate_json(path.read_text(encoding="utf-8"))
    if snapshot.capture_id != read_notes(root).capture_id:
        raise ValueError("Equipment snapshot belongs to another capture")
    return snapshot


def model_equipment_context(root: Path) -> dict[str, Any] | None:
    """Report evidence and eligible SI values, without inventing model constraints."""
    snapshot = load_capture_club(root)
    if snapshot is None:
        return None
    record = snapshot.club.effective_record()
    values: dict[str, float | None] = {}
    withheld: dict[str, str] = {}
    properties: tuple[tuple[Property, Component], ...] = (
        ("length", "assembled"),
        ("mass", "head"),
    )
    for prop, component in properties:
        key = f"{component}_{prop}_si"
        try:
            values[key] = record.physical_value(prop, component)
        except ValueError as exc:
            values[key] = None
            withheld[key] = str(exc)
    return {
        "selection": snapshot.model_dump(mode="json"),
        "club_number": record.identity.number,
        "eligible_values": values,
        "withheld": withheld,
        "constraints_applied": [],
        "usage": "Equipment context only. Current body models do not observe a club segment.",
    }
