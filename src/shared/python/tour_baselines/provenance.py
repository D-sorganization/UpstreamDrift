"""Provenance records for tour-average captures (TB-01 #10586).

Records what is known about manufacturer software, player identity, creation and
export timestamps, units, coordinate systems, and explicit unresolved fields
(averaging methodology, time-normalization, usage rights) without fabricated citations.
Separates shared player anatomy from capture-specific club geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.shared.python.contracts import postcondition, precondition

SHARED_PLAYER_ID = "967eac5b-2e78-4207-a99f-d57437296d70"


@dataclass(frozen=True)
class TourProvenance:
    """Rigorous provenance and origin record for a canonical capture file."""

    kind: str
    manufacturer_company: str
    manufacturer_software: str
    software_version: str
    player_id: str
    capture_id: str
    install_id: str
    capture_version: str
    gears_version: str
    created_at_utc: str
    exported_at_utc: str
    coordinate_units: str = "m"
    vertical_axis: str = "y"
    handedness: str = "right"
    averaging_method: str = "unresolved"
    time_normalization: str = "unresolved"
    usage_rights: str = "internal_fleet_reference_unrestricted_in_workspace"
    citation_policy: str = "explicit_unresolved_no_fabricated_citation"
    notes: str = (
        "Subject anatomy is shared across driver and iron (identical player ID); "
        "shaft and clubhead geometry are capture-specific."
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "manufacturer_company": self.manufacturer_company,
            "manufacturer_software": self.manufacturer_software,
            "software_version": self.software_version,
            "player_id": self.player_id,
            "capture_id": self.capture_id,
            "install_id": self.install_id,
            "capture_version": self.capture_version,
            "gears_version": self.gears_version,
            "created_at_utc": self.created_at_utc,
            "exported_at_utc": self.exported_at_utc,
            "coordinate_units": self.coordinate_units,
            "vertical_axis": self.vertical_axis,
            "handedness": self.handedness,
            "averaging_method": self.averaging_method,
            "time_normalization": self.time_normalization,
            "usage_rights": self.usage_rights,
            "citation_policy": self.citation_policy,
            "notes": self.notes,
        }


PROVENANCE_DRIVER = TourProvenance(
    kind="driver",
    manufacturer_company="Gears",
    manufacturer_software="GearsSports",
    software_version="v3",
    player_id=SHARED_PLAYER_ID,
    capture_id="22196b66-8e76-41cd-8815-edec0a74312e",
    install_id="7",
    capture_version="7f5691c3",
    gears_version="4aad4b00",
    created_at_utc="2018-04-23T10:04:56Z",
    exported_at_utc="2020-12-12T08:31:05Z",
)

PROVENANCE_IRON = TourProvenance(
    kind="iron",
    manufacturer_company="Gears",
    manufacturer_software="GearsSports",
    software_version="v3",
    player_id=SHARED_PLAYER_ID,
    capture_id="a805587c-e51b-4cbd-bf87-7a18014e2286",
    install_id="7",
    capture_version="7f5691c3",
    gears_version="4aad4b00",
    created_at_utc="2018-04-23T10:04:35Z",
    exported_at_utc="2020-12-12T08:35:13Z",
)

_PROVENANCE_MAP: dict[str, TourProvenance] = {
    "driver": PROVENANCE_DRIVER,
    "iron": PROVENANCE_IRON,
}


@precondition(lambda kind: isinstance(kind, str), "kind must be str")
@postcondition(
    lambda r: r.player_id == SHARED_PLAYER_ID, "player ID must match canonical subject"
)
def get_tour_provenance(kind: str) -> TourProvenance:
    """Return provenance record for 'driver' or 'iron'."""
    normalized = kind.strip().lower()
    if normalized not in _PROVENANCE_MAP:
        raise ValueError(
            f"Unknown capture kind {kind!r}; expected one of {sorted(_PROVENANCE_MAP)}"
        )
    return _PROVENANCE_MAP[normalized]
