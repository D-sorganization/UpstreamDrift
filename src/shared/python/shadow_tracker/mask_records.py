"""Immutable binary mask observations and lineage records for Shadow Tracker (Packet B)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any

from ._validation import (
    MASK_SCHEMA_VERSION,
    check_id,
    check_payload_keys,
    check_pos_int,
    check_schema_version,
)
from .source_records import FrameIdentity

_MASK_FRAME_KEYS = frozenset(
    (
        "schema_version",
        "frame",
        "width_px",
        "height_px",
        "body",
        "club",
        "valid",
        "revision_id",
        "parent_revision_id",
        "producer_id",
        "correction_note",
    )
)


def _check_mask_bytes(val: object, field_name: str, expected_len: int) -> bytes:
    """Validate mask byte payload. Reject bytearrays, non-0/1 values, and wrong length."""
    if isinstance(val, bytearray):
        raise TypeError(f"{field_name} must be bytes, got bytearray")
    if not isinstance(val, bytes):
        raise TypeError(f"{field_name} must be bytes, got {type(val).__name__}")
    if len(val) != expected_len:
        raise ValueError(
            f"{field_name} length must be exactly {expected_len} bytes, got {len(val)}"
        )
    for b in val:
        if b not in (0, 1):
            raise ValueError(f"{field_name} pixels must be 0 or 1, got {b}")
    return val


@dataclass(frozen=True, slots=True, kw_only=True)
class MaskFrame:
    """Immutable binary mask observation and revision lineage record."""

    schema_version: str
    frame: FrameIdentity
    width_px: int
    height_px: int
    body: bytes
    club: bytes
    valid: bytes
    revision_id: str
    parent_revision_id: str | None
    producer_id: str
    correction_note: str

    def __post_init__(self) -> None:
        check_schema_version(self.schema_version, MASK_SCHEMA_VERSION)
        if not isinstance(self.frame, FrameIdentity):
            raise TypeError(
                f"frame must be a FrameIdentity, got {type(self.frame).__name__}"
            )
        check_pos_int(self.width_px, "width_px")
        check_pos_int(self.height_px, "height_px")

        expected_size = self.width_px * self.height_px
        _check_mask_bytes(self.body, "body", expected_size)
        _check_mask_bytes(self.club, "club", expected_size)
        _check_mask_bytes(self.valid, "valid", expected_size)

        # Invariant: body and club must be zero wherever valid is zero
        for idx in range(expected_size):
            if self.valid[idx] == 0:
                if self.body[idx] != 0:
                    raise ValueError(
                        f"body pixel at index {idx} is non-zero where valid is zero"
                    )
                if self.club[idx] != 0:
                    raise ValueError(
                        f"club pixel at index {idx} is non-zero where valid is zero"
                    )

        check_id(self.revision_id, "revision_id")
        if self.parent_revision_id is not None:
            check_id(self.parent_revision_id, "parent_revision_id")
            if self.parent_revision_id == self.revision_id:
                raise ValueError("parent_revision_id cannot equal revision_id")

        check_id(self.producer_id, "producer_id")

        if not isinstance(self.correction_note, str):
            raise TypeError(
                f"correction_note must be a str, got {type(self.correction_note).__name__}"
            )
        if self.parent_revision_id is not None:
            if (
                not self.correction_note
                or self.correction_note.strip() != self.correction_note
            ):
                raise ValueError(
                    f"correction_note must be non-empty and trimmed for a revised mask, got {self.correction_note!r}"
                )
        else:
            if self.correction_note.strip() != self.correction_note:
                raise ValueError(
                    f"correction_note must be trimmed, got {self.correction_note!r}"
                )

    @property
    def has_valid_pixels(self) -> bool:
        """True if any pixel in valid mask is 1."""
        return any(b == 1 for b in self.valid)

    @property
    def has_observed_foreground(self) -> bool:
        """True if any pixel in body or club is 1."""
        return any(b == 1 for b in self.body) or any(b == 1 for b in self.club)

    @property
    def observation_hash(self) -> str:
        """Deterministic SHA-256 hash over canonical UTF-8 JSON serialization."""
        payload = self.to_dict()
        canonical_json = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize record to JSON-compatible dictionary."""
        return {
            "schema_version": self.schema_version,
            "frame": self.frame.to_dict(),
            "width_px": self.width_px,
            "height_px": self.height_px,
            "body": list(self.body),
            "club": list(self.club),
            "valid": list(self.valid),
            "revision_id": self.revision_id,
            "parent_revision_id": self.parent_revision_id,
            "producer_id": self.producer_id,
            "correction_note": self.correction_note,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MaskFrame:
        """Deserialize from dictionary, rejecting unknown fields and validating dimensions/lengths before conversion."""
        check_payload_keys(payload, _MASK_FRAME_KEYS)
        raw_frame = payload["frame"]
        if isinstance(raw_frame, FrameIdentity):
            frame = raw_frame
        elif isinstance(raw_frame, dict):
            frame = FrameIdentity.from_dict(raw_frame)
        else:
            raise TypeError(
                f"frame must be a FrameIdentity or dict, got {type(raw_frame).__name__}"
            )

        width_px = check_pos_int(payload["width_px"], "width_px")
        height_px = check_pos_int(payload["height_px"], "height_px")
        expected_len = width_px * height_px

        def _check_len_and_to_bytes(val: object, field_name: str) -> bytes:
            if isinstance(val, bytearray):
                raise TypeError(f"{field_name} must be bytes, got bytearray")
            if isinstance(val, bytes):
                if len(val) != expected_len:
                    raise ValueError(
                        f"{field_name} length must be exactly {expected_len} bytes, got {len(val)}"
                    )
                return val
            if isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
                if len(val) != expected_len:
                    raise ValueError(
                        f"{field_name} length must be exactly {expected_len} bytes, got {len(val)}"
                    )
                # Validate integer elements
                byte_vals = bytearray()
                for item in val:
                    if isinstance(item, bool) or not isinstance(item, int):
                        raise TypeError(
                            f"{field_name} list items must be int, got {type(item).__name__}"
                        )
                    if item not in (0, 1):
                        raise ValueError(
                            f"{field_name} list items must be 0 or 1, got {item}"
                        )
                    byte_vals.append(item)
                return bytes(byte_vals)
            raise TypeError(
                f"{field_name} must be bytes or Sequence[int], got {type(val).__name__}"
            )

        body = _check_len_and_to_bytes(payload["body"], "body")
        club = _check_len_and_to_bytes(payload["club"], "club")
        valid = _check_len_and_to_bytes(payload["valid"], "valid")

        return cls(
            schema_version=payload["schema_version"],
            frame=frame,
            width_px=width_px,
            height_px=height_px,
            body=body,
            club=club,
            valid=valid,
            revision_id=payload["revision_id"],
            parent_revision_id=payload["parent_revision_id"],
            producer_id=payload["producer_id"],
            correction_note=payload["correction_note"],
        )
