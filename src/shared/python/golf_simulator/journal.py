"""Durable delivery intent journal for golf simulator integration (GS-03, #10192).

Enforces DbC invariants:
- Intent is recorded before physical socket write with status PENDING.
- Successful response receipt transitions to ACKNOWLEDGED.
- Protocol rejection / error response transitions to REJECTED.
- Network drop or timeout during in-flight delivery transitions to AMBIGUOUS.
- Invariant: ambiguous deliveries are NOT automatically resubmitted to prevent
  duplicate ball flight launches on physical simulator greens.
- Idempotent record storage preventing accidental ID collisions.
"""

from __future__ import annotations

import enum
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import threading
from typing import Any

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DeliveryStatus(str, enum.Enum):
    """Delivery status states for outbound simulator shots."""

    PENDING = "PENDING"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    REJECTED = "REJECTED"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class JournalEntry:
    """Immutable audit record representing a shot delivery attempt."""

    shot_id: str
    status: DeliveryStatus
    payload_bytes: bytes
    created_at_utc: str
    updated_at_utc: str
    response_code: int | None = None
    response_message: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert entry to serializable dictionary."""
        return {
            "shot_id": self.shot_id,
            "status": self.status.value,
            "payload_hex": self.payload_bytes.hex(),
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "response_code": self.response_code,
            "response_message": self.response_message,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JournalEntry:
        """Construct entry from serialized dictionary."""
        return cls(
            shot_id=str(data["shot_id"]),
            status=DeliveryStatus(data["status"]),
            payload_bytes=bytes.fromhex(data["payload_hex"]),
            created_at_utc=str(data["created_at_utc"]),
            updated_at_utc=str(data["updated_at_utc"]),
            response_code=data.get("response_code"),
            response_message=data.get("response_message"),
            error=data.get("error"),
        )


class ShotJournal:
    """Thread-safe and durable delivery intent journal."""

    def __init__(
        self,
        storage_path: Path | None = None,
        auto_recover: bool = True,
    ) -> None:
        self._lock = threading.Lock()
        self._storage_path = Path(storage_path) if storage_path is not None else None
        self._entries: dict[str, JournalEntry] = {}
        if self._storage_path is not None and self._storage_path.is_file():
            self._load_from_storage()
            if auto_recover:
                self.recover_on_startup()

    def record_intent(self, shot_id: str, payload_bytes: bytes) -> JournalEntry:
        """Record intent to deliver shot before wire transmission.

        Preconditions:
            shot_id must be non-empty string.
            payload_bytes must be non-empty bytes.
            shot_id must not already exist in the journal.
        """
        if not shot_id or not isinstance(shot_id, str):
            raise ValueError("shot_id must be a non-empty string")
        if not payload_bytes or not isinstance(payload_bytes, (bytes, bytearray)):
            raise ValueError("payload_bytes must be non-empty bytes")

        now = _utc_now_iso()
        entry = JournalEntry(
            shot_id=shot_id,
            status=DeliveryStatus.PENDING,
            payload_bytes=bytes(payload_bytes),
            created_at_utc=now,
            updated_at_utc=now,
        )
        with self._lock:
            if shot_id in self._entries:
                raise ValueError(f"Shot intent for '{shot_id}' already exists")
            self._entries[shot_id] = entry
            self._persist_unlocked()
            logger.info("Recorded delivery intent for shot %s", shot_id)
            return entry

    def record_acknowledgment(
        self, shot_id: str, code: int, message: str
    ) -> JournalEntry:
        """Record successful simulator acknowledgment for shot."""
        now = _utc_now_iso()
        with self._lock:
            existing = self._get_entry_unlocked(shot_id)
            updated = JournalEntry(
                shot_id=existing.shot_id,
                status=DeliveryStatus.ACKNOWLEDGED,
                payload_bytes=existing.payload_bytes,
                created_at_utc=existing.created_at_utc,
                updated_at_utc=now,
                response_code=code,
                response_message=message,
                error=None,
            )
            self._entries[shot_id] = updated
            self._persist_unlocked()
            logger.info("Recorded ACK for shot %s (code=%d)", shot_id, code)
            return updated

    def record_ambiguity(self, shot_id: str, reason: str) -> JournalEntry:
        """Record unconfirmed / ambiguous delivery state (e.g. disconnect mid-flight)."""
        now = _utc_now_iso()
        with self._lock:
            existing = self._get_entry_unlocked(shot_id)
            updated = JournalEntry(
                shot_id=existing.shot_id,
                status=DeliveryStatus.AMBIGUOUS,
                payload_bytes=existing.payload_bytes,
                created_at_utc=existing.created_at_utc,
                updated_at_utc=now,
                response_code=existing.response_code,
                response_message=existing.response_message,
                error=reason,
            )
            self._entries[shot_id] = updated
            self._persist_unlocked()
            logger.warning(
                "Recorded AMBIGUOUS delivery for shot %s: %s", shot_id, reason
            )
            return updated

    def record_rejection(self, shot_id: str, code: int, message: str) -> JournalEntry:
        """Record simulator rejection or error response."""
        now = _utc_now_iso()
        with self._lock:
            existing = self._get_entry_unlocked(shot_id)
            updated = JournalEntry(
                shot_id=existing.shot_id,
                status=DeliveryStatus.REJECTED,
                payload_bytes=existing.payload_bytes,
                created_at_utc=existing.created_at_utc,
                updated_at_utc=now,
                response_code=code,
                response_message=message,
                error=message,
            )
            self._entries[shot_id] = updated
            self._persist_unlocked()
            logger.info(
                "Recorded REJECTED delivery for shot %s (code=%d)", shot_id, code
            )
            return updated

    def get_entry(self, shot_id: str) -> JournalEntry:
        """Retrieve audit entry by shot ID."""
        with self._lock:
            return self._get_entry_unlocked(shot_id)

    def recover_on_startup(self) -> int:
        """Scan journal on startup and transition unacknowledged PENDING shots to AMBIGUOUS.

        Prevents silent duplicate shot executions following process crashes or restarts.
        """
        now = _utc_now_iso()
        recovered_count = 0
        with self._lock:
            for shot_id, entry in list(self._entries.items()):
                if entry.status == DeliveryStatus.PENDING:
                    updated = JournalEntry(
                        shot_id=entry.shot_id,
                        status=DeliveryStatus.AMBIGUOUS,
                        payload_bytes=entry.payload_bytes,
                        created_at_utc=entry.created_at_utc,
                        updated_at_utc=now,
                        response_code=entry.response_code,
                        response_message=entry.response_message,
                        error="Crash recovery: unconfirmed pending delivery marked ambiguous on startup",
                    )
                    self._entries[shot_id] = updated
                    recovered_count += 1
            if recovered_count > 0:
                self._persist_unlocked()
                logger.warning(
                    "Crash recovery transitioned %d pending shot(s) to AMBIGUOUS",
                    recovered_count,
                )
        return recovered_count

    def prune_retention(
        self,
        max_age_seconds: float,
        keep_unresolved: bool = True,
    ) -> int:
        """Prune historical journal entries older than max_age_seconds.

        DbC Invariant:
        If keep_unresolved is True, PENDING and AMBIGUOUS records are strictly
        preserved regardless of age to protect unconfirmed delivery evidence.
        """
        if max_age_seconds < 0:
            raise ValueError("max_age_seconds must be non-negative")

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=max_age_seconds)
        pruned_count = 0

        with self._lock:
            for shot_id, entry in list(self._entries.items()):
                if keep_unresolved and entry.status in (
                    DeliveryStatus.PENDING,
                    DeliveryStatus.AMBIGUOUS,
                ):
                    continue

                try:
                    entry_dt = datetime.fromisoformat(entry.updated_at_utc)
                except Exception:
                    continue

                if entry_dt < cutoff:
                    del self._entries[shot_id]
                    pruned_count += 1

            if pruned_count > 0:
                self._persist_unlocked()
                logger.info(
                    "Pruned %d journal entries older than %gs",
                    pruned_count,
                    max_age_seconds,
                )
        return pruned_count

    def list_entries(self, status: DeliveryStatus | None = None) -> list[JournalEntry]:
        """List audit entries optionally filtered by delivery status."""
        with self._lock:
            entries = list(self._entries.values())
            if status is not None:
                entries = [e for e in entries if e.status == status]
            return entries

    def _get_entry_unlocked(self, shot_id: str) -> JournalEntry:
        if shot_id not in self._entries:
            raise KeyError(f"No journal entry found for shot '{shot_id}'")
        return self._entries[shot_id]

    def _load_from_storage(self) -> None:
        if self._storage_path is None or not self._storage_path.is_file():
            return
        try:
            raw = self._storage_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, list):
                for item in data:
                    entry = JournalEntry.from_dict(item)
                    self._entries[entry.shot_id] = entry
        except Exception:
            logger.exception(
                "Failed to load journal storage from %s", self._storage_path
            )

    def _persist_unlocked(self) -> None:
        if self._storage_path is None:
            return
        parent_dir = self._storage_path.parent
        parent_dir.mkdir(parents=True, exist_ok=True)
        entries_data = [e.to_dict() for e in self._entries.values()]
        temp_file = self._storage_path.with_suffix(".tmp")
        temp_file.write_text(json.dumps(entries_data, indent=2), encoding="utf-8")
        temp_file.replace(self._storage_path)
