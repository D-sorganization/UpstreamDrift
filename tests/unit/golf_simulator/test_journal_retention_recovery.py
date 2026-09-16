"""Unit tests for journal crash recovery and retention pruning (GS-08, #10197).

Follows TDD, DbC, Law of Demeter, and DRY.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest

from src.shared.python.golf_simulator.journal import (
    DeliveryStatus,
    JournalEntry,
    ShotJournal,
)

pytestmark = pytest.mark.unit


def _iso_offset(seconds_ago: float) -> str:
    dt = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    return dt.isoformat()


def test_crash_recovery_transitions_pending_to_ambiguous(tmp_path: Path) -> None:
    journal_path = tmp_path / "test_journal.json"
    journal = ShotJournal(storage_path=journal_path)

    # Record 3 shots: one acked, one rejected, one pending (simulating crash)
    journal.record_intent("shot-acked", b"acked_payload")
    journal.record_acknowledgment("shot-acked", 200, "OK")

    journal.record_intent("shot-rejected", b"rejected_payload")
    journal.record_rejection("shot-rejected", 400, "Bad payload")

    journal.record_intent("shot-pending", b"pending_payload")
    assert journal.get_entry("shot-pending").status == DeliveryStatus.PENDING

    # Simulate process restart and crash recovery
    recovered_count = journal.recover_on_startup()
    assert recovered_count == 1

    entry = journal.get_entry("shot-pending")
    assert entry.status == DeliveryStatus.AMBIGUOUS
    assert entry.error is not None
    assert "Crash recovery" in entry.error

    # Ensure other entries were not altered
    assert journal.get_entry("shot-acked").status == DeliveryStatus.ACKNOWLEDGED
    assert journal.get_entry("shot-rejected").status == DeliveryStatus.REJECTED


def test_retention_pruning_preserves_unresolved(tmp_path: Path) -> None:
    journal_path = tmp_path / "test_journal.json"
    journal = ShotJournal(storage_path=journal_path)

    # Directly insert mock entries with different timestamps and statuses
    old_time = _iso_offset(3600)  # 1 hour ago
    recent_time = _iso_offset(60)  # 1 minute ago

    old_acked = JournalEntry(
        shot_id="old-acked",
        status=DeliveryStatus.ACKNOWLEDGED,
        payload_bytes=b"1",
        created_at_utc=old_time,
        updated_at_utc=old_time,
    )
    recent_acked = JournalEntry(
        shot_id="recent-acked",
        status=DeliveryStatus.ACKNOWLEDGED,
        payload_bytes=b"2",
        created_at_utc=recent_time,
        updated_at_utc=recent_time,
    )
    old_ambiguous = JournalEntry(
        shot_id="old-ambiguous",
        status=DeliveryStatus.AMBIGUOUS,
        payload_bytes=b"3",
        created_at_utc=old_time,
        updated_at_utc=old_time,
        error="Connection lost",
    )

    with journal._lock:
        journal._entries["old-acked"] = old_acked
        journal._entries["recent-acked"] = recent_acked
        journal._entries["old-ambiguous"] = old_ambiguous

    # Prune with 10-minute horizon (600s)
    pruned_count = journal.prune_retention(max_age_seconds=600.0, keep_unresolved=True)
    assert pruned_count == 1  # Only old-acked should be pruned

    remaining_ids = {e.shot_id for e in journal.list_entries()}
    assert "old-acked" not in remaining_ids
    assert "recent-acked" in remaining_ids
    # Crucial acceptance criterion: uncertain/unresolved records MUST be preserved!
    assert "old-ambiguous" in remaining_ids
