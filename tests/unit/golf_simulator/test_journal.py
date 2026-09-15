"""Unit tests for golf simulator delivery intent journal (GS-03, #10192).

Enforces DbC invariants:
- Records intent BEFORE socket write (PENDING).
- Records receipt acknowledgment (ACKNOWLEDGED) or rejection (REJECTED).
- Abrupt disconnect / timeout marks delivery as AMBIGUOUS.
- Rejects duplicate shot IDs to avoid replay bugs.
- Preserves audit trail across file-backed persistence.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from src.shared.python.golf_simulator.journal import (
    DeliveryStatus,
    JournalEntry,
    ShotJournal,
)

pytestmark = pytest.mark.unit


def test_record_intent_and_acknowledgment() -> None:
    journal = ShotJournal()
    entry = journal.record_intent("shot-001", b'{"ShotNumber": 1}')
    assert entry.shot_id == "shot-001"
    assert entry.status == DeliveryStatus.PENDING
    assert entry.payload_bytes == b'{"ShotNumber": 1}'

    ack = journal.record_acknowledgment("shot-001", code=200, message="Shot received")
    assert ack.shot_id == "shot-001"
    assert ack.status == DeliveryStatus.ACKNOWLEDGED
    assert ack.response_code == 200
    assert ack.response_message == "Shot received"


def test_reject_duplicate_shot_intent() -> None:
    journal = ShotJournal()
    journal.record_intent("shot-001", b'{"ShotNumber": 1}')
    with pytest.raises(ValueError, match="already exists"):
        journal.record_intent("shot-001", b'{"ShotNumber": 1}')


def test_record_ambiguity_on_disconnect() -> None:
    journal = ShotJournal()
    journal.record_intent("shot-002", b'{"ShotNumber": 2}')

    amb = journal.record_ambiguity(
        "shot-002", reason="Connection reset by peer before ACK"
    )
    assert amb.status == DeliveryStatus.AMBIGUOUS
    assert amb.error == "Connection reset by peer before ACK"


def test_record_rejection_on_simulator_error() -> None:
    journal = ShotJournal()
    journal.record_intent("shot-003", b'{"ShotNumber": 3}')

    rej = journal.record_rejection(
        "shot-003", code=501, message="Unsupported shot type"
    )
    assert rej.status == DeliveryStatus.REJECTED
    assert rej.response_code == 501
    assert rej.response_message == "Unsupported shot type"


def test_list_entries_by_status() -> None:
    journal = ShotJournal()
    journal.record_intent("shot-1", b"1")
    journal.record_intent("shot-2", b"2")
    journal.record_intent("shot-3", b"3")

    journal.record_acknowledgment("shot-1", 200, "OK")
    journal.record_ambiguity("shot-2", "Timeout")

    pending = journal.list_entries(status=DeliveryStatus.PENDING)
    assert len(pending) == 1
    assert pending[0].shot_id == "shot-3"

    acked = journal.list_entries(status=DeliveryStatus.ACKNOWLEDGED)
    assert len(acked) == 1
    assert acked[0].shot_id == "shot-1"

    ambiguous = journal.list_entries(status=DeliveryStatus.AMBIGUOUS)
    assert len(ambiguous) == 1
    assert ambiguous[0].shot_id == "shot-2"


def test_file_persistence_and_reload() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        journal_path = Path(tmpdir) / "test_journal.json"
        j1 = ShotJournal(storage_path=journal_path)
        j1.record_intent("shot-p1", b"test-payload-1")
        j1.record_acknowledgment("shot-p1", 200, "OK")
        j1.record_intent("shot-p2", b"test-payload-2")
        j1.record_ambiguity("shot-p2", "Server disconnected")

        # Reload from storage
        j2 = ShotJournal(storage_path=journal_path)
        assert len(j2.list_entries()) == 2
        p1 = j2.get_entry("shot-p1")
        assert p1.status == DeliveryStatus.ACKNOWLEDGED
        assert p1.response_code == 200
        p2 = j2.get_entry("shot-p2")
        assert p2.status == DeliveryStatus.AMBIGUOUS
        assert p2.error == "Server disconnected"
