"""Tests for DiagnosticsCollector, DiagnosticsSnapshot, DiagnosticsHistory — Epic #5698.

Covers:
- DiagnosticsSnapshot structure and fields
- SystemMetrics and SimulationMetrics construction
- DiagnosticsCollector.collect() return type and structure
- active_simulations_fn callback integration
- DiagnosticsHistory ring-buffer: record, get_recent, capacity, overflow
- History clear / len / get_all
- JSON serialisation roundtrip
- DbC precondition enforcement
- Snapshot immutability (frozen dataclass)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.shared.python.diagnostics import (
    DiagnosticsCollector,
    DiagnosticsHistory,
    DiagnosticsSnapshot,
    SimulationMetrics,
    SystemMetrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_snapshot(
    cpu: float = 10.0,
    mem_used: float = 512.0,
    mem_total: float = 16384.0,
    mem_pct: float = 3.1,
    active: int = 2,
) -> DiagnosticsSnapshot:
    return DiagnosticsSnapshot(
        timestamp=datetime.now(tz=timezone.utc),
        system_metrics=SystemMetrics(
            cpu_percent=cpu,
            memory_used_mb=mem_used,
            memory_total_mb=mem_total,
            memory_percent=mem_pct,
        ),
        simulation_metrics=SimulationMetrics(active_simulations=active),
    )


# ---------------------------------------------------------------------------
# SystemMetrics
# ---------------------------------------------------------------------------


class TestSystemMetrics:
    def test_defaults(self) -> None:
        sm = SystemMetrics()
        assert sm.cpu_percent == -1.0
        assert sm.memory_used_mb == -1.0
        assert sm.open_file_handles == -1

    def test_create_with_values(self) -> None:
        sm = SystemMetrics(cpu_percent=42.0, memory_used_mb=1024.0)
        assert sm.cpu_percent == 42.0
        assert sm.memory_used_mb == 1024.0

    def test_frozen_immutable(self) -> None:
        sm = SystemMetrics(cpu_percent=10.0)
        with pytest.raises((AttributeError, TypeError)):
            sm.cpu_percent = 99.0  # type: ignore[misc]

    def test_invalid_cpu_type_raises(self) -> None:
        with pytest.raises((ValueError, TypeError, Exception)):
            SystemMetrics(cpu_percent="high")  # type: ignore[arg-type]

    def test_invalid_open_file_handles_type_raises(self) -> None:
        with pytest.raises((ValueError, TypeError, Exception)):
            SystemMetrics(open_file_handles=3.5)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# SimulationMetrics
# ---------------------------------------------------------------------------


class TestSimulationMetrics:
    def test_defaults(self) -> None:
        sm = SimulationMetrics()
        assert sm.active_simulations == -1
        assert sm.registered_engines == 0

    def test_create_with_values(self) -> None:
        sm = SimulationMetrics(active_simulations=3, registered_engines=5)
        assert sm.active_simulations == 3
        assert sm.registered_engines == 5

    def test_negative_registered_engines_raises(self) -> None:
        with pytest.raises((ValueError, Exception)):
            SimulationMetrics(registered_engines=-1)

    def test_frozen_immutable(self) -> None:
        sm = SimulationMetrics(active_simulations=2)
        with pytest.raises((AttributeError, TypeError)):
            sm.active_simulations = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DiagnosticsSnapshot
# ---------------------------------------------------------------------------


class TestDiagnosticsSnapshot:
    def test_create_snapshot(self) -> None:
        snap = _make_snapshot()
        assert isinstance(snap.timestamp, datetime)
        assert isinstance(snap.system_metrics, SystemMetrics)
        assert isinstance(snap.simulation_metrics, SimulationMetrics)

    def test_snapshot_is_frozen(self) -> None:
        snap = _make_snapshot()
        with pytest.raises((AttributeError, TypeError)):
            snap.system_metrics = SystemMetrics()  # type: ignore[misc]

    def test_extra_defaults_to_empty_dict(self) -> None:
        snap = _make_snapshot()
        assert snap.extra == {}

    def test_extra_custom_values(self) -> None:
        snap = DiagnosticsSnapshot(
            timestamp=datetime.now(tz=timezone.utc),
            system_metrics=SystemMetrics(),
            simulation_metrics=SimulationMetrics(),
            extra={"foo": "bar"},
        )
        assert snap.extra["foo"] == "bar"

    def test_to_dict_returns_dict(self) -> None:
        snap = _make_snapshot()
        d = snap.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_timestamp_is_string(self) -> None:
        snap = _make_snapshot()
        d = snap.to_dict()
        assert isinstance(d["timestamp"], str)

    def test_to_dict_contains_system_metrics(self) -> None:
        snap = _make_snapshot(cpu=55.5)
        d = snap.to_dict()
        assert "system_metrics" in d
        assert d["system_metrics"]["cpu_percent"] == 55.5

    def test_to_json_is_valid_json(self) -> None:
        snap = _make_snapshot()
        raw = snap.to_json()
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)

    def test_to_json_roundtrip_cpu(self) -> None:
        snap = _make_snapshot(cpu=77.7)
        parsed = json.loads(snap.to_json())
        assert parsed["system_metrics"]["cpu_percent"] == pytest.approx(77.7)

    def test_invalid_timestamp_type_raises(self) -> None:
        with pytest.raises((ValueError, TypeError, Exception)):
            DiagnosticsSnapshot(
                timestamp="not-a-datetime",  # type: ignore[arg-type]
                system_metrics=SystemMetrics(),
                simulation_metrics=SimulationMetrics(),
            )

    def test_invalid_system_metrics_type_raises(self) -> None:
        with pytest.raises((ValueError, TypeError, Exception)):
            DiagnosticsSnapshot(
                timestamp=datetime.now(tz=timezone.utc),
                system_metrics={"cpu": 10},  # type: ignore[arg-type]
                simulation_metrics=SimulationMetrics(),
            )


# ---------------------------------------------------------------------------
# DiagnosticsCollector
# ---------------------------------------------------------------------------


class TestDiagnosticsCollector:
    def test_collect_returns_snapshot(self) -> None:
        collector = DiagnosticsCollector()
        snap = collector.collect()
        assert isinstance(snap, DiagnosticsSnapshot)

    def test_collect_timestamp_is_utc(self) -> None:
        collector = DiagnosticsCollector()
        snap = collector.collect()
        assert snap.timestamp.tzinfo is not None

    def test_collect_system_metrics_populated(self) -> None:
        collector = DiagnosticsCollector()
        snap = collector.collect()
        assert isinstance(snap.system_metrics, SystemMetrics)

    def test_collect_simulation_metrics_populated(self) -> None:
        collector = DiagnosticsCollector()
        snap = collector.collect()
        assert isinstance(snap.simulation_metrics, SimulationMetrics)

    def test_active_simulations_fn_used(self) -> None:
        collector = DiagnosticsCollector(active_simulations_fn=lambda: 7)
        snap = collector.collect()
        assert snap.simulation_metrics.active_simulations == 7

    def test_active_simulations_fn_zero(self) -> None:
        collector = DiagnosticsCollector(active_simulations_fn=lambda: 0)
        snap = collector.collect()
        assert snap.simulation_metrics.active_simulations == 0

    def test_active_simulations_fn_none_yields_minus_one(self) -> None:
        collector = DiagnosticsCollector(active_simulations_fn=None)
        snap = collector.collect()
        assert snap.simulation_metrics.active_simulations == -1

    def test_active_simulations_fn_must_be_callable(self) -> None:
        with pytest.raises((ValueError, Exception)):
            DiagnosticsCollector(active_simulations_fn=42)  # type: ignore[arg-type]

    def test_collect_multiple_times_returns_distinct_snapshots(self) -> None:
        collector = DiagnosticsCollector()
        s1 = collector.collect()
        s2 = collector.collect()
        assert s1 is not s2

    def test_failing_active_simulations_fn_falls_back_gracefully(self) -> None:
        def bad_fn() -> int:
            raise RuntimeError("engine unavailable")

        collector = DiagnosticsCollector(active_simulations_fn=bad_fn)
        snap = collector.collect()  # Must not raise
        assert snap.simulation_metrics.active_simulations == -1

    def test_collect_without_psutil_still_returns_snapshot(self) -> None:
        """When psutil is unavailable the collector should degrade gracefully."""
        with patch(
            "src.shared.python.diagnostics._collector._try_import_psutil",
            return_value=None,
        ):
            collector = DiagnosticsCollector()
            snap = collector.collect()
        assert isinstance(snap, DiagnosticsSnapshot)
        assert snap.system_metrics.cpu_percent == -1.0


# ---------------------------------------------------------------------------
# DiagnosticsHistory
# ---------------------------------------------------------------------------


class TestDiagnosticsHistory:
    def test_initial_len_is_zero(self) -> None:
        h = DiagnosticsHistory()
        assert len(h) == 0

    def test_default_capacity(self) -> None:
        h = DiagnosticsHistory()
        assert h.capacity == 100

    def test_custom_capacity(self) -> None:
        h = DiagnosticsHistory(capacity=10)
        assert h.capacity == 10

    def test_record_increases_len(self) -> None:
        h = DiagnosticsHistory()
        h.record(_make_snapshot())
        assert len(h) == 1

    def test_record_multiple(self) -> None:
        h = DiagnosticsHistory()
        for _ in range(5):
            h.record(_make_snapshot())
        assert len(h) == 5

    def test_ring_buffer_overflow_evicts_oldest(self) -> None:
        h = DiagnosticsHistory(capacity=3)
        snaps = [_make_snapshot() for _ in range(5)]
        for s in snaps:
            h.record(s)
        assert len(h) == 3
        # The 3 most recent should be retained
        retained = h.get_all()
        assert retained == snaps[-3:]

    def test_len_never_exceeds_capacity(self) -> None:
        h = DiagnosticsHistory(capacity=5)
        for _ in range(20):
            h.record(_make_snapshot())
        assert len(h) <= 5

    def test_get_recent_returns_list(self) -> None:
        h = DiagnosticsHistory()
        h.record(_make_snapshot())
        result = h.get_recent(1)
        assert isinstance(result, list)

    def test_get_recent_zero_returns_empty(self) -> None:
        h = DiagnosticsHistory()
        h.record(_make_snapshot())
        assert h.get_recent(0) == []

    def test_get_recent_one(self) -> None:
        h = DiagnosticsHistory()
        s = _make_snapshot()
        h.record(s)
        result = h.get_recent(1)
        assert len(result) == 1
        assert result[0] is s

    def test_get_recent_n_larger_than_history(self) -> None:
        h = DiagnosticsHistory()
        h.record(_make_snapshot())
        result = h.get_recent(1000)
        assert len(result) == 1

    def test_get_recent_preserves_order(self) -> None:
        h = DiagnosticsHistory()
        snaps = [_make_snapshot(cpu=float(i)) for i in range(5)]
        for s in snaps:
            h.record(s)
        recent = h.get_recent(5)
        assert [s.system_metrics.cpu_percent for s in recent] == [
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
        ]

    def test_get_all_returns_all(self) -> None:
        h = DiagnosticsHistory()
        snaps = [_make_snapshot() for _ in range(4)]
        for s in snaps:
            h.record(s)
        assert h.get_all() == snaps

    def test_clear_empties_history(self) -> None:
        h = DiagnosticsHistory()
        for _ in range(5):
            h.record(_make_snapshot())
        h.clear()
        assert len(h) == 0
        assert h.get_all() == []

    def test_record_after_clear(self) -> None:
        h = DiagnosticsHistory()
        for _ in range(3):
            h.record(_make_snapshot())
        h.clear()
        s = _make_snapshot()
        h.record(s)
        assert len(h) == 1
        assert h.get_all()[0] is s

    def test_invalid_capacity_zero_raises(self) -> None:
        with pytest.raises((ValueError, Exception)):
            DiagnosticsHistory(capacity=0)

    def test_invalid_capacity_negative_raises(self) -> None:
        with pytest.raises((ValueError, Exception)):
            DiagnosticsHistory(capacity=-5)

    def test_record_non_snapshot_raises(self) -> None:
        h = DiagnosticsHistory()
        with pytest.raises((ValueError, TypeError, Exception)):
            h.record({"not": "a snapshot"})  # type: ignore[arg-type]

    def test_get_recent_negative_n_raises(self) -> None:
        h = DiagnosticsHistory()
        with pytest.raises((ValueError, Exception)):
            h.get_recent(-1)

    def test_capacity_one_ring_buffer(self) -> None:
        h = DiagnosticsHistory(capacity=1)
        s1 = _make_snapshot(cpu=1.0)
        s2 = _make_snapshot(cpu=2.0)
        h.record(s1)
        h.record(s2)
        assert len(h) == 1
        assert h.get_all()[0] is s2
