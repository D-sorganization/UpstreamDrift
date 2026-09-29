"""Tests for :mod:`training.runtime.progress_sinks`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import (
    MetricKind,
    TrainingMetric,
    TrainingStatus,
)
from training.contracts import ProgressSink
from training.runtime import (
    CompositeProgressSink,
    InMemoryProgressSink,
    JsonlFileProgressSink,
    NullProgressSink,
    RealtimeChannelProgressSink,
    training_channel_for,
)

pytestmark = pytest.mark.unit


def _metric() -> TrainingMetric:
    return TrainingMetric(
        name="loss",
        value=0.5,
        step=1,
        timestamp=100.0,
        kind=MetricKind.LOSS,
    )


class TestNullProgressSink:
    def test_drops_emissions(self) -> None:
        sink = NullProgressSink()
        sink.emit_metric(_metric())
        sink.emit_status(TrainingStatus.RUNNING)

    def test_satisfies_protocol(self) -> None:
        assert isinstance(NullProgressSink(), ProgressSink)


class TestInMemoryProgressSink:
    def test_records_metrics(self) -> None:
        sink = InMemoryProgressSink()
        sink.emit_metric(_metric())
        assert len(sink.metrics) == 1
        assert sink.metrics[0] == _metric()

    def test_records_status(self) -> None:
        sink = InMemoryProgressSink()
        sink.emit_status(TrainingStatus.RUNNING, message="started")
        assert sink.statuses == ((TrainingStatus.RUNNING, "started"),)

    def test_clear(self) -> None:
        sink = InMemoryProgressSink()
        sink.emit_metric(_metric())
        sink.emit_status(TrainingStatus.RUNNING)
        sink.clear()
        assert sink.metrics == ()
        assert sink.statuses == ()

    def test_rejects_non_metric(self) -> None:
        sink = InMemoryProgressSink()
        with pytest.raises(TypeError):
            sink.emit_metric({"name": "loss"})  # type: ignore[arg-type]

    def test_rejects_non_status(self) -> None:
        sink = InMemoryProgressSink()
        with pytest.raises(TypeError):
            sink.emit_status("running")  # type: ignore[arg-type]


class TestJsonlFileProgressSink:
    def test_writes_metric_line(self, tmp_path: Path) -> None:
        sink = JsonlFileProgressSink(tmp_path / "metrics.jsonl")
        sink.emit_metric(_metric())
        content = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8")
        assert content.count("\n") == 1
        parsed = json.loads(content.splitlines()[0])
        assert parsed["name"] == "loss"
        assert parsed["value"] == 0.5
        assert parsed["kind"] == "loss"

    def test_appends_multiple_metrics(self, tmp_path: Path) -> None:
        sink = JsonlFileProgressSink(tmp_path / "metrics.jsonl")
        for i in range(3):
            sink.emit_metric(
                TrainingMetric(name="loss", value=float(i), step=i, timestamp=0.0)
            )
        lines = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3

    def test_writes_status_to_separate_file(self, tmp_path: Path) -> None:
        sink = JsonlFileProgressSink(tmp_path / "metrics.jsonl")
        sink.emit_status(TrainingStatus.RUNNING, message="kicking off")
        assert sink.status_path.exists()
        status_line = json.loads(
            sink.status_path.read_text(encoding="utf-8").splitlines()[0]
        )
        assert status_line["event"] == "status"
        assert status_line["status"] == "running"
        assert status_line["message"] == "kicking off"

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "metrics.jsonl"
        JsonlFileProgressSink(nested)
        assert nested.parent.is_dir()

    def test_custom_status_path(self, tmp_path: Path) -> None:
        metrics = tmp_path / "m.jsonl"
        status = tmp_path / "s.jsonl"
        sink = JsonlFileProgressSink(metrics, status_path=status)
        sink.emit_status(TrainingStatus.RUNNING)
        assert status.exists()

    def test_rejects_non_path_metrics(self) -> None:
        with pytest.raises(TypeError):
            JsonlFileProgressSink("metrics.jsonl")  # type: ignore[arg-type]


class TestCompositeProgressSink:
    def test_fan_out_metric(self) -> None:
        a = InMemoryProgressSink()
        b = InMemoryProgressSink()
        composite = CompositeProgressSink((a, b))
        composite.emit_metric(_metric())
        assert len(a.metrics) == 1
        assert len(b.metrics) == 1

    def test_fan_out_status(self) -> None:
        a = InMemoryProgressSink()
        b = InMemoryProgressSink()
        composite = CompositeProgressSink((a, b))
        composite.emit_status(TrainingStatus.RUNNING, message="m")
        assert a.statuses == ((TrainingStatus.RUNNING, "m"),)
        assert b.statuses == ((TrainingStatus.RUNNING, "m"),)

    def test_rejects_empty_iterable(self) -> None:
        with pytest.raises(ValueError):
            CompositeProgressSink(())

    def test_error_does_not_block_other_sinks(self) -> None:
        class _BrokenSink:
            def emit_metric(self, metric: TrainingMetric) -> None:
                raise OSError("broken")

            def emit_status(
                self, status: TrainingStatus, *, message: str | None = None
            ) -> None:
                raise OSError("broken")

        good = InMemoryProgressSink()
        composite = CompositeProgressSink((_BrokenSink(), good))
        with pytest.raises(OSError):
            composite.emit_metric(_metric())
        # Despite the broken sink raising, the good sink still received the metric.
        assert len(good.metrics) == 1


class TestTrainingChannelHelper:
    def test_channel_name_format(self) -> None:
        assert training_channel_for("abc123") == "training/abc123/progress"

    def test_rejects_empty_job_id(self) -> None:
        with pytest.raises(ValueError):
            training_channel_for("")

    def test_rejects_non_string(self) -> None:
        with pytest.raises(ValueError):
            training_channel_for(None)  # type: ignore[arg-type]


class TestRealtimeChannelProgressSink:
    def test_emits_metric_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        published: list[tuple[str, dict[str, object]]] = []

        def fake_publish(channel: str, payload: object, transport=None) -> None:
            published.append((channel, payload))  # type: ignore[arg-type]

        def fake_register(name: str, description: str, owner_tool_id=None) -> None:
            return None

        import src.shared.python.realtime as realtime  # noqa: PLC0415 - test setup

        monkeypatch.setattr(realtime, "publish", fake_publish)
        monkeypatch.setattr(realtime, "register_channel", fake_register)
        sink = RealtimeChannelProgressSink("job-abc")
        sink.emit_metric(_metric())
        assert len(published) == 1
        channel, payload = published[0]
        assert channel == "training/job-abc/progress"
        assert payload["event"] == "metric"
        assert payload["metric"]["name"] == "loss"

    def test_emits_status_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        published: list[tuple[str, dict[str, object]]] = []

        def fake_publish(channel: str, payload: object, transport=None) -> None:
            published.append((channel, payload))  # type: ignore[arg-type]

        def fake_register(name: str, description: str, owner_tool_id=None) -> None:
            return None

        import src.shared.python.realtime as realtime  # noqa: PLC0415

        monkeypatch.setattr(realtime, "publish", fake_publish)
        monkeypatch.setattr(realtime, "register_channel", fake_register)
        sink = RealtimeChannelProgressSink("job-xyz")
        sink.emit_status(TrainingStatus.RUNNING, message="hello")
        assert published[0][1]["event"] == "status"
        assert published[0][1]["status"] == "running"
        assert published[0][1]["message"] == "hello"

    def test_satisfies_protocol(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import src.shared.python.realtime as realtime  # noqa: PLC0415

        monkeypatch.setattr(realtime, "publish", lambda *a, **kw: None)
        monkeypatch.setattr(realtime, "register_channel", lambda *a, **kw: None)
        sink = RealtimeChannelProgressSink("job-protocol")
        assert isinstance(sink, ProgressSink)
