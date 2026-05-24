"""Tests for :mod:`training.runtime.wire_protocol`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import (
    MetricKind,
    RunResult,
    TrainingConfig,
    TrainingFramework,
    TrainingMetric,
    TrainingStatus,
    new_run_id,
    run_result_to_dict,
    training_config_to_dict,
    training_metric_to_dict,
)
from training.runtime.wire_protocol import (
    COMMAND_CANCEL,
    COMMAND_RUN,
    EVENT_METRIC,
    EVENT_RESULT,
    EVENT_STATUS,
    WireProtocolError,
    decode_command,
    decode_event,
    encode_command,
    encode_event,
)

pytestmark = pytest.mark.unit


def _sample_config() -> TrainingConfig:
    return TrainingConfig(
        framework=TrainingFramework.PYTORCH,
        entry_point="pkg.mod:train",
        output_dir=Path("/tmp/run"),
        hyperparameters={"lr": 0.001},
    )


def _sample_metric() -> TrainingMetric:
    return TrainingMetric(
        name="loss",
        value=0.5,
        step=3,
        timestamp=12.5,
        kind=MetricKind.LOSS,
        tags={"split": "train"},
    )


def _sample_result() -> RunResult:
    return RunResult(
        run_id=new_run_id(),
        status=TrainingStatus.COMPLETED,
        duration_s=1.25,
        final_metrics=(_sample_metric(),),
    )


class TestEncodeEvent:
    def test_encodes_status_event_with_message(self) -> None:
        line = encode_event(
            EVENT_STATUS,
            {"status": TrainingStatus.RUNNING.value, "message": "started"},
        )
        assert line.endswith("\n")
        record = json.loads(line)
        assert record == {
            "event": EVENT_STATUS,
            "status": "running",
            "message": "started",
        }

    def test_encodes_metric_event(self) -> None:
        metric_dict = training_metric_to_dict(_sample_metric())
        line = encode_event(EVENT_METRIC, {"metric": metric_dict})
        record = json.loads(line)
        assert record["event"] == EVENT_METRIC
        assert record["metric"] == metric_dict

    def test_encodes_result_event(self) -> None:
        result_dict = run_result_to_dict(_sample_result())
        line = encode_event(EVENT_RESULT, {"result": result_dict})
        record = json.loads(line)
        assert record["event"] == EVENT_RESULT
        assert record["result"] == result_dict

    def test_rejects_unknown_event(self) -> None:
        with pytest.raises(WireProtocolError, match="unknown event"):
            encode_event("nope", {})

    def test_rejects_non_dict_payload(self) -> None:
        with pytest.raises(WireProtocolError, match="payload must be a dict"):
            encode_event(EVENT_STATUS, "oops")  # type: ignore[arg-type]

    def test_rejects_payload_with_event_key(self) -> None:
        with pytest.raises(WireProtocolError, match="must not contain an 'event'"):
            encode_event(EVENT_STATUS, {"event": "x"})


class TestDecodeEvent:
    def test_round_trip_status(self) -> None:
        line = encode_event(
            EVENT_STATUS,
            {"status": "running", "message": None},
        )
        name, payload = decode_event(line)
        assert name == EVENT_STATUS
        assert payload == {"status": "running", "message": None}

    def test_round_trip_metric(self) -> None:
        metric_dict = training_metric_to_dict(_sample_metric())
        line = encode_event(EVENT_METRIC, {"metric": metric_dict})
        name, payload = decode_event(line)
        assert name == EVENT_METRIC
        assert payload == {"metric": metric_dict}

    def test_round_trip_result(self) -> None:
        result_dict = run_result_to_dict(_sample_result())
        line = encode_event(EVENT_RESULT, {"result": result_dict})
        name, payload = decode_event(line)
        assert name == EVENT_RESULT
        assert payload == {"result": result_dict}

    def test_rejects_non_string(self) -> None:
        with pytest.raises(WireProtocolError, match="line must be str"):
            decode_event(123)  # type: ignore[arg-type]

    def test_rejects_empty(self) -> None:
        with pytest.raises(WireProtocolError, match="non-empty"):
            decode_event("   ")

    def test_rejects_invalid_json(self) -> None:
        with pytest.raises(WireProtocolError, match="not valid JSON"):
            decode_event("{not json")

    def test_rejects_non_object_json(self) -> None:
        with pytest.raises(WireProtocolError, match="JSON object"):
            decode_event("[1, 2, 3]")

    def test_rejects_unknown_event_name(self) -> None:
        with pytest.raises(WireProtocolError, match="unknown or missing event"):
            decode_event(json.dumps({"event": "nope"}))

    def test_rejects_missing_event_field(self) -> None:
        with pytest.raises(WireProtocolError, match="unknown or missing event"):
            decode_event(json.dumps({"other": 1}))


class TestEncodeCommand:
    def test_encodes_run_command_with_config(self) -> None:
        config_dict = training_config_to_dict(_sample_config())
        line = encode_command(COMMAND_RUN, {"config": config_dict})
        assert line.endswith("\n")
        record = json.loads(line)
        assert record == {"command": COMMAND_RUN, "config": config_dict}

    def test_encodes_cancel_command_without_payload(self) -> None:
        line = encode_command(COMMAND_CANCEL)
        record = json.loads(line)
        assert record == {"command": COMMAND_CANCEL}

    def test_rejects_unknown_command(self) -> None:
        with pytest.raises(WireProtocolError, match="unknown command"):
            encode_command("nope")

    def test_rejects_payload_with_command_key(self) -> None:
        with pytest.raises(WireProtocolError, match="must not contain a 'command'"):
            encode_command(COMMAND_CANCEL, {"command": "x"})

    def test_rejects_non_dict_payload(self) -> None:
        with pytest.raises(WireProtocolError, match="payload must be a dict"):
            encode_command(COMMAND_CANCEL, "oops")  # type: ignore[arg-type]


class TestDecodeCommand:
    def test_round_trip_run(self) -> None:
        config_dict = training_config_to_dict(_sample_config())
        line = encode_command(COMMAND_RUN, {"config": config_dict})
        name, payload = decode_command(line)
        assert name == COMMAND_RUN
        assert payload == {"config": config_dict}

    def test_round_trip_cancel(self) -> None:
        line = encode_command(COMMAND_CANCEL)
        name, payload = decode_command(line)
        assert name == COMMAND_CANCEL
        assert payload == {}

    def test_rejects_non_string(self) -> None:
        with pytest.raises(WireProtocolError, match="line must be str"):
            decode_command(123)  # type: ignore[arg-type]

    def test_rejects_empty(self) -> None:
        with pytest.raises(WireProtocolError, match="non-empty"):
            decode_command("")

    def test_rejects_invalid_json(self) -> None:
        with pytest.raises(WireProtocolError, match="not valid JSON"):
            decode_command("{")

    def test_rejects_non_object_json(self) -> None:
        with pytest.raises(WireProtocolError, match="JSON object"):
            decode_command('"a string"')

    def test_rejects_unknown_command(self) -> None:
        with pytest.raises(WireProtocolError, match="unknown or missing command"):
            decode_command(json.dumps({"command": "nope"}))
