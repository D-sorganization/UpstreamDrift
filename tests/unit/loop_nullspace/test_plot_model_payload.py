"""``plot_model_payload`` (NSX-44, #12576): the shared report -> JSON step."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("highspy")

from src.shared.python.loop_nullspace.plot_model import (  # noqa: E402
    PLOT_MODEL_SCHEMA_VERSION,
    NSXPlotModel,
)
from src.shared.python.loop_nullspace.plot_model_decisions import (  # noqa: E402
    plot_model_payload,
)
from tests.unit.loop_nullspace.plot_model_golden import (  # noqa: E402
    GOLDEN_PATH,
    build_golden,
    fixture_report,
)

pytestmark = pytest.mark.unit


def test_reproduces_the_golden_from_its_source_report() -> None:
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert build_golden() == golden


def test_payload_is_json_safe_and_round_trips() -> None:
    payload = plot_model_payload(fixture_report(), preset="envelope_only")
    assert payload["schema_version"] == PLOT_MODEL_SCHEMA_VERSION
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    assert NSXPlotModel.from_dict(payload).to_dict() == payload


def test_preset_must_be_the_reports_own() -> None:
    with pytest.raises(ValueError, match="differs"):
        plot_model_payload(fixture_report(), preset="smooth_capacity")


def test_malformed_report_is_a_value_error() -> None:
    with pytest.raises(ValueError):
        plot_model_payload({"status": "complete"})
    report = fixture_report()
    del report["invariants"]
    with pytest.raises(ValueError):
        plot_model_payload(report)
    with pytest.raises(TypeError):
        plot_model_payload([])  # type: ignore[arg-type]
