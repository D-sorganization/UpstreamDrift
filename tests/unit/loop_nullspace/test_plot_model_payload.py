"""``plot_model_payload`` (NSX-44, #12576): the shared report -> JSON step."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("highspy")

from src.shared.python.loop_nullspace.output_policy_integration import (  # noqa: E402
    DisplayAction,
    DisplayDecision,
)
from src.shared.python.loop_nullspace.plot_model import (  # noqa: E402
    PLOT_MODEL_SCHEMA_VERSION,
    NSXPlotModel,
    decision_key,
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


def _override(action: DisplayAction) -> dict[str, DisplayDecision]:
    return {
        decision_key("envelopes", "tau_4"): DisplayDecision(
            "research_view", action, "fixture override"
        )
    }


def test_override_may_only_tighten_to_withhold() -> None:
    with pytest.raises(ValueError, match="only withhold"):
        plot_model_payload(
            fixture_report(), decision_overrides=_override(DisplayAction.SHOW_VALUE)
        )


def test_override_of_an_unknown_key_is_rejected() -> None:
    unknown = {
        decision_key("envelopes", "tau_99"): DisplayDecision(
            "research_view", DisplayAction.WITHHOLD, "fixture override"
        )
    }
    with pytest.raises(ValueError, match="unknown decision key"):
        plot_model_payload(fixture_report(), decision_overrides=unknown)
