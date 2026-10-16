"""JSON payload tests for the Spine Viewer core (SPN-15c, #12815). No Qt."""

from __future__ import annotations

import json

import pytest

from src.tools.spine_viewer import core

from .conftest import SOURCE

pytestmark = pytest.mark.unit


def test_payload_is_json_serialisable_and_carries_status(study_factory) -> None:
    view = core.open_study(study_factory())
    payload = core.view_to_dict(view)
    json.dumps(payload, allow_nan=False)
    assert payload["scientific_status"] == "unqualified"
    assert payload["distribution_source"] == SOURCE
    assert payload["banner"] == core.status_banner(view)
    assert payload["levels"] == ["L5_S1", "L4_L5"]


def test_payload_heatmaps_reuse_view_heatmap(study_factory) -> None:
    view = core.open_study(study_factory())
    payload = core.view_to_dict(view)
    assert set(payload["heatmaps"]) == set(core.LOAD_QUANTITIES)
    comp = payload["heatmaps"]["compression"]
    expected = core.view_heatmap(view, "compression")
    assert comp["values"] == expected.values.tolist()
    assert comp["time_s"] == expected.time_s.tolist()
    assert comp["units"] == expected.units
    assert comp["status_label"] == "UNQUALIFIED"
    assert comp["distribution_source"] == SOURCE
    assert payload["heatmaps"]["tension"]["values"][2][0] == -300.0


def test_payload_qualified_only_when_receipt_earned_it(study_factory) -> None:
    ok = core.view_to_dict(
        core.open_study(study_factory(status="pass", scientific_status="qualified"))
    )
    assert ok["scientific_status"] == "qualified"
    bad = core.view_to_dict(
        core.open_study(
            study_factory(status="unqualified", scientific_status="qualified")
        )
    )
    assert bad["scientific_status"] == "unqualified"
    assert bad["status_reasons"] == ["invented reason"]


def test_payload_rejects_non_view() -> None:
    with pytest.raises(TypeError):
        core.view_to_dict("nope")  # type: ignore[arg-type]
