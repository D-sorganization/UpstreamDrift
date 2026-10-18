"""NSX-43: the tab view model shows exactly what the shared golden says."""

from __future__ import annotations

import dataclasses

import pytest

from src.shared.python.loop_nullspace.plot_model import NSXPlotModel
from src.tools.grip_wrench_plots import nsx_view
from tests.unit.tools.grip_wrench_plots.golden_support import (
    golden_dict,
    golden_model,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def raw() -> dict:
    return golden_dict()


@pytest.fixture(scope="module")
def views() -> dict:
    return nsx_view.panel_views(golden_model())


def test_tabs_are_the_four_spec_panels(views) -> None:
    assert tuple(views) == ("invariants", "envelopes", "hand_split", "diagnostics")
    assert [v.title for v in views.values()] == [
        "Invariants",
        "Envelopes",
        "Hand Split",
        "Diagnostics",
    ]


def test_invariant_rows_follow_the_shown_flags(views, raw) -> None:
    rows = {r.name: r for r in views["invariants"].rows}
    for series in raw["invariants"]["series"]:
        flags = [f["shown"] for f in series["frames"]]
        expected = (
            "shown" if all(flags) else "withheld" if not any(flags) else "partial"
        )
        assert rows[series["name"]].state == expected
    assert rows["net_club_wrench"].state == "withheld"
    assert "club_ne" in rows["net_club_wrench"].reason
    assert rows["loop_power"].state == "shown"


def test_invariant_curves_carry_only_shown_numbers(views, raw) -> None:
    curves = {c.name: c for c in views["invariants"].curves}
    assert "net_club_wrench" not in curves
    assert "base_centroidal_rows" not in curves
    power = next(s for s in raw["invariants"]["series"] if s["name"] == "loop_power")
    assert curves["loop_power"].lines[0].values == tuple(
        f["values"][0] for f in power["frames"]
    )
    assert len(curves["tangent_pairing"].lines) == 3


def test_envelope_bands_and_withheld_come_from_the_model(views, raw) -> None:
    view = views["envelopes"]
    names = [c.name for c in view.curves]
    assert names == [b["quantity_id"] for b in raw["envelopes"]["bands"]]
    assert "tau_4" not in names
    rows = {r.name: r for r in view.rows}
    assert rows["tau_4"].state == "withheld"
    assert rows["tau_4"].reason == raw["envelopes"]["withheld"][0]["reason"]
    tau1 = next(c for c in view.curves if c.name == "tau_1")
    band = raw["envelopes"]["bands"][0]
    assert tau1.lower == tuple(band["lower"])
    assert tau1.upper == tuple(band["upper"])


def test_segment_display_classes_are_the_models(views, raw) -> None:
    classes = {
        (c.name, s.first_frame_index): s.display_class
        for c in views["envelopes"].curves
        for s in c.segments
    }
    expected = {
        (b["quantity_id"], s["first_frame_index"]): s["display_class"]
        for b in raw["envelopes"]["bands"]
        for s in b["segments"]
    }
    assert classes == expected
    assert {"DETERMINED_SIGN", "INDETERMINATE", "UNBOUNDED"} <= set(classes.values())


def test_overlays_sit_on_their_bands(views, raw) -> None:
    by_name = {c.name: c for c in views["envelopes"].curves}
    for ov in raw["allocation"]["overlays"]:
        assert by_name[ov["band_id"]].overlay.values == tuple(ov["values"])
    assert by_name["tau_1"].overlay is not None
    assert all(c.overlay is None for c in views["hand_split"].curves)


def test_allocation_withheld_and_non_claim_are_noted(views, raw) -> None:
    notes = views["envelopes"].notes
    assert raw["allocation"]["non_claim"] in notes
    assert any("tau_4" in n and "never shown alone" in n for n in notes)


def test_hand_split_is_routed_and_framed(views, raw) -> None:
    view = views["hand_split"]
    assert [c.name for c in view.curves] == [
        b["quantity_id"] for b in raw["hand_split"]["bands"]
    ]
    assert f"frame: {raw['hand_split']['frame']}" in view.notes


def test_badge_is_the_decisions_evidence(views, raw) -> None:
    band = raw["envelopes"]["bands"][0]["decision"]
    assert band["label"] in views["envelopes"].badge
    assert band["action"] in views["envelopes"].badge
    assert "UNVALIDATED" in views["envelopes"].badge
    assert "SHOW_VALUE" in views["invariants"].badge
    assert "SHOW_REASON" in views["invariants"].badge


def test_diagnostics_show_admission_ranks_and_residuals(views, raw) -> None:
    rows = {r.name: r for r in views["diagnostics"].rows}
    assert rows["admission"].state == raw["diagnostics"]["admission"]["verdict"]
    assert "rank A 5" in rows["frame 0"].detail
    assert rows["residual phi"].state == "withheld"
    assert "no plant" in rows["residual phi"].reason
    assert rows["residual phi_dot"].state == "shown"
    names = [c.name for c in views["diagnostics"].curves]
    assert "residual phi" not in names
    assert "residual phi_dot" in names


def test_unavailable_panel_says_why_and_draws_nothing(raw) -> None:
    data = dict(raw)
    data["envelopes"] = {
        "bands": [],
        "withheld": [],
        "frames": [],
        "unavailable_reason": "admission rejected",
    }
    data["allocation"] = {
        **raw["allocation"],
        "overlays": [],
        "withheld": [],
    }
    view = nsx_view.panel_view(NSXPlotModel.from_dict(data), "envelopes")
    assert view.status == "unavailable: admission rejected"
    assert view.curves == ()
    assert view.badge == "no quantity"


def test_evidence_text_without_a_decision_is_a_withhold_marker() -> None:
    assert nsx_view.evidence_text(None) == "no display decision"


def test_view_is_frozen_and_validates_inputs(views) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        views["envelopes"].badge = "x"  # type: ignore[misc]
    with pytest.raises(TypeError):
        nsx_view.panel_view({}, "envelopes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown NSX tab"):
        nsx_view.panel_view(golden_model(), "allocation")
