"""NSX-42: the framework-free plot model and its cross-UI golden (#12574)."""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("highspy")
jsonschema = pytest.importorskip("jsonschema")

from src.shared.python.loop_nullspace import facade  # noqa: E402
from src.shared.python.loop_nullspace.facade import AnalysisOptions, analyze  # noqa: E402
from src.shared.python.loop_nullspace.output_policy_integration import (  # noqa: E402
    DisplayAction,
    DisplayDecision,
)
from src.shared.python.loop_nullspace.plot_model import (  # noqa: E402
    DisplayClass,
    NSXPlotModel,
    build_plot_model,
    decision_key,
)
from src.shared.python.loop_nullspace.plot_model_decisions import (  # noqa: E402
    report_decisions,
)
from src.shared.python.loop_nullspace.registry import load_registry  # noqa: E402
from src.shared.python.loop_nullspace.solvers import solve_lp  # noqa: E402
from tests.unit.loop_nullspace import plot_model_golden  # noqa: E402
from tests.unit.loop_nullspace.report_support import (  # noqa: E402
    load_schema,
    planar_engine,
    planar_trajectory,
)

pytestmark = pytest.mark.unit

PRESET = "envelope_only"


@pytest.fixture(scope="module")
def report() -> dict:
    return analyze(
        planar_trajectory(),
        planar_engine(),
        PRESET,
        AnalysisOptions(include_allocation=True),
    )


@pytest.fixture(scope="module")
def decisions(report) -> dict[str, DisplayDecision]:
    return report_decisions(report, "research_view")


def _withhold(reason: str = "test: withheld") -> DisplayDecision:
    return DisplayDecision("research_view", DisplayAction.WITHHOLD, reason)


def _all_bands(model: NSXPlotModel):
    return (*model.envelopes.bands, *model.hand_split.bands)


def _band(model: NSXPlotModel, name: str):
    return next(b for b in _all_bands(model) if b.quantity_id == name)


# ------------------------------------------------------------------ report
def test_report_extension_validates_and_carries_bound_evidence(report) -> None:
    jsonschema.validate(report, load_schema(), cls=jsonschema.Draft202012Validator)
    bounds = report["envelopes"]["bounds"]["tau_1"]
    n = len(report["trajectory"]["analyzed_frame_indices"])
    for key in (
        "sign_class",
        "binding_lower",
        "binding_upper",
        "binding_hypotheses_lower",
        "binding_hypotheses_upper",
    ):
        assert len(bounds[key]) == n
    assert report["frame_diagnostics"][0]["rank_A"] >= 1
    assert len(report["frame_diagnostics"]) == n


# ------------------------------------------------------------------ golden
def test_golden_equals_regeneration() -> None:
    checked_in = json.loads(plot_model_golden.GOLDEN_PATH.read_text(encoding="utf-8"))
    assert checked_in == plot_model_golden.build_golden()


def test_golden_plot_model_round_trips_and_covers_every_display_class() -> None:
    golden = json.loads(plot_model_golden.GOLDEN_PATH.read_text(encoding="utf-8"))
    model = NSXPlotModel.from_dict(golden["plot_model"])
    assert model.to_dict() == golden["plot_model"]
    classes = {s.display_class for b in _all_bands(model) for s in b.segments}
    assert classes == set(DisplayClass)
    assert model.hand_split.bands and model.allocation.overlays
    assert model.envelopes.withheld


# ------------------------------------------- representative never alone
def test_overlay_always_sits_on_its_band(report, decisions) -> None:
    model = build_plot_model(report, decisions)
    assert model.allocation.overlays
    ids = {b.quantity_id for b in _all_bands(model)}
    for overlay in model.allocation.overlays:
        assert overlay.band_id in ids
        assert overlay.frame_indices == _band(model, overlay.band_id).frame_indices


def test_constructed_lone_representative_raises(report, decisions) -> None:
    model = build_plot_model(report, decisions)
    data = model.to_dict()
    data["envelopes"]["bands"] = []
    data["hand_split"]["bands"] = []
    with pytest.raises(ValueError, match="band"):
        NSXPlotModel.from_dict(data)


def test_overlay_without_a_shown_band_decision_raises(report, decisions) -> None:
    data = build_plot_model(report, decisions).to_dict()
    data["allocation"]["overlays"][0]["decision"]["action"] = "WITHHOLD"
    with pytest.raises(ValueError, match="SHOW_REPRESENTATIVE_ON_BAND"):
        NSXPlotModel.from_dict(data)


def test_withheld_band_takes_its_representative_with_it(report, decisions) -> None:
    dec = dict(decisions)
    dec[decision_key("envelopes", "tau_1")] = _withhold("band withheld by test")
    model = build_plot_model(report, dec)
    assert all(o.band_id != "tau_1" for o in model.allocation.overlays)
    gone = [w for w in model.allocation.withheld if w.quantity_id == "tau_1"]
    assert gone
    assert "band" in gone[0].reason
    assert any(o.band_id == "tau_2" for o in model.allocation.overlays)


def test_representative_decision_must_admit_a_representative(report, decisions) -> None:
    dec = dict(decisions)
    key = decision_key("allocation", "torque")
    dec[key] = decisions[decision_key("envelopes", "tau_1")]  # SHOW_BAND only
    model = build_plot_model(report, dec)
    assert model.allocation.overlays == ()
    assert model.allocation.withheld


# ------------------------------------------------- binding from the duals
def test_binding_ids_equal_active_constraints_from_the_duals(report) -> None:
    from src.shared.python.loop_nullspace import registry

    traj, engine = planar_trajectory(), planar_engine()
    hyp = registry.load_registry().preset(PRESET)
    options = AnalysisOptions()
    kept, decs, _ = facade._decompose_all(traj, options)
    idx = traj.frame_indices
    model = build_plot_model(report, report_decisions(report, "research_view"))
    checked = 0
    for pos, dec in zip(kept, decs, strict=True):
        fs = facade._frame_set(dec, traj.snapshots[pos], idx[pos], hyp, engine)
        form = fs.to_lp()
        a_ub, b_ub = form.A_ub, form.b_ub
        for j in range(dec.n_actuated):
            band = _band(model, f"tau_{j + 1}")
            seg = next(
                s
                for s in band.segments
                if s.first_frame_index <= idx[pos] <= s.last_frame_index
            )
            support_rows, support_hyps = set(), set()
            for sign in (+1.0, -1.0):
                res = solve_lp(
                    sign * dec.N[j], None, None, a_ub, b_ub, [(None, None)] * fs.n_z
                )
                duals = np.abs(np.asarray(res.row_dual, dtype=float))
                for r, d in zip(fs.rows, duals, strict=True):
                    if d > 1e-9:
                        support_rows.add(r.label)
                        support_hyps.add(r.hypothesis_id)
            # Complementary slackness: every row with a non-zero dual is active,
            # and the reported ids are exactly the active (binding) rows.
            assert support_rows <= set(seg.active_constraints)
            assert support_hyps <= set(seg.binding_hypotheses)
            checked += 1
    assert checked >= 4
    for band in model.envelopes.bands:
        for seg in band.segments:
            assert seg.binding_hypotheses == tuple(sorted(seg.binding_hypotheses))
            assert seg.active_constraints


def test_binding_ids_are_the_reports_active_rows(report, decisions) -> None:
    model = build_plot_model(report, decisions)
    bounds = report["envelopes"]["bounds"]["tau_2"]
    first = model.envelopes.bands[1].segments[0]
    rows = set(bounds["binding_lower"][0]) | set(bounds["binding_upper"][0])
    hyps = set(bounds["binding_hypotheses_lower"][0]) | set(
        bounds["binding_hypotheses_upper"][0]
    )
    assert set(first.active_constraints) == rows
    assert set(first.binding_hypotheses) == hyps


# --------------------------------------------------------- display classes
def _edited(report, name, frame, **changes):
    out = copy.deepcopy(report)
    b = out["envelopes"]["bounds"][name]
    for key, value in changes.items():
        b[key][frame] = value
    return out


def test_partial_unbounded_records_reach_plot_model(report) -> None:
    """#13150: mixed bounded and UNBOUNDED frames render without SHOW_REASON."""
    from src.shared.python.loop_nullspace import registry
    from src.shared.python.loop_nullspace.envelopes import (
        BoundStatus,
        static_envelopes,
    )
    from src.shared.python.loop_nullspace.report_serialization import envelope_dict

    traj, engine = planar_trajectory(), planar_engine()
    hyp = registry.load_registry().preset(PRESET)
    kept, decs, _ = facade._decompose_all(traj, AnalysisOptions())
    snaps = [traj.snapshots[p] for p in kept]
    idx = [traj.frame_indices[p] for p in kept]
    sets = [
        facade._frame_set(d, s, i, hyp, engine)
        for d, s, i in zip(decs, snaps, idx, strict=True)
    ]
    series = static_envelopes(sets, facade._quantities(sets, snaps, engine))
    upper = np.array(series.upper, copy=True)
    upper[1, 0] = np.nan
    status_upper = list(series.status_upper)
    row = list(status_upper[1])
    row[0] = BoundStatus.UNBOUNDED
    status_upper[1] = tuple(row)
    series = dataclasses.replace(series, upper=upper, status_upper=tuple(status_upper))
    records = series.records(engine.model_provenance, traj.provenance)
    rep = copy.deepcopy(report)
    rep["envelopes"] = envelope_dict(series, records, idx, {})
    decisions = report_decisions(rep, "research_view")
    band_key = decision_key("envelopes", "tau_1")
    assert decisions[band_key].action is DisplayAction.SHOW_BAND
    model = build_plot_model(rep, decisions)
    classes = [s.display_class for s in _band(model, "tau_1").segments]
    assert DisplayClass.UNBOUNDED in classes


def test_display_class_is_one_of_three(report, decisions) -> None:
    rep = _edited(report, "tau_1", 0, sign_class="positive")
    rep = _edited(rep, "tau_1", 1, status_upper="UNBOUNDED", upper=None)
    model = build_plot_model(rep, decisions)
    classes = [s.display_class for s in _band(model, "tau_1").segments]
    assert set(classes) <= set(DisplayClass)
    assert classes[0] is DisplayClass.DETERMINED_SIGN
    assert classes[1] is DisplayClass.UNBOUNDED
    assert classes[2] is DisplayClass.INDETERMINATE
    assert {c.value for c in DisplayClass} == {
        "DETERMINED_SIGN",
        "INDETERMINATE",
        "UNBOUNDED",
    }


def test_from_dict_rejects_a_fourth_class(report, decisions) -> None:
    data = build_plot_model(report, decisions).to_dict()
    data["envelopes"]["bands"][0]["segments"][0]["display_class"] = "MAYBE"
    with pytest.raises(ValueError):
        NSXPlotModel.from_dict(data)


def test_infeasible_frames_are_listed_not_segmented(report, decisions) -> None:
    rep = copy.deepcopy(report)
    rep["envelopes"]["frames"][1]["feasible"] = False
    rep["envelopes"]["frames"][1]["conflict_hypotheses"] = ["H-FRICTION-CONE"]
    model = build_plot_model(rep, decisions)
    band = model.envelopes.bands[0]
    frame = band.frame_indices[1]
    assert frame in band.infeasible_frame_indices
    assert all(
        not (s.first_frame_index <= frame <= s.last_frame_index) for s in band.segments
    )
    assert model.envelopes.frames[1].feasible is False


# --------------------------------------------------- decisions are obeyed
def test_withheld_quantities_carry_no_numbers(report, decisions) -> None:
    dec = dict(decisions)
    dec[decision_key("envelopes", "tau_3")] = _withhold("band withheld by test")
    first = report["invariants"]["frames"][0]["index"]
    dec[decision_key("invariants", "loop_power", first)] = _withhold("no loop power")
    model = build_plot_model(report, dec)
    assert all(b.quantity_id != "tau_3" for b in model.envelopes.bands)
    item = next(w for w in model.envelopes.withheld if w.quantity_id == "tau_3")
    assert item.reason == "band withheld by test"
    entry = next(s for s in model.invariants.series if s.name == "loop_power")
    frame = next(f for f in entry.frames if f.frame_index == first)
    assert (frame.shown, frame.values) == (False, ())
    assert frame.reason == "no loop power"
    text = json.dumps(model.to_dict())
    lo = report["envelopes"]["bounds"]["tau_3"]["lower"][0]
    assert repr(lo) not in text


def test_every_shown_item_has_a_shown_decision(report, decisions) -> None:
    model = build_plot_model(report, decisions).to_dict()
    shown_actions = {"SHOW_BAND", "SHOW_REPRESENTATIVE_ON_BAND", "SHOW_VALUE"}
    for band in model["envelopes"]["bands"]:
        assert band["decision"]["action"] in shown_actions
    for series in model["invariants"]["series"]:
        for f in series["frames"]:
            assert f["shown"] == (
                f["decision"]["action"] == "SHOW_VALUE" and bool(f["values"])
            )


def test_missing_decision_withholds_by_name(report, decisions) -> None:
    dec = {
        k: v for k, v in decisions.items() if k != decision_key("envelopes", "tau_4")
    }
    model = build_plot_model(report, dec)
    item = next(w for w in model.envelopes.withheld if w.quantity_id == "tau_4")
    assert "no display decision" in item.reason


def test_show_reason_decision_emits_only_the_reason(report, decisions) -> None:
    dec = dict(decisions)
    dec[decision_key("envelopes", "tau_2")] = DisplayDecision(
        "research_view", DisplayAction.SHOW_REASON, "unavailable: test"
    )
    model = build_plot_model(report, dec)
    assert all(b.quantity_id != "tau_2" for b in model.envelopes.bands)
    item = next(w for w in model.envelopes.withheld if w.quantity_id == "tau_2")
    assert item.reason == "unavailable: test"
    assert item.action == "SHOW_REASON"


# ------------------------------------------------------- hand split panel
def test_hand_split_quantities_route_to_their_panel(report, decisions) -> None:
    name = "s1_axial_pull_positive_squeeze_negative"
    rep = copy.deepcopy(report)
    env = rep["envelopes"]
    env["quantities"][name] = copy.deepcopy(env["quantities"]["tau_1"])
    env["bounds"][name] = copy.deepcopy(env["bounds"]["tau_1"])
    dec = dict(decisions)
    dec[decision_key("hand_split", name)] = decisions[
        decision_key("envelopes", "tau_1")
    ]
    model = build_plot_model(rep, dec)
    assert [b.quantity_id for b in model.hand_split.bands] == [name]
    assert all(b.quantity_id != name for b in model.envelopes.bands)
    assert model.hand_split.frame == "grip frame"


# ----------------------------------------------------- diagnostics / status
def test_diagnostics_carry_rank_condition_and_residuals(report, decisions) -> None:
    diag = build_plot_model(report, decisions).diagnostics
    assert diag.frames[0].rank_A == report["frame_diagnostics"][0]["rank_A"]
    assert diag.frames[0].condition_number > 0
    assert diag.admission is not None and diag.admission.verdict == "ADMITTED"
    assert [r.name for r in diag.residuals] == ["phi", "phi_dot", "phi_ddot"]
    assert diag.kkt is not None


def test_non_complete_report_yields_an_honest_empty_model() -> None:
    rep = analyze(planar_trajectory((0, 3)), planar_engine(), PRESET, AnalysisOptions())
    assert rep["status"] == "insufficient_frames"
    model = build_plot_model(rep, {})
    assert model.status == "insufficient_frames"
    assert model.reason
    assert model.envelopes.bands == () and model.envelopes.unavailable_reason
    assert model.invariants.series == ()


def test_registry_hypotheses_chips_pass_through(report, decisions) -> None:
    band = build_plot_model(report, decisions).envelopes.bands[0]
    assert set(band.decision.hypothesis_chips) <= {
        h.id for h in load_registry().hypotheses
    }
    assert band.decision.badges


# ----------------------------------------------------- serialisation / DbC
def test_round_trip_and_json_safety(report, decisions) -> None:
    model = build_plot_model(report, decisions)
    data = model.to_dict()
    text = json.dumps(data, allow_nan=False)
    assert NSXPlotModel.from_dict(json.loads(text)) == model
    assert model.to_dict() == data


def test_from_dict_rejects_unknown_and_missing_keys(report, decisions) -> None:
    data = build_plot_model(report, decisions).to_dict()
    with pytest.raises(ValueError, match="unknown"):
        NSXPlotModel.from_dict({**data, "extra": 1})
    short = {k: v for k, v in data.items() if k != "diagnostics"}
    with pytest.raises(ValueError, match="missing"):
        NSXPlotModel.from_dict(short)


def test_type_errors(report, decisions) -> None:
    with pytest.raises(TypeError, match="report"):
        build_plot_model([], decisions)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="display_decisions"):
        build_plot_model(report, [])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="DisplayDecision"):
        build_plot_model(report, {"envelopes/tau_1": "SHOW_BAND"})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="from_dict"):
        NSXPlotModel.from_dict([])  # type: ignore[arg-type]


def test_value_errors(report, decisions) -> None:
    with pytest.raises(ValueError, match="unknown display decision"):
        build_plot_model(report, {**decisions, "envelopes/nope": _withhold()})
    old = copy.deepcopy(report)
    del old["envelopes"]["bounds"]["tau_1"]["sign_class"]
    with pytest.raises(ValueError, match="sign_class"):
        build_plot_model(old, decisions)
    with pytest.raises(ValueError, match="decision_key"):
        decision_key("nope", "x")
    with pytest.raises(ValueError, match="frame_index"):
        decision_key("envelopes", "tau_1", 3)
    with pytest.raises(ValueError, match="frame_index"):
        decision_key("invariants", "loop_power")


def test_module_is_framework_free() -> None:
    import ast

    import src.shared.python.loop_nullspace.plot_model as pm

    tree = ast.parse(Path(pm.__file__).read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert not roots & {"PyQt6", "PyQt5", "matplotlib", "numpy", "scipy"}
