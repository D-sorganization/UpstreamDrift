"""NSX-41: display_decision decision table, fail-closed default, delegation."""

from __future__ import annotations

import copy
import itertools
from enum import Enum

import numpy as np
import pytest

from src.shared.python.loop_nullspace.evidence import (
    AllocationDependence,
    InputProvenance,
    ModelProvenance,
    QuantityRecord,
    ValidationStatus,
)
from src.shared.python.loop_nullspace.output_policy_integration import (
    DisplayAction,
    DisplayDecision,
    DisplaySurface,
    display_decision,
)
from src.shared.python.loop_nullspace.registry import load_raw_registry, parse_registry
from src.shared.python.motion_matching.output_policy import (
    evaluate_output_policy,
)

pytestmark = pytest.mark.unit

VERIFIED_HYP = "H-NOISE-SIGNAL-DEPENDENT"
VERIFIED_COND = "H-DECLARED-JOINT-IMPEDANCE"
BAD_ID = "H-TEST-UNVERIFIED"
PRESET = "envelope_only"
_MODEL = ModelProvenance("planar-fixture", "1")
_INPUT = InputProvenance("synthetic", "unit-test")


def _registry():
    raw = copy.deepcopy(load_raw_registry())
    raw["hypotheses"].append(
        {
            "id": BAD_ID,
            "statement": "Test-only unverified prior.",
            "source": {"citation_id": None, "note": "test"},
            "verification_status": "UNVERIFIED",
        }
    )
    return parse_registry(raw)


REG = _registry()


def _record(dep, validated, hyp, cond):
    unavailable = dep is AllocationDependence.UNAVAILABLE
    return QuantityRecord(
        values=np.array([np.nan] if unavailable else [1.0]),
        units="N m",
        allocation_dependence=dep,
        model_provenance=_MODEL,
        input_provenance=_INPUT,
        validation_status=(
            ValidationStatus.VALIDATED if validated else ValidationStatus.UNVALIDATED
        ),
        hypotheses=hyp,
        conditioning=cond,
        reason="no closure data" if unavailable else None,
    )


def _policy(kind):
    if kind == "none":
        return None
    out = {"joint_loads": {}}
    if kind == "admits":
        out = {
            "joint_loads": {
                "validated": True,
                "calibration": "cal-1",
                "evidence": "ev-1",
            }
        }
    return evaluate_output_policy({"outputs": out})


def _expected(dep, validated, hyp_ok, cond_ok, surface, policy, representative, preset):
    """Independent statement of the contract (not the implementation)."""
    if surface not in ("research_view", "swing_report"):
        return DisplayAction.WITHHOLD
    unavailable = dep is AllocationDependence.UNAVAILABLE
    if surface == "swing_report":
        if policy != "admits" or unavailable or not validated:
            return DisplayAction.WITHHOLD
    if unavailable:
        return DisplayAction.SHOW_REASON
    if not (hyp_ok and cond_ok):
        return DisplayAction.WITHHOLD
    if dep is AllocationDependence.INVARIANT:
        return DisplayAction.SHOW_VALUE
    if preset == "unknown":
        return DisplayAction.WITHHOLD
    if representative:
        if preset == "none":
            return DisplayAction.WITHHOLD
        return DisplayAction.SHOW_REPRESENTATIVE_ON_BAND
    return DisplayAction.SHOW_BAND


_CELLS = list(
    itertools.product(
        list(AllocationDependence),
        [False, True],
        [True, False],
        [True, False],
        ["research_view", "swing_report", "nonsense_surface"],
        ["none", "withholds", "admits"],
        [False, True],
        ["none", "good", "unknown"],
    )
)
# Hypotheses exist only on FAMILY_DEPENDENT records, so the "hypothesis not
# verified" axis is meaningful for that value alone.
_CELLS = [c for c in _CELLS if c[0] is AllocationDependence.FAMILY_DEPENDENT or c[2]]


@pytest.mark.parametrize("dep,validated,hyp_ok,cond_ok,surface,pol,rep,preset", _CELLS)
def test_decision_table_every_cell(
    dep, validated, hyp_ok, cond_ok, surface, pol, rep, preset
):
    is_fam = dep is AllocationDependence.FAMILY_DEPENDENT
    hyp = ((VERIFIED_HYP if hyp_ok else BAD_ID),) if is_fam else ()
    cond = (VERIFIED_COND if cond_ok else BAD_ID,)
    rec = _record(dep, validated, hyp, cond)
    statuses = _policy(pol)
    kwargs = {
        "registry": REG,
        "representative": rep,
        "preset": {"none": None, "good": PRESET, "unknown": "no-such-preset"}[preset],
    }
    if statuses is not None:
        kwargs.update(output_name="joint_loads", policy_statuses=statuses)
    got = display_decision(rec, surface, **kwargs)
    want = _expected(
        dep, validated, hyp_ok or not is_fam, cond_ok, surface, pol, rep, preset
    )
    assert isinstance(got, DisplayDecision)
    assert got.action is want, got.reason
    assert got.reason


def test_unknown_surface_withholds_even_for_best_record():
    rec = _record(AllocationDependence.INVARIANT, True, (), ())
    d = display_decision(rec, "export_csv", registry=REG)
    assert d.action is DisplayAction.WITHHOLD
    assert "export_csv" in d.reason


def test_unknown_allocation_axis_value_withholds():
    class Fake(Enum):
        WEIRD = "WEIRD"

    rec = _record(AllocationDependence.INVARIANT, True, (), ())
    object.__setattr__(rec, "allocation_dependence", Fake.WEIRD)
    assert (
        display_decision(rec, DisplaySurface.RESEARCH_VIEW, registry=REG).action
        is DisplayAction.WITHHOLD
    )


def test_invariant_shows_model_provenance_badge():
    rec = _record(AllocationDependence.INVARIANT, False, (), ())
    d = display_decision(rec, DisplaySurface.RESEARCH_VIEW, registry=REG)
    assert d.action is DisplayAction.SHOW_VALUE
    assert any("planar-fixture" in b for b in d.badges)
    assert d.conditioned is False
    assert d.conditioning_chips == () and d.hypothesis_chips == ()
    assert "unconditioned" in d.label


def test_conditioning_chips_on_invariant_distinct_from_hypothesis_chips():
    rec = _record(AllocationDependence.INVARIANT, False, (), (VERIFIED_COND,))
    d = display_decision(rec, DisplaySurface.RESEARCH_VIEW, registry=REG)
    assert d.action is DisplayAction.SHOW_VALUE
    assert d.conditioning_chips == (VERIFIED_COND,)
    assert d.hypothesis_chips == ()
    assert d.conditioned is True
    assert "conditioned on" in d.label and "unconditioned" not in d.label


def test_family_chips_are_separate_axes():
    rec = _record(
        AllocationDependence.FAMILY_DEPENDENT, False, (VERIFIED_HYP,), (VERIFIED_COND,)
    )
    d = display_decision(rec, DisplaySurface.RESEARCH_VIEW, registry=REG)
    assert d.action is DisplayAction.SHOW_BAND
    assert d.hypothesis_chips == (VERIFIED_HYP,)
    assert d.conditioning_chips == (VERIFIED_COND,)


@pytest.mark.parametrize("bad", [BAD_ID, "H-NOT-IN-REGISTRY"])
def test_unverified_conditioning_gated_like_hypotheses(bad):
    inv = _record(AllocationDependence.INVARIANT, False, (), (bad,))
    fam = _record(AllocationDependence.FAMILY_DEPENDENT, False, (bad,), ())
    di = display_decision(inv, DisplaySurface.RESEARCH_VIEW, registry=REG)
    df = display_decision(fam, DisplaySurface.RESEARCH_VIEW, registry=REG)
    assert di.action is df.action is DisplayAction.WITHHOLD
    assert bad in di.reason and bad in df.reason
    with pytest.raises(ValueError):
        REG.require_verified_for_display([bad])


def test_representative_is_labelled_with_preset():
    rec = _record(AllocationDependence.FAMILY_DEPENDENT, False, (VERIFIED_HYP,), ())
    d = display_decision(
        rec, "research_view", registry=REG, representative=True, preset=PRESET
    )
    assert d.action is DisplayAction.SHOW_REPRESENTATIVE_ON_BAND
    assert d.preset == PRESET
    assert PRESET in d.label and "representative" in d.label


def test_unavailable_renders_reason():
    rec = _record(AllocationDependence.UNAVAILABLE, False, (), ())
    d = display_decision(rec, "research_view", registry=REG)
    assert d.action is DisplayAction.SHOW_REASON
    assert "no closure data" in d.reason


def test_swing_report_without_policy_context_withholds():
    rec = _record(AllocationDependence.INVARIANT, True, (), ())
    d = display_decision(rec, DisplaySurface.SWING_REPORT, registry=REG)
    assert d.action is DisplayAction.WITHHOLD
    assert "policy" in d.reason


def test_swing_report_tracks_real_output_policy():
    rec = _record(AllocationDependence.INVARIANT, True, (), ())
    no_claim = evaluate_output_policy({})
    assert no_claim["joint_loads"].status == "withheld"
    claim = {"joint_loads": {"validated": True, "calibration": "c", "evidence": "e"}}
    ok = evaluate_output_policy({"outputs": claim})
    assert ok["joint_loads"].status == "validated"
    kw = {"registry": REG, "output_name": "joint_loads"}
    held = display_decision(rec, "swing_report", policy_statuses=no_claim, **kw)
    shown = display_decision(rec, "swing_report", policy_statuses=ok, **kw)
    assert held.action is DisplayAction.WITHHOLD
    assert shown.action is DisplayAction.SHOW_VALUE
    other = display_decision(
        rec,
        "swing_report",
        policy_statuses=ok,
        registry=REG,
        output_name="ball_flight",
    )
    assert other.action is DisplayAction.WITHHOLD
    missing = display_decision(
        rec, "swing_report", policy_statuses=ok, registry=REG, output_name="nope"
    )
    assert missing.action is DisplayAction.WITHHOLD


def test_swing_report_admits_advertised_output():
    rec = _record(AllocationDependence.INVARIANT, True, (), ())
    st = evaluate_output_policy({})
    assert st["clubhead_speed"].status == "advertised"
    d = display_decision(
        rec,
        "swing_report",
        policy_statuses=st,
        output_name="clubhead_speed",
        registry=REG,
    )
    assert d.action is DisplayAction.SHOW_VALUE


def test_research_surface_ignores_policy():
    rec = _record(AllocationDependence.INVARIANT, False, (), ())
    d = display_decision(
        rec,
        "research_view",
        policy_statuses=evaluate_output_policy({}),
        output_name="joint_loads",
        registry=REG,
    )
    assert d.action is DisplayAction.SHOW_VALUE


def test_policy_arguments_must_come_together():
    rec = _record(AllocationDependence.INVARIANT, True, (), ())
    with pytest.raises(ValueError):
        display_decision(rec, "swing_report", output_name="joint_loads", registry=REG)
    with pytest.raises(ValueError):
        display_decision(
            rec, "swing_report", policy_statuses=_policy("admits"), registry=REG
        )


@pytest.mark.parametrize(
    "kwargs,exc",
    [
        ({"record": "not-a-record"}, TypeError),
        ({"surface": 3}, TypeError),
        ({"surface": None}, TypeError),
        ({"representative": "yes"}, TypeError),
        ({"preset": 5}, TypeError),
        ({"preset": ""}, ValueError),
        ({"registry": object()}, TypeError),
        ({"output_name": 4, "policy_statuses": {}}, TypeError),
        ({"output_name": "", "policy_statuses": {}}, ValueError),
        ({"output_name": "x", "policy_statuses": [1]}, TypeError),
        ({"output_name": "x", "policy_statuses": {"x": "admitted"}}, TypeError),
    ],
)
def test_wrong_types_raise(kwargs, exc):
    args = {
        "record": _record(AllocationDependence.INVARIANT, True, (), ()),
        "surface": "research_view",
        "registry": REG,
    }
    args.update(kwargs)
    with pytest.raises(exc):
        display_decision(**args)


def test_decision_is_frozen_and_self_consistent():
    rec = _record(AllocationDependence.INVARIANT, False, (), (VERIFIED_COND,))
    d = display_decision(rec, "research_view", registry=REG)
    with pytest.raises(AttributeError):
        d.action = DisplayAction.WITHHOLD  # type: ignore[misc]
    with pytest.raises(ValueError):
        DisplayDecision(
            surface="research_view",
            action=DisplayAction.SHOW_VALUE,
            reason="x",
            conditioning_chips=("A",),
            conditioned=False,
        )
    with pytest.raises(ValueError):
        DisplayDecision(surface="s", action=DisplayAction.WITHHOLD, reason="")
    with pytest.raises(TypeError):
        DisplayDecision(surface="s", action="SHOW_VALUE", reason="x")  # type: ignore[arg-type]


def test_default_registry_is_shipped_registry():
    rec = _record(AllocationDependence.INVARIANT, False, (), (VERIFIED_COND,))
    assert display_decision(rec, "research_view").action is DisplayAction.SHOW_VALUE
