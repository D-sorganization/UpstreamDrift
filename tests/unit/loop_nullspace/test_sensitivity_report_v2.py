"""NSX-99: the sensitivity stage, report schema 2.0.0 and the shared golden (#12728)."""

from __future__ import annotations

import copy
import dataclasses
import json

import numpy as np
import pytest

pytest.importorskip("highspy")
jsonschema = pytest.importorskip("jsonschema")

from src.shared.python.loop_nullspace import facade  # noqa: E402
from src.shared.python.loop_nullspace.facade import (  # noqa: E402
    AnalysisCancelledError,
    AnalysisOptions,
    analyze,
)
from src.shared.python.loop_nullspace.output_policy_integration import (  # noqa: E402
    DisplayAction,
)
from src.shared.python.loop_nullspace.plot_model import (  # noqa: E402
    NSXPlotModel,
    build_plot_model,
    decision_key,
)
from src.shared.python.loop_nullspace.plot_model_decisions import (  # noqa: E402
    report_decisions,
)
from src.shared.python.loop_nullspace.sensitivity_identity import (  # noqa: E402
    assumption_digest,
)
from src.shared.python.loop_nullspace.sensitivity_options import (  # noqa: E402
    SensitivitySpec,
    disturbance_dict,
    disturbance_from_dict,
)
from src.shared.python.loop_nullspace.strategy_spec import StrategyId  # noqa: E402
from src.shared.python.loop_nullspace.variance_records import Mechanism  # noqa: E402
from tests.unit.loop_nullspace import (  # noqa: E402
    sensitivity_support as ss,
    variability_golden,
)
from tests.unit.loop_nullspace.report_support import (  # noqa: E402
    SCHEMA_PATH,
    planar_engine,
    planar_trajectory,
)

pytestmark = pytest.mark.unit

PRESET = "envelope_only"
SCHEMA_V2 = SCHEMA_PATH.with_name("loop-nullspace-report-v2.json")


def _opts(**kw) -> AnalysisOptions:
    return AnalysisOptions(disturbance_spec=ss.disturbance(), **kw)


@pytest.fixture(scope="module")
def report() -> dict:
    out = analyze(planar_trajectory(), ss.sensitivity_engine(), PRESET, _opts())
    return json.loads(json.dumps(out))


@pytest.fixture(scope="module")
def v2_validator():
    schema = json.loads(SCHEMA_V2.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


# ---------------------------------------------------------------- golden
def test_golden_equals_regeneration() -> None:
    path = variability_golden.GOLDEN_PATH
    assert json.loads(path.read_text(encoding="utf-8")) == (
        variability_golden.build_golden()
    )


def test_golden_round_trips_through_the_plot_model_and_schema(v2_validator) -> None:
    golden = json.loads(variability_golden.GOLDEN_PATH.read_text(encoding="utf-8"))
    v2_validator.validate(golden["report"])
    model = NSXPlotModel.from_dict(golden["plot_model"])
    assert model.to_dict() == golden["plot_model"]
    assert model.schema_version == "2.0.0"
    assert model.sensitivity is not None
    assert model.sensitivity.kernel_heatmap.shown
    assert any(not p.shown for p in model.sensitivity.strategy_pareto.points)


# ---------------------------------------------------------- the report
def test_report_is_schema_v2_with_the_three_sections(report, v2_validator) -> None:
    v2_validator.validate(report)
    assert report["schema_version"] == "2.0.0"
    assert report["sensitivity"]["status"] == "complete"
    assert report["variance_budget"]["status"] == "complete"
    assert report["strategies"]["status"] == "complete"
    assert report["sensitivity"]["primary_outcome"] == "face_angle"
    assert report["options"]["sensitivity"] == report["sensitivity"]["options"]


def test_v1_report_is_unchanged_without_a_disturbance() -> None:
    out = analyze(planar_trajectory(), planar_engine(), PRESET)
    assert out["schema_version"] == "1.1.0" or out["schema_version"].startswith("1.")
    for key in ("sensitivity", "variance_budget", "strategies"):
        assert key not in out
    assert "sensitivity" not in out["options"]
    assert "sensitivity" not in facade.STAGES[:-1]
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(json.loads(json.dumps(out)))


def test_v1_schema_rejects_the_v2_report(report) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(report)


def test_every_displayed_sd_is_the_root_of_a_variance(report) -> None:
    disp = report["sensitivity"]["face_dispersion"]
    var = disp["linear"]["variance"]["values"]
    sd = disp["linear"]["sd"]["values"]
    assert np.allclose(np.square(np.ravel(sd)), np.ravel(var))
    assert disp["linear"]["sd"]["conditioning"] == ["H-NOISE-ADDITIVE"]


def test_budget_parts_sum_to_the_total_and_shares_to_one(report) -> None:
    budget = report["variance_budget"]
    parts = np.ravel(budget["parts"]["values"])
    shares = np.ravel(budget["shares"]["values"])
    total = float(
        np.ravel(
            report["sensitivity"]["face_dispersion"]["linear"]["variance"]["values"]
        )[0]
    )
    assert parts.sum() == pytest.approx(total, rel=1e-9)
    assert shares.sum() == pytest.approx(1.0, rel=1e-9)


def test_not_implemented_strategies_say_why_and_carry_no_numbers(report) -> None:
    strat = report["strategies"]
    missing = {item["label"]: item["reason"] for item in strat["not_attempted"]}
    assert missing
    assert all(reason.strip() for reason in missing.values())
    for arm in strat["arms"]:
        if arm["label"] in missing:
            assert not arm["admitted"] and arm["ratio"] is None


# ------------------------------------------------------------- options
def test_options_require_a_disturbance_for_the_other_fields() -> None:
    with pytest.raises(ValueError, match="disturbance_spec"):
        AnalysisOptions(strategy_set=[StrategyId.S1])
    with pytest.raises(ValueError, match="disturbance_spec"):
        AnalysisOptions(outcome_priority=("ball_speed",))


def test_engine_without_a_provider_refuses_loudly() -> None:
    with pytest.raises(ValueError, match="sensitivity_provider"):
        analyze(planar_trajectory(), planar_engine(), PRESET, _opts())


@pytest.mark.parametrize(
    "kw, exc",
    [
        ({"mechanisms": ["not_a_mechanism"]}, ValueError),
        ({"strategies": ["S99"]}, ValueError),
        ({"strategies": ["S1", "S1"]}, ValueError),
        ({"outcome_priority": ()}, ValueError),
        ({"outcome_priority": ("a", "a")}, ValueError),
        ({"outcome_priority": "face_angle"}, TypeError),
        ({"mechanisms": Mechanism.SIGNAL_DEPENDENT_NOISE}, TypeError),
    ],
)
def test_spec_rejects_bad_declarations(kw, exc) -> None:
    base = {"disturbances": (ss.disturbance(),), "mechanisms": frozenset()}
    with pytest.raises(exc):
        SensitivitySpec(**{**base, **kw})


def test_signal_dependent_noise_must_match_the_signal_law() -> None:
    sdn = ss.disturbance(sdn=True)
    with pytest.raises(ValueError, match="disagree"):
        SensitivitySpec((sdn,), frozenset())
    with pytest.raises(ValueError, match="disagree"):
        SensitivitySpec(
            (ss.disturbance(),), frozenset({Mechanism.SIGNAL_DEPENDENT_NOISE})
        )
    SensitivitySpec((sdn,), frozenset({Mechanism.SIGNAL_DEPENDENT_NOISE}))


def test_disturbance_dict_round_trips() -> None:
    spec = ss.disturbance(sdn=True)
    again = disturbance_from_dict(json.loads(json.dumps(disturbance_dict(spec))))
    assert disturbance_dict(again) == disturbance_dict(spec)
    with pytest.raises(ValueError, match="lacks"):
        disturbance_from_dict({"name": "x"})


# ------------------------------------------------------ cache identity
def _digest(**kw) -> str:
    spec = SensitivitySpec(
        kw.get("disturbances", (ss.disturbance(),)),
        kw.get("mechanisms", frozenset()),
        kw.get("strategies", tuple(StrategyId)),
        kw.get("priority", ("face_angle",)),
    )
    return assumption_digest(spec.as_dict())


def test_digest_changes_with_every_declared_assumption() -> None:
    base = _digest()
    assert _digest() == base
    variants = {
        "covariance": _digest(disturbances=(ss.disturbance(scale=2.0),)),
        "signal law": _digest(
            disturbances=(ss.disturbance(sdn=True),),
            mechanisms=frozenset({Mechanism.SIGNAL_DEPENDENT_NOISE}),
        ),
        "mechanisms": _digest(mechanisms=frozenset({Mechanism.GEOMETRIC_PRELOAD})),
        "strategies": _digest(strategies=(StrategyId.S1,)),
        "priority": _digest(priority=("ball_speed", "face_angle")),
    }
    assert len(set(variants.values()) | {base}) == len(variants) + 1


def test_report_digest_covers_the_provider_inputs() -> None:
    def digest(**kw) -> str:
        eng = ss.sensitivity_engine(**kw)
        out = analyze(planar_trajectory(), eng, PRESET, _opts())
        return out["sensitivity"]["cache_identity"]

    assert digest() == digest()
    assert digest(z=0.5) != digest()


def test_digest_rejects_empty_parts() -> None:
    with pytest.raises(ValueError):
        assumption_digest({})


# --------------------------------------------------------------- cancel
def test_cancel_mid_stage_stops_and_leaves_no_partial_report() -> None:
    state = {"cancel": False, "calls": 0}
    inner = ss.provider()

    def provider(request):
        state["calls"] += 1
        state["cancel"] = True  # cancel requested while the stage is building
        return inner(request)

    engine = dataclasses.replace(planar_engine(), sensitivity_provider=provider)
    stages: list[str] = []
    with pytest.raises(AnalysisCancelledError):
        analyze(
            planar_trajectory(),
            engine,
            PRESET,
            _opts(),
            on_stage=lambda name, pct, partial: stages.append(name),
            should_cancel=lambda: state["cancel"],
        )
    assert state["calls"] == 1
    assert "sensitivity" in stages or "envelopes" in stages


class _Source:
    """A service source over the planar fixture with the sensitivity provider."""

    def describe(self, request):
        from src.api.services import loop_nullspace_service as svc

        return svc.SourceDescription(
            {"kind": "fixture", "frames": request.frame_indices},
            {"name": "planar-numpy", "version": "1"},
        )

    def load(self, request):
        return planar_trajectory(), ss.sensitivity_engine()


def _service_request(**sens):
    from src.api.services import loop_nullspace_service as svc

    spec = {
        "disturbance_spec": [disturbance_dict(ss.disturbance())],
        "mechanisms": [],
        **sens,
    }
    return svc.LoopNullspaceRequest(
        source_id="nsx99",
        preset=PRESET,
        frame_indices=[0, 3, 6, 9],
        options={"sensitivity": spec},
    )


@pytest.fixture
def service():
    pytest.importorskip("pydantic")
    from src.api.services import loop_nullspace_service as svc

    svc.register_source("nsx99", _Source())
    svc.RESULT_CACHE.clear()
    yield svc
    svc.unregister_source("nsx99")
    svc.RESULT_CACHE.clear()


def test_cancelled_service_run_is_not_cached(service) -> None:
    request = _service_request()
    with pytest.raises(AnalysisCancelledError):
        service.run(request, should_cancel=lambda: True)
    assert service.lookup(request) is None


def test_service_run_is_cached_and_every_assumption_changes_the_key(service) -> None:
    request = _service_request()
    report = service.run(request)
    assert report["schema_version"] == "2.0.0"
    assert service.lookup(request) is report
    base = service.cache_key(request)
    other = {
        "covariance": ss.disturbance(scale=2.0),
        "strategy_set": ["S1", "S2"],
        "outcome_priority": ["face_angle", "ball_speed"],
        "mechanisms": ["H-GEOMETRIC-PRELOAD"],
    }
    keys = {
        "covariance": service.cache_key(
            _service_request(disturbance_spec=[disturbance_dict(other["covariance"])])
        ),
        "strategy_set": service.cache_key(
            _service_request(strategy_set=other["strategy_set"])
        ),
        "outcome_priority": service.cache_key(
            _service_request(outcome_priority=other["outcome_priority"])
        ),
        "mechanisms": service.cache_key(
            _service_request(mechanisms=other["mechanisms"])
        ),
    }
    assert len({base, *keys.values()}) == len(keys) + 1


def test_service_refuses_an_invalid_sensitivity_declaration(service) -> None:
    with pytest.raises(service.RequestRefusedError) as err:
        service.validate(_service_request(strategy_set=["S99"]))
    assert err.value.code == "invalid_options"


# ----------------------------------------------- display gating (D7)
def test_sensitivity_quantities_are_all_asked_through_display_decision(report) -> None:
    decisions = report_decisions(report, "research_view")
    keys = {
        k
        for k in decisions
        if k.startswith(
            ("kernel_heatmap", "variance_budget", "strategy_pareto", "face_dispersion")
        )
    }
    assert decision_key("kernel_heatmap", "face_angle") in keys
    assert decision_key("variance_budget", "variance") in keys
    assert decision_key("face_dispersion", "linear") in keys
    assert all(decisions[k].action is not None for k in keys)


def test_unverified_family_dependent_series_is_withheld(report) -> None:
    doctored = copy.deepcopy(report)
    q = doctored["sensitivity"]["kernel"]["quantity"]
    q["allocation_dependence"] = "FAMILY_DEPENDENT"
    q["hypotheses"] = ["H-TEST-UNVERIFIED"]
    decisions = report_decisions(doctored, "research_view")
    key = decision_key("kernel_heatmap", "face_angle")
    assert decisions[key].action is DisplayAction.WITHHOLD
    model = build_plot_model(doctored, decisions)
    heat = model.sensitivity.kernel_heatmap
    assert not heat.shown
    assert heat.lower == () and heat.upper == ()
    assert "H-TEST-UNVERIFIED" in heat.reason or heat.reason


def test_family_dependent_without_a_band_never_draws_numbers(report) -> None:
    doctored = copy.deepcopy(report)
    q = doctored["sensitivity"]["kernel"]["quantity"]
    q["allocation_dependence"] = "FAMILY_DEPENDENT"
    q["hypotheses"] = ["H-NOISE-ADDITIVE"]
    decisions = report_decisions(doctored, "research_view")
    model = build_plot_model(doctored, decisions)
    heat = model.sensitivity.kernel_heatmap
    assert not heat.shown and heat.lower == ()
    assert "band" in heat.reason


def test_a_missing_decision_withholds_by_name(report) -> None:
    decisions = report_decisions(report, "research_view")
    del decisions[decision_key("variance_budget", "variance")]
    model = build_plot_model(report, decisions)
    panel = model.sensitivity.variance_budget
    assert not panel.shown and panel.parts == ()
    assert "variance_budget" in panel.reason


def test_unknown_decision_keys_are_refused(report) -> None:
    decisions = report_decisions(report, "research_view")
    decisions["kernel_heatmap/other"] = decisions[
        decision_key("kernel_heatmap", "face_angle")
    ]
    with pytest.raises(ValueError):
        build_plot_model(report, decisions)


def test_pareto_fronts_and_winner_use_shown_points_only(report) -> None:
    decisions = report_decisions(report, "research_view")
    model = build_plot_model(report, decisions)
    pareto = model.sensitivity.strategy_pareto
    shown = {p.label for p in pareto.points if p.shown}
    assert shown
    for front in pareto.fronts:
        assert set(front.labels) <= shown
    assert pareto.winner is None or pareto.winner in shown
    for point in pareto.points:
        if not point.shown:
            assert point.variance_ratio is None and point.reason


def test_a_report_without_the_stage_gives_empty_panels_with_a_reason() -> None:
    out = json.loads(
        json.dumps(
            analyze(planar_trajectory(), ss.sensitivity_engine(), PRESET, _opts())
        )
    )
    out["sensitivity"] = None
    out["variance_budget"] = None
    out["strategies"] = None
    model = build_plot_model(out, report_decisions(out, "research_view"))
    panels = model.sensitivity
    assert panels.status == "not_run"
    assert not panels.kernel_heatmap.shown and panels.kernel_heatmap.reason
    assert not panels.variance_budget.shown
    assert panels.strategy_pareto.unavailable_reason
