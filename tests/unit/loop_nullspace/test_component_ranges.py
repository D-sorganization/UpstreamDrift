"""Tests for sampled feasible component-range classification (#15494, Slice B)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.shared.python.loop_nullspace.component_ranges import (
    classify_component_ranges,
)

pytestmark = pytest.mark.unit

NAMES = ("net_force", "hand_split", "wrist_torque")
UNITS = ("N", "N", "N m")


def _histories() -> np.ndarray:
    """Three candidates, 4 samples, 3 components with known spreads."""
    t = np.linspace(0.0, 1.0, 4)
    cands = []
    for k, split in enumerate((0.0, 5.0, 10.0)):
        net = 100.0 + 0.0 * t  # identical in every candidate
        hand = split + 0.0 * t  # varies widely with the candidate
        wrist = 20.0 + 0.01 * k + 0.0 * t  # varies a little
        cands.append(np.stack([net, hand, wrist], axis=1))
    return np.asarray(cands)


def _classify(**kw):
    args = {
        "names": NAMES,
        "units": UNITS,
        "atols": (1e-9,),
        "declared_invariant": ("net_force",),
        "prior_bounds": {"wrist_torque": (0.0, 100.0)},
    }
    args.update(kw)
    return classify_component_ranges(_histories(), **args)


def _by_name(report):
    return {c["name"]: c for c in report["components"]}


def test_invariant_narrow_and_variable_are_separated():
    comps = _by_name(_classify(feasible_bounds={"wrist_torque": (19.99, 20.05)}))
    assert comps["net_force"]["classification"] == "MODEL_INVARIANT"
    assert comps["wrist_torque"]["classification"] == "PRIOR_CONSTRAINED_NARROW"
    assert comps["hand_split"]["classification"] == "VARIABLE"


def test_small_sampled_spread_in_broad_prior_is_not_prior_constrained():
    # Codex FIX-NEEDED on #15611: two nearby samples in a 100 N prior say
    # nothing about the feasible interval, so no causal label may be given.
    report = classify_component_ranges(
        [[[50.0]], [[50.01]]],
        names=["force"],
        units=["N"],
        atols=[1e-9],
        prior_bounds={"force": (0.0, 100.0)},
    )
    comp = report["components"][0]
    assert comp["classification"] == "SAMPLED_SMALL_SPREAD"
    assert comp["classification_by_atol"] == ["SAMPLED_SMALL_SPREAD"]


def test_small_spread_without_any_prior_is_variable():
    comp = classify_component_ranges(
        [[[50.0]], [[50.01]]], names=["f"], units=["N"], atols=[1e-9]
    )["components"][0]
    assert comp["classification"] == "VARIABLE"


def test_wide_feasible_bounds_do_not_earn_the_narrow_label():
    comps = _by_name(_classify(feasible_bounds={"wrist_torque": (0.0, 100.0)}))
    assert comps["wrist_torque"]["classification"] == "SAMPLED_SMALL_SPREAD"


@pytest.mark.parametrize(
    "bounds, match",
    [
        ({"wrist_torque": (20.5, 21.0)}, "outside"),  # samples escape it
        ({"wrist_torque": (21.0, 19.0)}, "lo < hi"),
        ({"wrist_torque": (19.0, float("inf"))}, "lo < hi"),
        ({"nope": (0.0, 1.0)}, "unknown"),
    ],
)
def test_bad_feasible_bounds_are_rejected(bounds, match):
    with pytest.raises(ValueError, match=match):
        _classify(feasible_bounds=bounds)


def test_feasible_bounds_must_sit_inside_the_prior():
    with pytest.raises(ValueError, match="prior"):
        _classify(
            prior_bounds={"wrist_torque": (19.0, 21.0)},
            feasible_bounds={"wrist_torque": (18.0, 20.5)},
        )


def test_feasible_bounds_require_a_prior_to_judge_narrowness():
    with pytest.raises(ValueError, match="prior"):
        _classify(prior_bounds={}, feasible_bounds={"wrist_torque": (19.9, 20.1)})


def test_ranges_and_width_match_numpy_oracle():
    comps = _by_name(_classify())
    hand = comps["hand_split"]
    assert hand["min"] == [0.0] * 4
    assert hand["max"] == [10.0] * 4
    assert hand["max_width"] == pytest.approx(10.0)
    assert hand["units"] == "N"


def test_unmoved_without_declaration_is_not_called_invariant():
    comps = _by_name(_classify(declared_invariant=()))
    assert comps["net_force"]["classification"] == "UNMOVED_IN_SAMPLES"


def test_declared_invariant_that_moves_is_reported_violated():
    comps = _by_name(_classify(declared_invariant=("net_force", "hand_split")))
    assert comps["hand_split"]["classification"] == "INVARIANT_VIOLATED"
    assert _classify(declared_invariant=("hand_split",))["ok"] is False


def test_tolerance_dependence_is_flagged():
    # wrist spread is 0.02: unmoved at 0.05, moving at 1e-9.
    report = _classify(
        atols=(1e-9, 0.05),
        declared_invariant=("net_force",),
        feasible_bounds={"wrist_torque": (19.99, 20.05)},
    )
    wrist = _by_name(report)["wrist_torque"]
    assert wrist["classification_by_atol"] == [
        "PRIOR_CONSTRAINED_NARROW",
        "UNMOVED_IN_SAMPLES",
    ]
    assert wrist["tolerance_dependent"] is True
    assert _by_name(report)["net_force"]["tolerance_dependent"] is False


def test_report_is_json_serialisable_and_carries_caveats():
    report = _classify()
    json.dumps(report)
    assert report["pointwise_bounds_jointly_reachable"] is False
    assert report["candidate_count"] == 3
    assert report["sample_count"] == 4


@pytest.mark.parametrize(
    "mutate",
    [
        lambda h: h[:1],  # a single candidate has no range
        lambda h: np.where(np.arange(h.size).reshape(h.shape) == 3, np.nan, h),
        lambda h: h[:, :, :2],  # component count mismatch with names
    ],
)
def test_invalid_histories_are_rejected(mutate):
    with pytest.raises(ValueError):
        classify_component_ranges(
            mutate(_histories()),
            names=NAMES,
            units=UNITS,
            atols=(1e-9,),
        )


@pytest.mark.parametrize("atols", [(), (0.0,), (-1.0,), (float("nan"),)])
def test_bad_tolerances_are_rejected(atols):
    with pytest.raises(ValueError):
        _classify(atols=atols)


def test_samples_outside_declared_prior_are_rejected_not_clipped():
    with pytest.raises(ValueError, match="outside"):
        _classify(prior_bounds={"wrist_torque": (0.0, 10.0)})


def test_unknown_declared_names_are_rejected():
    with pytest.raises(ValueError, match="unknown"):
        _classify(declared_invariant=("nope",))
    with pytest.raises(ValueError, match="unknown"):
        _classify(prior_bounds={"nope": (0.0, 1.0)})
