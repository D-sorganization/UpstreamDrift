"""Tests for the movement-variant generator (SPN-13b, issue #12767).

The synthetic swing below is invented (smooth ISB angle curves with a
right-handed backswing peak at ``top``); it carries no measured data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.spine.coupling_data import coupling_law
from src.shared.python.spine.distribution import distribute_series_detailed
from src.shared.python.spine.golf_terms import evaluate, registered_terms
from src.shared.python.spine.movement_variants import (
    DEFAULT_VARIANTS_PATH,
    JCS_TERMS,
    MovementVariant,
    RangeCheck,
    VariantResult,
    apply_variant,
    check_undeclared_terms,
    check_coordinate_ranges,
    load_movement_variants,
    movement_variant,
    sample_variant_parameters,
)

pytestmark = pytest.mark.unit

N = 201
TIME = np.linspace(0.0, 2.0, N)
EVENTS = {"address": 20, "top": 100, "impact": 150}
EXPECTED_IDS = {
    "x_factor_top_scale",
    "downswing_crunch_offset",
    "downswing_crunch_scale",
    "impact_extension_offset",
    "x_factor_peak_shift",
}
TARGET_TOL_DEG = 1e-6


def _angles_deg(handedness: str = "right") -> np.ndarray:
    """Invented ISB (FE, LB, AR) curves in degrees, shape (N, 3)."""
    s = np.linspace(0.0, 1.0, N)
    sign = -1.0 if handedness == "right" else 1.0
    fe = -20.0 + 15.0 * np.sin(np.pi * s) + 5.0 * s
    lb = sign * (-8.0 * np.sin(2.0 * np.pi * s) + 3.0 * s)
    ar = sign * 45.0 * np.exp(-(((s - 0.5) / 0.18) ** 2)) + sign * 2.0 * s
    return np.column_stack([fe, lb, ar])


def _rotations(handedness: str = "right") -> np.ndarray:
    return Rotation.from_euler("ZXY", _angles_deg(handedness), degrees=True).as_matrix()


def _terms(rot: np.ndarray, hand: str) -> dict[str, np.ndarray]:
    out = {
        t: evaluate(t, {"pelvis_to_thorax_rotation": rot}, handedness=hand)
        for t in JCS_TERMS
    }
    out["crunch_factor_rate"] = evaluate(
        "crunch_factor_rate",
        {
            "trunk_lateral_flexion_jcs": out["trunk_lateral_flexion_jcs"],
            "x_factor_jcs_axial": out["x_factor_jcs_axial"],
            "time_s": TIME,
        },
        handedness=hand,
    )
    return out


def _apply(variant_id: str, hand: str = "right", **kw) -> VariantResult:
    return apply_variant(
        movement_variant(variant_id),
        _rotations(hand),
        TIME,
        EVENTS,
        handedness=hand,
        **kw,
    )


# ------------------------------------------------------------------ registry


def test_default_config_lists_the_five_variants() -> None:
    variants = load_movement_variants()
    assert set(variants) == EXPECTED_IDS
    assert DEFAULT_VARIANTS_PATH.name == "spine_movement_variants.json"
    for vid, var in variants.items():
        assert isinstance(var, MovementVariant)
        assert var.variant_id == vid
        assert var.version.strip()
        assert var.term_id in registered_terms()
        assert var.term_id in var.affected_terms
        assert var.non_trunk_policy in ("thorax_relative_hold", "trunk_only")
        lo, hi = var.parameter_bounds
        assert lo <= var.parameter_default <= hi
        assert var.undeclared_tolerance_deg > 0.0
        assert var.formula.strip()


def test_unknown_variant_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unknown movement variant"):
        movement_variant("no_such_variant")
    with pytest.raises(TypeError):
        movement_variant(3)  # type: ignore[arg-type]


def test_loader_rejects_bad_entries(tmp_path: Path) -> None:
    raw = json.loads(DEFAULT_VARIANTS_PATH.read_text(encoding="utf-8"))
    bad = dict(raw)
    entry = dict(raw["variants"][0])
    entry["term_id"] = "not_a_term"
    bad["variants"] = [entry]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="not_a_term"):
        load_movement_variants(path)
    entry = dict(raw["variants"][0])
    entry["operation"] = "add_angles"
    bad["variants"] = [entry]
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="operation"):
        load_movement_variants(path)


# ------------------------------------------------------------ target achieved


@pytest.mark.parametrize("hand", ["right", "left"])
@pytest.mark.parametrize("variant_id", sorted(EXPECTED_IDS))
def test_variant_hits_target_and_leaves_undeclared_terms(
    variant_id: str, hand: str
) -> None:
    variant = movement_variant(variant_id)
    base = _rotations(hand)
    result = _apply(variant_id, hand)
    assert result.target_error_deg <= TARGET_TOL_DEG
    before = _terms(base, hand)
    after = _terms(np.asarray(result.rotations), hand)
    # independently re-evaluated target
    idx = result.target_indices
    np.testing.assert_allclose(
        after[variant.term_id][idx], result.target_values_deg, atol=TARGET_TOL_DEG
    )
    check = check_undeclared_terms(
        variant, base, np.asarray(result.rotations), TIME, handedness=hand
    )
    assert check.ok, check.max_change
    for term, values in before.items():  # independent per-sample cross-check
        if term not in variant.affected_terms:
            np.testing.assert_allclose(
                after[term], values, atol=variant.undeclared_tolerance_deg, rtol=0
            )


@pytest.mark.parametrize("factor", [0.8, 1.2, 0.5])
def test_x_factor_scale_at_top(factor: float) -> None:
    base = _terms(_rotations(), "right")["x_factor_jcs_axial"]
    result = _apply("x_factor_top_scale", parameter=factor)
    after = _terms(np.asarray(result.rotations), "right")["x_factor_jcs_axial"]
    top = EVENTS["top"]
    assert abs(after[top] - factor * base[top]) <= TARGET_TOL_DEG
    assert base[top] > 30.0  # backswing positive for the right-handed swing


def test_crunch_offset_and_scale_at_impact() -> None:
    base = _terms(_rotations(), "right")["trunk_lateral_flexion_jcs"]
    imp = EVENTS["impact"]
    off = _apply("downswing_crunch_offset", parameter=-6.0)
    lat = _terms(np.asarray(off.rotations), "right")["trunk_lateral_flexion_jcs"]
    assert abs(lat[imp] - (base[imp] - 6.0)) <= TARGET_TOL_DEG
    removed = _apply("downswing_crunch_scale", parameter=0.0)
    lat0 = _terms(np.asarray(removed.rotations), "right")["trunk_lateral_flexion_jcs"]
    assert abs(lat0[imp]) <= TARGET_TOL_DEG


def test_impact_extension_offset() -> None:
    base = _terms(_rotations(), "right")["trunk_extension_jcs"]
    result = _apply("impact_extension_offset", parameter=7.5)
    ext = _terms(np.asarray(result.rotations), "right")["trunk_extension_jcs"]
    assert abs(ext[EVENTS["impact"]] - (base[EVENTS["impact"]] + 7.5)) <= 1e-6


@pytest.mark.parametrize("shift", [-8, 5, 12])
def test_peak_shift_moves_argmax_and_keeps_peak_value(shift: int) -> None:
    base = _terms(_rotations(), "right")["x_factor_jcs_axial"]
    lo, hi = EVENTS["address"], EVENTS["impact"]
    old_peak = lo + int(np.argmax(base[lo : hi + 1]))
    result = _apply("x_factor_peak_shift", parameter=shift)
    after = _terms(np.asarray(result.rotations), "right")["x_factor_jcs_axial"]
    new_peak = lo + int(np.argmax(after[lo : hi + 1]))
    assert new_peak == old_peak + shift
    assert abs(after[new_peak] - base[old_peak]) <= TARGET_TOL_DEG
    assert result.provenance["achieved_shift_s"] == pytest.approx(
        TIME[new_peak] - TIME[old_peak]
    )


def test_peak_shift_rejects_non_integer_and_out_of_window() -> None:
    with pytest.raises(TypeError, match="integer"):
        _apply("x_factor_peak_shift", parameter=2.5)
    variant = movement_variant("x_factor_peak_shift")
    events = {"address": 95, "top": 100, "impact": 104}
    with pytest.raises(ValueError, match="window"):
        apply_variant(
            variant, _rotations(), TIME, events, handedness="right", parameter=10
        )


# ---------------------------------------------------- rotation-space property


@pytest.mark.parametrize("variant_id", sorted(EXPECTED_IDS))
def test_variant_is_a_composition_with_proper_rotations(variant_id: str) -> None:
    base = _rotations()
    result = _apply(variant_id)
    rots = np.asarray(result.rotations)
    assert rots.shape == base.shape
    gram = np.einsum("nji,njk->nik", rots, rots)
    np.testing.assert_allclose(gram, np.broadcast_to(np.eye(3), gram.shape), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(rots), 1.0, atol=1e-12)
    # the correction is itself a proper rotation applied by composition
    corr = np.asarray(result.corrections)
    side = movement_variant(variant_id).composition_side
    recon = base @ corr if side == "right" else corr @ base
    np.testing.assert_allclose(recon, rots, atol=1e-12)


@pytest.mark.parametrize("variant_id", sorted(EXPECTED_IDS))
def test_frames_outside_window_are_bit_identical(variant_id: str) -> None:
    base = _rotations()
    result = _apply(variant_id)
    lo, hi = result.window
    rots = np.asarray(result.rotations)
    assert np.array_equal(rots[:lo], base[:lo])
    assert np.array_equal(rots[hi + 1 :], base[hi + 1 :])
    assert not np.array_equal(rots[lo : hi + 1], base[lo : hi + 1])


def test_result_arrays_are_read_only() -> None:
    result = _apply("x_factor_top_scale")
    with pytest.raises(ValueError):
        result.rotations[0, 0, 0] = 1.0  # type: ignore[index]
    with pytest.raises(ValueError):
        result.delta_deg[0] = 1.0  # type: ignore[index]


# ----------------------------------------------------------------- contracts


def test_parameter_outside_bounds_raises() -> None:
    lo, hi = movement_variant("x_factor_top_scale").parameter_bounds
    with pytest.raises(ValueError, match="bounds"):
        _apply("x_factor_top_scale", parameter=hi + 0.1)
    with pytest.raises(TypeError):
        _apply("x_factor_top_scale", parameter="big")  # type: ignore[arg-type]


def test_missing_event_names_the_event() -> None:
    with pytest.raises(ValueError, match="impact"):
        apply_variant(
            movement_variant("x_factor_top_scale"),
            _rotations(),
            TIME,
            {"address": 20, "top": 100},
            handedness="right",
        )


def test_bad_inputs_raise() -> None:
    var = movement_variant("x_factor_top_scale")
    with pytest.raises(TypeError, match="MovementVariant"):
        apply_variant("x", _rotations(), TIME, EVENTS, handedness="right")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="handedness"):
        apply_variant(var, _rotations(), TIME, EVENTS, handedness="both")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="time_s"):
        apply_variant(var, _rotations(), TIME[::-1], EVENTS, handedness="right")
    with pytest.raises(ValueError, match="rotation"):
        apply_variant(var, _rotations() * 2.0, TIME, EVENTS, handedness="right")
    with pytest.raises(ValueError, match="order"):
        apply_variant(
            var,
            _rotations(),
            TIME,
            {"address": 120, "top": 100, "impact": 150},
            handedness="right",
        )


def test_gimbal_lock_in_window_raises() -> None:
    angles = _angles_deg()
    angles[90:110, 1] = 90.0
    rot = Rotation.from_euler("ZXY", angles, degrees=True).as_matrix()
    with pytest.raises(ValueError, match="gimbal"):
        apply_variant(
            movement_variant("downswing_crunch_offset"),
            rot,
            TIME,
            {"address": 20, "top": 85, "impact": 150},
            handedness="right",
        )


def test_term_bounds_violation_is_reported_not_repaired() -> None:
    var = movement_variant("x_factor_top_scale")
    lo, hi = var.parameter_bounds
    result = _apply("x_factor_top_scale", parameter=hi)
    assert result.term_bounds_ok == bool(
        np.all(
            (result.term_values_deg >= var.term_bounds_deg[0])
            & (result.term_values_deg <= var.term_bounds_deg[1])
        )
    )
    huge = np.radians(_angles_deg())
    huge[:, 2] *= 1.7
    rot = Rotation.from_euler("ZXY", huge).as_matrix()
    res = apply_variant(var, rot, TIME, EVENTS, handedness="right", parameter=hi)
    assert res.term_bounds_ok is False
    assert res.target_error_deg <= TARGET_TOL_DEG


# --------------------------------------------------------------- provenance


def test_provenance_records_parameters_and_source() -> None:
    result = _apply("x_factor_top_scale", parameter=1.2)
    prov = result.provenance
    assert prov["variant_id"] == "x_factor_top_scale"
    assert prov["variant_version"] == movement_variant("x_factor_top_scale").version
    assert prov["parameter_name"] == "factor"
    assert prov["parameter_value"] == 1.2
    assert prov["term_id"] == "x_factor_jcs_axial"
    assert prov["handedness"] == "right"
    assert len(prov["input_sha256"]) == 64
    assert result.distribution_source.startswith("movement_variant:x_factor_top_scale@")
    assert "factor=1.2" in result.distribution_source


def test_default_parameter_is_recorded() -> None:
    var = movement_variant("downswing_crunch_offset")
    result = _apply("downswing_crunch_offset")
    assert result.provenance["parameter_value"] == var.parameter_default


def test_variant_is_deterministic() -> None:
    a = _apply("x_factor_peak_shift", parameter=4)
    b = _apply("x_factor_peak_shift", parameter=4)
    assert np.asarray(a.rotations).tobytes() == np.asarray(b.rotations).tobytes()
    assert a.provenance == b.provenance


def test_coupling_law_is_untouched_by_distribution_of_a_variant() -> None:
    law = coupling_law("uniform_reference")
    phi = np.array([0.1, -0.05, 0.2])
    before = {lv: a.copy() for lv, a in law.expand(phi).items()}
    result = _apply("x_factor_top_scale")
    sub = slice(95, 106)
    detailed = distribute_series_detailed(
        np.asarray(result.rotations)[sub], TIME[sub], law
    )
    assert detailed.series.distribution_source.startswith("coupling:uniform_reference")
    after = law.expand(phi)
    for lv, arr in before.items():
        np.testing.assert_array_equal(after[lv], arr)


# ------------------------------------------------------------------ sampling


def test_sample_parameters_is_seeded_and_bounded() -> None:
    var = movement_variant("x_factor_top_scale")
    a = sample_variant_parameters(var, seed=7, count=16)
    b = sample_variant_parameters(var, seed=7, count=16)
    c = sample_variant_parameters(var, seed=8, count=16)
    assert a == b
    assert a != c
    lo, hi = var.parameter_bounds
    assert all(lo <= v <= hi for v in a)
    shift = sample_variant_parameters(
        movement_variant("x_factor_peak_shift"), seed=1, count=10
    )
    assert all(isinstance(v, int) for v in shift)


def test_sample_parameters_contracts() -> None:
    var = movement_variant("x_factor_top_scale")
    with pytest.raises(TypeError, match="seed"):
        sample_variant_parameters(var, seed=1.5, count=2)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="count"):
        sample_variant_parameters(var, seed=1, count=0)


# ------------------------------------------------------------- range check


def test_check_coordinate_ranges_reports_violations() -> None:
    coords = {
        "a": np.array([0.0, 0.5, 1.2]),
        "b": np.array([-0.2, -0.1, 0.0]),
    }
    ranges = {"a": (-1.0, 1.0), "b": (-1.0, 1.0)}
    check = check_coordinate_ranges(coords, ranges)
    assert isinstance(check, RangeCheck)
    assert check.ok is False
    assert len(check.violations) == 1
    v = check.violations[0]
    assert (v.coordinate, v.frame) == ("a", 2)
    assert v.value == pytest.approx(1.2)
    assert (v.lower, v.upper) == (-1.0, 1.0)
    ok = check_coordinate_ranges({"b": coords["b"]}, ranges)
    assert ok.ok is True and ok.violations == ()


def test_check_coordinate_ranges_contracts() -> None:
    with pytest.raises(ValueError, match="'c'"):
        check_coordinate_ranges({"c": np.zeros(2)}, {"a": (-1.0, 1.0)})
    with pytest.raises(ValueError, match="lower"):
        check_coordinate_ranges({"a": np.zeros(2)}, {"a": (1.0, -1.0)})
    with pytest.raises(TypeError):
        check_coordinate_ranges([1, 2], {"a": (-1.0, 1.0)})  # type: ignore[arg-type]


# ------------------------------------------------ undeclared-term check (#12825)


def _leak(rotations: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate every frame about the pelvis Z axis: leaks into trunk extension."""
    spin = Rotation.from_euler("z", degrees, degrees=True).as_matrix()
    return spin @ rotations


@pytest.mark.parametrize("hand", ["right", "left"])
@pytest.mark.parametrize("variant_id", sorted(EXPECTED_IDS))
def test_check_undeclared_terms_accepts_every_shipped_variant(
    variant_id: str, hand: str
) -> None:
    variant = movement_variant(variant_id)
    result = _apply(variant_id, hand)
    check = check_undeclared_terms(
        variant, _rotations(hand), result.rotations, TIME, handedness=hand
    )
    assert check.ok is True
    assert check.violations == ()
    assert check.tolerance == variant.undeclared_tolerance_deg
    assert set(check.max_change) == (set(JCS_TERMS) | {"crunch_factor_rate"}) - set(
        variant.affected_terms
    )
    assert all(v <= check.tolerance for v in check.max_change.values())


def test_check_undeclared_terms_flags_a_leaky_variant() -> None:
    variant = movement_variant("x_factor_top_scale")
    result = _apply("x_factor_top_scale")
    leaky = _leak(np.asarray(result.rotations), 2.0)
    check = check_undeclared_terms(
        variant, _rotations(), leaky, TIME, handedness="right"
    )
    assert check.ok is False
    assert "trunk_extension_jcs" in check.violations
    assert check.max_change["trunk_extension_jcs"] > 0.5  # degrees
    assert "x_factor_jcs_axial" not in check.max_change  # declared, not judged


def test_check_undeclared_terms_treats_lost_samples_as_a_change() -> None:
    variant = movement_variant("impact_extension_offset")
    base = _rotations()
    gimbal = base.copy()
    gimbal[5] = Rotation.from_euler("ZXY", [0.0, 90.0, 0.0], degrees=True).as_matrix()
    check = check_undeclared_terms(variant, base, gimbal, TIME, handedness="right")
    assert check.ok is False
    assert np.isinf(check.max_change["x_factor_jcs_axial"])  # AR lost at gimbal
    assert "trunk_lateral_flexion_jcs" in check.violations


def test_check_undeclared_terms_validates_arguments() -> None:
    variant = movement_variant("x_factor_top_scale")
    rots = _rotations()
    with pytest.raises(TypeError):
        check_undeclared_terms("x", rots, rots, TIME, handedness="right")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="same shape"):
        check_undeclared_terms(variant, rots, rots[:-1], TIME, handedness="right")
