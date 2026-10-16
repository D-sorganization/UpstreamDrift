"""Tests for spinal load series, sign conventions and the axial bridge (SPN-5)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.body_part_viz.axial_loads import (
    AxialLoadProvider,
    read_axial_load_frame,
)
from src.shared.python.core.constants import GRAVITY_FLOAT
from src.shared.python.spine.levels import SpinalLevel
from src.shared.python.spine.loads import (
    REQUIRED_PROVENANCE_KEYS,
    LoadObservable,
    SpinalAxialLoadProvider,
    SpinalLoadSeries,
    from_wrench_in_frame,
)

pytestmark = pytest.mark.unit

LEVELS = (SpinalLevel.L5_S1, SpinalLevel.L4_L5)
EYE = np.eye(3)


def _prov(**over: str) -> dict[str, str]:
    base = {key: f"value-{key}" for key in REQUIRED_PROVENANCE_KEYS}
    base.update(over)
    return base


def _series(force, moment=None, time_s=(0.0, 0.1), levels=LEVELS) -> SpinalLoadSeries:
    force = np.asarray(force, dtype=float)
    moment = np.zeros_like(force) if moment is None else np.asarray(moment, float)
    return SpinalLoadSeries(
        time_s=np.asarray(time_s, dtype=float),
        levels=levels,
        observable=LoadObservable.JOINT_REACTION,
        force_n=force,
        moment_nm=moment,
        provenance=_prov(),
    )


def _flat(n=2, levels=2):
    return np.zeros((n, levels, 3))


def test_observables_are_three_distinct():
    assert {o.name for o in LoadObservable} == {
        "GENERALIZED_EFFORT",
        "JOINT_REACTION",
        "PASSIVE_ELEMENT",
    }


def test_gravity_only_superior_body_gives_positive_compression():
    # A superior body of mass m at rest: inferior body pushes it up (+Y).
    mass = 30.0
    force = _flat()
    force[:, :, 1] = mass * GRAVITY_FLOAT
    s = _series(force)
    assert np.all(s.compression_n() > 0)
    assert s.compression_n() == pytest.approx(mass * GRAVITY_FLOAT)


def test_anterior_force_is_positive_ap_shear_and_right_is_ml_shear():
    force = _flat()
    force[:, :, 0] = 12.0
    force[:, :, 2] = -5.0
    s = _series(force)
    assert np.all(s.ap_shear_n() == 12.0)
    assert np.all(s.ml_shear_n() == -5.0)
    assert np.all(s.compression_n() == 0.0)


def test_peak_returns_value_and_time_per_level():
    force = _flat(n=3)
    force[:, 0, 1] = [10.0, 40.0, 20.0]
    force[:, 1, 1] = [5.0, 6.0, 7.0]
    s = _series(force, time_s=(0.0, 0.5, 1.0))
    peaks = s.peak("compression")
    assert peaks[SpinalLevel.L5_S1] == (40.0, 0.5)
    assert peaks[SpinalLevel.L4_L5] == (7.0, 1.0)
    with pytest.raises(ValueError, match="quantity"):
        s.peak("bogus")


def test_in_body_weights_divides_by_mg():
    force = _flat()
    force[:, :, 1] = 800.0
    s = _series(force).in_body_weights(80.0)
    assert s.compression_n() == pytest.approx(800.0 / (80.0 * GRAVITY_FLOAT))
    assert s.provenance["normalised_by_body_mass_kg"] == "80.0"
    with pytest.raises(ValueError, match="already normalised"):
        s.in_body_weights(80.0)


@pytest.mark.parametrize("mass", [0.0, -1.0, float("nan"), float("inf")])
def test_in_body_weights_rejects_bad_mass(mass):
    with pytest.raises(ValueError, match="body_mass_kg"):
        _series(_flat()).in_body_weights(mass)


def test_in_body_weights_rejects_non_number():
    with pytest.raises(TypeError, match="body_mass_kg"):
        _series(_flat()).in_body_weights("80")  # type: ignore[arg-type]


def test_from_wrench_moves_moment_by_r_cross_f():
    # F = (0, 100, 0) N at r = (0.05, 0, 0) m from the joint centre.
    # M' = M + r x F = (0,0,0) + (0.05,0,0)x(0,100,0) = (0,0,5).
    force = _flat()
    force[:, :, 1] = 100.0
    s = from_wrench_in_frame(
        time_s=[0.0, 0.1],
        levels=LEVELS,
        observable=LoadObservable.JOINT_REACTION,
        force=force,
        moment=_flat(),
        rotation_frame_to_isb=EYE,
        point_offset_m=(0.05, 0.0, 0.0),
        provenance=_prov(),
    )
    assert s.moment_nm[:, :, 2] == pytest.approx(5.0)
    assert s.moment_nm[:, :, :2] == pytest.approx(0.0)
    assert s.force_n[:, :, 1] == pytest.approx(100.0)


def test_from_wrench_rotates_into_inferior_isb_frame():
    # Source x axis maps to ISB y (cranial): R @ (1,0,0) = (0,1,0).
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    force = _flat()
    force[:, :, 0] = 50.0
    s = from_wrench_in_frame(
        time_s=[0.0, 0.1],
        levels=LEVELS,
        observable=LoadObservable.GENERALIZED_EFFORT,
        force=force,
        moment=_flat(),
        rotation_frame_to_isb=rot,
        point_offset_m=(0.0, 0.0, 0.0),
        provenance=_prov(),
    )
    assert s.compression_n() == pytest.approx(50.0)
    assert s.ap_shear_n() == pytest.approx(0.0)


def test_from_wrench_rejects_non_rotation():
    with pytest.raises(ValueError, match="rotation"):
        from_wrench_in_frame(
            time_s=[0.0, 0.1],
            levels=LEVELS,
            observable=LoadObservable.JOINT_REACTION,
            force=_flat(),
            moment=_flat(),
            rotation_frame_to_isb=2.0 * EYE,
            point_offset_m=(0.0, 0.0, 0.0),
            provenance=_prov(),
        )


def test_bridge_gives_tension_equal_minus_compression():
    force = _flat()
    force[:, 0, 1] = 600.0
    force[:, 1, 1] = 450.0
    provider = SpinalAxialLoadProvider(_series(force), time_s=0.1)
    assert isinstance(provider, AxialLoadProvider)
    frame = read_axial_load_frame(provider, 0.1)
    assert frame is not None
    assert frame["sign_convention"] == "tension-positive"
    assert frame["values_n"] == {"L5_S1": -600.0, "L4_L5": -450.0}
    assert "spine.loads" in frame["source"]


def test_bridge_stale_time_reads_none_and_off_grid_time_raises():
    provider = SpinalAxialLoadProvider(_series(_flat()), time_s=0.0)
    assert read_axial_load_frame(provider, 0.1) is None
    with pytest.raises(ValueError, match="sample"):
        SpinalAxialLoadProvider(_series(_flat()), time_s=0.05)


def test_bridge_rejects_body_weight_series():
    s = _series(_flat()).in_body_weights(70.0)
    with pytest.raises(ValueError, match="newtons"):
        SpinalAxialLoadProvider(s, time_s=0.0)


@pytest.mark.parametrize("key", REQUIRED_PROVENANCE_KEYS)
def test_missing_provenance_key_rejected(key):
    prov = _prov()
    del prov[key]
    with pytest.raises(ValueError, match=key):
        SpinalLoadSeries(
            time_s=np.array([0.0, 0.1]),
            levels=LEVELS,
            observable=LoadObservable.JOINT_REACTION,
            force_n=_flat(),
            moment_nm=_flat(),
            provenance=prov,
        )


def test_empty_provenance_value_rejected():
    with pytest.raises(ValueError, match="distribution_source"):
        SpinalLoadSeries(
            time_s=np.array([0.0, 0.1]),
            levels=LEVELS,
            observable=LoadObservable.JOINT_REACTION,
            force_n=_flat(),
            moment_nm=_flat(),
            provenance=_prov(distribution_source="  "),
        )


def test_provenance_is_immutable_and_arrays_read_only():
    s = _series(_flat())
    with pytest.raises(TypeError):
        s.provenance["solver"] = "x"  # type: ignore[index]
    assert not s.force_n.flags.writeable
    assert not s.moment_nm.flags.writeable
    assert not s.time_s.flags.writeable


def _kw(**over):
    base = {
        "time_s": np.array([0.0, 0.1]),
        "levels": LEVELS,
        "observable": LoadObservable.JOINT_REACTION,
        "force_n": _flat(),
        "moment_nm": _flat(),
        "provenance": _prov(),
    }
    base.update(over)
    return base


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError, match="force_n"):
        SpinalLoadSeries(**_kw(force_n=np.zeros((2, 3, 3))))
    with pytest.raises(ValueError, match="moment_nm"):
        SpinalLoadSeries(**_kw(moment_nm=np.zeros((2, 2))))


def test_nan_rejected():
    bad = _flat()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        SpinalLoadSeries(**_kw(force_n=bad))
    with pytest.raises(ValueError, match="finite"):
        SpinalLoadSeries(**_kw(moment_nm=bad))


@pytest.mark.parametrize(
    "times", [[0.0, 0.0], [0.1, 0.0], [0.0, np.nan], [], [[0.0, 0.1]]]
)
def test_bad_time_rejected(times):
    with pytest.raises(ValueError, match="time_s"):
        SpinalLoadSeries(**_kw(time_s=np.asarray(times, dtype=float)))


def test_bad_types_rejected():
    with pytest.raises(TypeError, match="observable"):
        SpinalLoadSeries(**_kw(observable="joint_reaction"))
    with pytest.raises(TypeError, match="levels"):
        SpinalLoadSeries(**_kw(levels=("L5_S1", "L4_L5")))
    with pytest.raises(ValueError, match="levels"):
        SpinalLoadSeries(**_kw(levels=(SpinalLevel.L5_S1, SpinalLevel.L5_S1)))
