"""SPN-12: heuristic / model-based spinal load provider split.

The golden tests pin the pre-split analyzer outputs (invented inputs, captured
before the provider seam existed) and must pass both before and after.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.injury.spinal_load_analysis import (
    SpinalLoadAnalyzer,
    SpinalRiskLevel,
)
from src.shared.python.injury.spinal_load_providers import (
    HEURISTIC_PROVIDER_ID,
    HeuristicSpinalLoadProvider,
    ModelSpinalLoadProvider,
    SpinalLoadProvider,
)
from src.shared.python.spine.levels import SpinalLevel
from src.shared.python.spine.loads import (
    REQUIRED_PROVENANCE_KEYS,
    LoadObservable,
    SpinalLoadSeries,
)
from tests.unit.injury.test_unit_spinal_load_analysis import (
    _make_analyzer,
    _make_inputs,
)

pytestmark = pytest.mark.unit

_ATOL = 1e-12
_GOLDEN_PEAKS = (
    1.330174098124608,
    0.5359083529079683,
    25.682402295655628,
    39.18568348616488,
)
_GOLDEN_IMPULSES = (1628.6385462741769, 38408.395397085405)
_GOLDEN_SUMS = {
    "L3-L4": (4596.794984861227, -1462.9101311569973, 120891.99862689179),
    "L4-L5": (4609.569645448065, -1706.72848634983, 105780.4987985303),
    "L5-S1": (4622.344306034902, -1950.5468415426635, 80594.66575126119),
}


def _model_series(n: int = 11, scale: float = 1.0) -> SpinalLoadSeries:
    levels = (SpinalLevel.L5_S1, SpinalLevel.L4_L5)
    time = np.linspace(0.0, 1.0, n)
    force = np.zeros((n, 2, 3))
    force[:, :, 1] = scale * np.linspace(400.0, 900.0, n)[:, None]
    force[:, :, 0] = 50.0
    force[:, :, 2] = -20.0
    moment = np.zeros((n, 2, 3))
    moment[:, :, 1] = 7.0
    prov = dict.fromkeys(REQUIRED_PROVENANCE_KEYS, "synthetic")
    return SpinalLoadSeries(
        time_s=time,
        levels=levels,
        observable=LoadObservable.JOINT_REACTION,
        force_n=force,
        moment_nm=moment,
        provenance=prov,
    )


def test_default_analyzer_matches_golden_outputs() -> None:
    result = _make_analyzer().analyze(**_make_inputs(11))
    got = (
        result.peak_compression_bw,
        result.peak_ap_shear_bw,
        result.peak_lateral_shear_bw,
        result.peak_torsion,
    )
    np.testing.assert_allclose(got, _GOLDEN_PEAKS, rtol=0, atol=_ATOL)
    np.testing.assert_allclose(
        (result.cumulative_compression_impulse, result.cumulative_shear_impulse),
        _GOLDEN_IMPULSES,
        rtol=1e-12,
        atol=0,
    )
    assert result.compression_risk is SpinalRiskLevel.SAFE
    assert result.shear_risk is SpinalRiskLevel.CRITICAL
    assert result.overall_risk is SpinalRiskLevel.CRITICAL
    for name, (c, a, lat) in _GOLDEN_SUMS.items():
        seg = result.segments[name]
        np.testing.assert_allclose(
            (seg.compression.sum(), seg.ap_shear.sum(), seg.lateral_shear.sum()),
            (c, a, lat),
            rtol=1e-12,
            atol=0,
        )


def test_default_provider_is_heuristic_and_labelled() -> None:
    analyzer = _make_analyzer()
    assert analyzer.provider.provider_id == HEURISTIC_PROVIDER_ID == "heuristic_v1"
    assert analyzer.provider.is_model_based is False
    result = analyzer.analyze(**_make_inputs(11))
    assert result.provider_id == "heuristic_v1"
    assert result.is_model_based is False
    assert result.risk_assessed is True


def test_explicit_heuristic_provider_equals_default() -> None:
    default = _make_analyzer().analyze(**_make_inputs(11))
    analyzer = _make_analyzer()
    explicit = SpinalLoadAnalyzer(
        body_weight=80.0,
        height=1.75,
        provider=HeuristicSpinalLoadProvider(analyzer),
    ).analyze(**_make_inputs(11))
    for name, seg in default.segments.items():
        other = explicit.segments[name]
        np.testing.assert_allclose(other.compression, seg.compression, atol=1e-12)
        np.testing.assert_allclose(other.ap_shear, seg.ap_shear, atol=1e-12)
        np.testing.assert_allclose(other.torsion, seg.torsion, atol=1e-12)
    assert explicit.peak_compression_bw == pytest.approx(
        default.peak_compression_bw, abs=1e-12
    )


def test_heuristic_series_is_labelled_heuristic_not_model() -> None:
    analyzer = _make_analyzer()
    provider = HeuristicSpinalLoadProvider(analyzer)
    inputs = _make_inputs(11)
    series = provider.loads(
        time_s=inputs["time"],
        joint_angles=inputs["joint_angles"],
        joint_velocities=inputs["joint_velocities"],
        joint_torques=inputs["joint_torques"],
    )
    assert isinstance(series, SpinalLoadSeries)
    assert provider.is_model_based is False
    assert series.provenance["solver"] == "heuristic_v1"
    assert "heuristic" in series.provenance["distribution_source"]
    assert series.provenance["provider_label"] == "heuristic"
    assert series.provenance["model_id"] == "n/a (heuristic)"
    assert SpinalLevel.L4_L5 in series.levels
    k = series.levels.index(SpinalLevel.L4_L5)
    ref = analyzer.analyze(**inputs).segments["L4-L5"]
    np.testing.assert_allclose(
        series.compression_n()[:, k], ref.compression, atol=1e-12
    )
    np.testing.assert_allclose(series.ap_shear_n()[:, k], ref.ap_shear, atol=1e-12)
    np.testing.assert_allclose(series.ml_shear_n()[:, k], ref.lateral_shear, atol=1e-12)


def test_model_provider_passes_series_through_with_provenance() -> None:
    series = _model_series()
    provider = ModelSpinalLoadProvider(series)
    assert provider.is_model_based is True
    assert provider.provider_id != HEURISTIC_PROVIDER_ID
    out = provider.loads(time_s=series.time_s)
    assert out is series
    assert out.provenance["solver"] == "synthetic"
    assert isinstance(provider, SpinalLoadProvider)


def test_model_provider_rejects_mismatched_time_and_levels() -> None:
    series = _model_series()
    provider = ModelSpinalLoadProvider(series)
    with pytest.raises(ValueError, match="time"):
        provider.loads(time_s=series.time_s[:-1])
    with pytest.raises(ValueError, match="time"):
        provider.loads(time_s=series.time_s + 0.01)
    with pytest.raises(ValueError, match="level"):
        provider.loads(time_s=series.time_s, levels=(SpinalLevel.T1_T2,))
    sub = provider.loads(time_s=series.time_s, levels=(SpinalLevel.L4_L5,))
    assert sub.levels == (SpinalLevel.L4_L5,)


def test_model_provider_rejects_non_series() -> None:
    with pytest.raises(TypeError, match="SpinalLoadSeries"):
        ModelSpinalLoadProvider([1, 2, 3])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="provider_id"):
        ModelSpinalLoadProvider(_model_series(), provider_id=" ")


def test_analyzer_with_model_provider_reports_model_and_skips_heuristic_risk() -> None:
    series = _model_series()
    analyzer = SpinalLoadAnalyzer(
        body_weight=80.0, provider=ModelSpinalLoadProvider(series)
    )
    inputs = _make_inputs(11)
    inputs["time"] = series.time_s
    inputs = {
        "joint_angles": {k: v[:11] for k, v in inputs["joint_angles"].items()},
        "joint_velocities": {k: v[:11] for k, v in inputs["joint_velocities"].items()},
        "joint_torques": {k: v[:11] for k, v in inputs["joint_torques"].items()},
        "time": series.time_s,
    }
    result = analyzer.analyze(**inputs)
    assert result.provider_id != "heuristic_v1"
    assert result.is_model_based is True
    assert result.risk_assessed is False
    # Heuristic bands do not apply to model loads; never report "safe".
    for risk in (
        result.compression_risk,
        result.shear_risk,
        result.x_factor_risk,
        result.overall_risk,
    ):
        assert risk is SpinalRiskLevel.NOT_ASSESSED
    assert set(result.segments) == {"L5-S1", "L4-L5"}
    np.testing.assert_allclose(
        result.segments["L4-L5"].compression, series.compression_n()[:, 1], atol=1e-12
    )
    expected = 900.0 / analyzer.body_weight_N
    assert result.peak_compression_bw == pytest.approx(expected, abs=1e-12)
    assert result.compression_risk is SpinalRiskLevel.SAFE


def test_analyzer_rejects_bad_provider() -> None:
    with pytest.raises(TypeError, match="provider"):
        SpinalLoadAnalyzer(body_weight=80.0, provider=object())  # type: ignore[arg-type]


def test_swing_objective_spinal_compression_is_labelled_heuristic() -> None:
    from src.shared.python.optimization import _swing_objectives

    assert _swing_objectives.SPINAL_OBJECTIVE_PROVIDER_ID == "heuristic_v1"
    doc = _swing_objectives.compute_metrics.__doc__ or ""
    assert "heuristic" in doc.lower()
