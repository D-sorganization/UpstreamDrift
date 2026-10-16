"""SPN-4b: SpinalLoadAnalyzer X-factor / crunch delegate to the golf-term registry.

Characterisation tests pin the pre-migration outputs (golden values captured
from the legacy inline formulas); the delegation tests assert the analyzer
routes those two quantities through ``spine.golf_terms.evaluate``.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.injury.spinal_load_analysis import SpinalLoadAnalyzer
from src.shared.python.spine import golf_terms
from tests.unit.injury.test_unit_spinal_load_analysis import (
    _make_analyzer,
    _make_inputs,
)

pytestmark = pytest.mark.unit

_ATOL = 1e-12
_TIME = np.linspace(0.0, 1.0, 11)

# Golden outputs captured from the legacy inline implementation (invented inputs).
_XF_ANGLE = [
    0.0, -8.044797933386604, -16.35357526604965, -25.16673090372517,
    -34.67960914032774, -45.02505013395945, -56.261523691218045,
    -68.36790633351127, -81.24536819448849, -94.7262011976617,
    -108.58879991940134,
]  # fmt: skip
_CRUNCH = [
    2.9999999999999996, 1.05, 0.0, 0.24000000000000005, 0.9, 3.2999999999999994,
    4.32, 1.65, 1.2600000000000002, 6.120000000000001, 3.6,
]  # fmt: skip


def _synthetic_angles() -> tuple[np.ndarray, np.ndarray]:
    ramp = np.linspace(0, 1, 11)
    zeros = np.zeros(11)
    pelvis = np.stack([zeros, zeros, np.deg2rad(np.linspace(0, 40, 11))], 1)
    wobble = 10 * np.sin(np.linspace(0, 3, 11))
    thorax = np.stack([zeros, zeros, np.deg2rad(-70 * ramp + wobble)], 1)
    return pelvis, thorax


def _synthetic_crunch() -> tuple[np.ndarray, np.ndarray]:
    lat = np.deg2rad(np.array([-10, -5, 0, 8, 15, 22, 18, 5, -3, -12, -6.0]))
    return lat, np.deg2rad(np.linspace(-30, 60, 11))


def test_x_factor_golden_synthetic() -> None:
    pelvis, thorax = _synthetic_angles()
    xf = _make_analyzer()._compute_x_factor(pelvis, thorax, _TIME)
    np.testing.assert_allclose(xf.x_factor_angle, _XF_ANGLE, rtol=0, atol=_ATOL)
    assert xf.x_factor_stretch == pytest.approx(108.58879991940134, abs=_ATOL)
    assert xf.x_factor_stretch_time == pytest.approx(1.0, abs=_ATOL)
    assert xf.separation_rate == pytest.approx(138.62598721739644, abs=1e-9)
    assert xf.transition_duration == pytest.approx(0.0, abs=_ATOL)


def test_crunch_golden_synthetic() -> None:
    lat, rot = _synthetic_crunch()
    cf = _make_analyzer()._compute_crunch_factor(lat, rot, _TIME)
    np.testing.assert_allclose(cf.crunch_factor, _CRUNCH, rtol=0, atol=_ATOL)
    assert cf.peak_crunch == pytest.approx(6.120000000000001, abs=_ATOL)
    assert cf.peak_crunch_time == pytest.approx(0.9, abs=_ATOL)
    assert cf.asymmetry_ratio == pytest.approx(3.5732483962635397, abs=1e-9)


def test_analyze_golden_fixture() -> None:
    n = 100
    inputs = _make_inputs(n)
    inputs["pelvis_angles"] = np.deg2rad(np.linspace(0, 45, n))
    inputs["thorax_angles"] = np.deg2rad(60 * np.sin(np.linspace(0, np.pi, n)))
    inputs["joint_angles"]["lumbar_lateral"] = inputs["joint_angles"][
        "lumbar_lateral_bend"
    ]
    result = _make_analyzer().analyze(**inputs)
    xf, cf = result.x_factor, result.crunch_factor
    assert xf is not None and cf is not None
    assert xf.x_factor_stretch == pytest.approx(44.99999999999999, abs=_ATOL)
    assert xf.x_factor_stretch_time == pytest.approx(1.5, abs=_ATOL)
    assert xf.separation_rate == pytest.approx(155.64261665234733, abs=1e-9)
    assert float(xf.x_factor_angle.sum()) == pytest.approx(1531.2041099461367, abs=1e-8)
    assert cf.peak_crunch == pytest.approx(6563.959946162046, abs=1e-8)
    assert cf.peak_crunch_time == pytest.approx(0.3787878787878788, abs=_ATOL)
    assert cf.asymmetry_ratio == pytest.approx(0.9999999999998769, abs=1e-9)
    assert float(cf.crunch_factor.sum()) == pytest.approx(324997.8286511626, abs=1e-6)


def _spy(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    real = golf_terms.evaluate

    def spy(term_id, inputs, *, handedness):  # noqa: ANN001, ANN202
        calls.append(term_id)
        return real(term_id, inputs, handedness=handedness)

    monkeypatch.setattr(golf_terms, "evaluate", spy)
    return calls


def test_x_factor_delegates_to_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _spy(monkeypatch)
    pelvis, thorax = _synthetic_angles()
    SpinalLoadAnalyzer(body_weight=80.0)._compute_x_factor(pelvis, thorax, _TIME)
    assert calls == ["x_factor_euler_yaw_legacy_v1"]


def test_crunch_delegates_to_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _spy(monkeypatch)
    lat, rot = _synthetic_crunch()
    SpinalLoadAnalyzer(body_weight=80.0)._compute_crunch_factor(lat, rot, _TIME)
    assert calls == ["crunch_factor_legacy_v1"]
