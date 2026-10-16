"""SPN-4c: swing-metrics and API X-factor delegate to the golf-term registry.

Golden values were captured from the legacy inline formula (invented inputs).
Characterisation tests pin behaviour; delegation tests assert the registry
term ``x_factor_position_proxy`` is the single source of the formula.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.analysis.swing_metrics import SwingMetricsMixin
from src.shared.python.spine import golf_terms

pytestmark = pytest.mark.unit

_ATOL = 1e-12
_T = np.linspace(0.0, 1.0, 9)
_POS = np.column_stack(
    [np.sin(2 * _T), np.cos(3 * _T), 0.5 * _T**2, np.sin(5 * _T + 1)]
)
_XF_3_1 = [
    -9.08304350087284, 3.897471984142186, 2.657626230758382, -9.610251088359705,
    -24.151341580716956, -30.530457663477264, -21.263561922690137,
    4.647747755227073, 40.71306302748131,
]  # fmt: skip
_XF_NEG1_0 = [
    48.21273601220948, 43.03642882436196, 17.11125038112227, -23.96068055251228,
    -68.3111344828013, -102.06527690279961, -114.40751212401727,
    -101.54958585515026, -68.10823365656107,
]  # fmt: skip
_STRETCH = [
    103.84412388012021, 46.96267892652489, -54.030892290007564,
    -107.23587124590135, -83.68082630047024, 11.551118632107276,
    140.71282167481735, 247.9064998006858, 288.5225221780339,
]  # fmt: skip


def _ctx(positions: np.ndarray = _POS) -> SwingMetricsMixin:
    class _Ctx(SwingMetricsMixin):
        pass

    ctx = _Ctx()
    ctx.joint_positions = positions
    ctx.times = _T
    ctx.dt = float(_T[1] - _T[0])
    ctx.club_head_speed = None
    return ctx


def test_x_factor_golden() -> None:
    out = _ctx().compute_x_factor(3, 1)
    np.testing.assert_allclose(out, _XF_3_1, rtol=0, atol=_ATOL)


def test_x_factor_negative_index_golden() -> None:
    out = _ctx().compute_x_factor(-1, 0)
    np.testing.assert_allclose(out, _XF_NEG1_0, rtol=0, atol=_ATOL)


def test_x_factor_nan_passes_through() -> None:
    pos = _POS.copy()
    pos[2, 3] = np.nan
    out = _ctx(pos).compute_x_factor(3, 1)
    assert np.isnan(out[2])
    np.testing.assert_allclose(np.delete(out, 2), np.delete(_XF_3_1, 2), atol=_ATOL)


def test_x_factor_out_of_range_returns_none() -> None:
    assert _ctx().compute_x_factor(0, 9) is None


def test_x_factor_stretch_golden() -> None:
    result = _ctx().compute_x_factor_stretch(3, 1)
    assert result is not None
    np.testing.assert_allclose(result[0], _STRETCH, rtol=0, atol=1e-9)
    assert result[1] == pytest.approx(288.5225221780339, abs=1e-9)


def _spy(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    real = golf_terms.evaluate

    def spy(term_id, inputs, *, handedness):  # noqa: ANN001, ANN202
        calls.append(term_id)
        return real(term_id, inputs, handedness=handedness)

    monkeypatch.setattr(golf_terms, "evaluate", spy)
    return calls


def test_swing_metrics_x_factor_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _spy(monkeypatch)
    _ctx().compute_x_factor(3, 1)
    assert calls == ["x_factor_position_proxy"]


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_api_x_factor_delegates(
    monkeypatch: pytest.MonkeyPatch, anyio_backend: str
) -> None:
    from unittest.mock import MagicMock

    from src.api.models.requests import AnalysisRequest
    from src.api.services.analysis_service import AnalysisService

    calls = _spy(monkeypatch)
    service = AnalysisService(MagicMock())
    request = AnalysisRequest(
        analysis_type="swing_sequence",
        data_source="simulation",
        parameters={},
        data={
            "joint_positions": [[0.0, 0.0], [np.pi / 2.0, np.pi / 6.0]],
            "shoulder_joint_idx": 0,
            "hip_joint_idx": 1,
            "times": [0.0, 0.1],
        },
    )

    class _Engine:
        pass

    result = await service._analyze_swing_sequence(request, _Engine())
    assert result["x_factor"]["values"] == pytest.approx([0.0, 60.0], abs=_ATOL)
    assert result["x_factor"]["peak_stretch_rate"] == pytest.approx(600.0, abs=1e-9)
    assert set(calls) == {"x_factor_position_proxy"}
