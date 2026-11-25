"""Fixture tests for isolated hand-load plots (#15494, #15725). Agg only."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex
from matplotlib.figure import Figure

from hand_load_plots import (
    HandLoadPlotInput,
    StrategyLoads,
    adapt_simulation_output,
    opposing_force_cancellation,
    plot_cancellation,
    plot_invariant_dashboard,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

# Okabe–Ito oracle (https://jfly.uni-koeln.de/color/), independent of the package.
OKABE_ITO = {
    "#000000",
    "#e69f00",
    "#56b4e9",
    "#009e73",
    "#f0e442",
    "#0072b2",
    "#d55e00",
    "#cc79a7",
}
_SHA = "a1b2c3d4e5f6"
_HASH = "0123456789abcdef"
_N = 5
_TIME = np.linspace(0.0, 1.0, _N)


def _vec(value: float) -> np.ndarray:
    return np.tile(np.array([value, 0.0, 0.0]), (_N, 1))


def _loads(
    strategy_id: str,
    left: float,
    right: float,
    alpha: float,
    *,
    evidence: str = "allocation-only",
    left_force: np.ndarray | None = None,
    alpha_n: np.ndarray | None = None,
    zvcf_moment: float | None = 0.8,
) -> StrategyLoads:
    moment = np.full(_N, np.nan if zvcf_moment is None else zvcf_moment)
    return StrategyLoads(
        strategy_id=strategy_id,
        evidence=evidence,
        force_left_n=_vec(left) if left_force is None else left_force,
        force_right_n=_vec(right),
        alpha_n=np.full(_N, alpha) if alpha_n is None else alpha_n,
        midhands_force_n={"total": _vec(10.0), "ztcf": _vec(6.0), "zvcf": _vec(4.0)},
        midhands_moment_nm={
            "total": np.full(_N, 2.0),
            "ztcf": np.full(_N, 1.2),
            "zvcf": moment,
        },
    )


def _record(**overrides: object) -> HandLoadPlotInput:
    fields: dict[str, object] = {
        "time_s": _TIME,
        "git_sha": _SHA,
        "data_hash": _HASH,
        "objective": "Minimum squared hand force versus baseline",
        "phases": (
            ("Address", 0.0),
            ("Top", 0.35),
            ("Impact", 0.6),
            ("Follow-Through", 1.0),
        ),
        "strategies": (
            _loads("baseline", 8.0, 6.0, 3.0),
            _loads("min_squared_force", 5.0, 5.0, 0.0),
        ),
    }
    fields.update(overrides)
    return HandLoadPlotInput(**fields)  # type: ignore[arg-type]


def _joined(fig: Figure) -> str:
    parts = [fig.get_suptitle(), *(text.get_text() for text in fig.texts)]
    for axis in fig.axes:
        parts.extend((axis.get_title(), axis.get_xlabel(), axis.get_ylabel()))
        parts.extend(text.get_text() for text in axis.texts)
        parts.extend(str(line.get_label()) for line in axis.lines)
    return "\n".join(parts)


def _y(axis, label: str) -> np.ndarray:
    found = [line for line in axis.lines if line.get_label() == label]
    assert len(found) == 1, label
    return np.asarray(found[0].get_ydata(), dtype=float)


def _chrome(fig: Figure, badge: str) -> None:
    assert fig.get_suptitle() == "Objective: Minimum squared hand force versus baseline"
    texts = {text.get_gid(): text.get_text() for text in fig.texts}
    assert texts["evidence-badge"] == badge
    assert _SHA in texts["provenance-footer"] and _HASH in texts["provenance-footer"]
    for axis in fig.axes:
        assert axis.get_xlabel() == "Time (s)"
        for line in axis.lines:
            if str(line.get_label()).startswith("_"):
                continue
            assert to_hex(line.get_color()).lower() in OKABE_ITO


def test_dashboard_invariants_match_while_hand_forces_shrink() -> None:
    fig = plot_invariant_dashboard(_record())
    try:
        assert [axis.get_title() for axis in fig.axes] == [
            "Invariant Midhands Force",
            "Invariant Midhands Moment",
            "Left-Hand Force",
            "Right-Hand Force",
        ]
        assert fig.axes[0].get_ylabel() == "Force (N)"
        assert fig.axes[1].get_ylabel() == "Moment (N·m)"
        _chrome(fig, "allocation-only")
        force, moment, left, right = fig.axes
        for component, level in (("Total", 10.0), ("ZTCF", 6.0), ("ZVCF", 4.0)):
            base = _y(force, f"{component} force baseline")
            np.testing.assert_allclose(
                base, _y(force, f"{component} force min squared force")
            )
            np.testing.assert_allclose(base, np.full(_N, level))
        for component, level in (("Total", 2.0), ("ZTCF", 1.2), ("ZVCF", 0.8)):
            base = _y(moment, f"{component} moment baseline")
            np.testing.assert_allclose(
                base, _y(moment, f"{component} moment min squared force")
            )
            np.testing.assert_allclose(base, np.full(_N, level))
        assert np.all(
            _y(left, "LH force min squared force") < _y(left, "LH force baseline")
        )
        assert np.all(
            _y(right, "RH force min squared force") < _y(right, "RH force baseline")
        )
    finally:
        plt.close(fig)


def test_missing_samples_stay_nan_and_blank_series_say_unavailable() -> None:
    left = _vec(8.0)
    left[2] = np.nan
    record = _record(
        strategies=(
            _loads(
                "baseline",
                8.0,
                6.0,
                3.0,
                left_force=left,
                alpha_n=np.full(_N, np.nan),
                zvcf_moment=None,
            ),
            _loads("min_squared_force", 5.0, 5.0, 0.0),
        )
    )
    fig = plot_invariant_dashboard(record)
    try:
        drawn = _y(fig.axes[2], "LH force baseline")
        assert np.isnan(drawn[2]) and drawn[2] != 0.0
        labels = [line.get_label() for line in fig.axes[1].lines]
        assert "ZVCF moment baseline" not in labels
        assert "UNAVAILABLE" in _joined(fig)
    finally:
        plt.close(fig)
    cancel = plot_cancellation(record)
    try:
        assert "baseline" not in [line.get_label() for line in cancel.axes[1].lines]
        assert "UNAVAILABLE" in _joined(cancel)
        np.testing.assert_allclose(
            _y(cancel.axes[1], "min squared force"), np.zeros(_N)
        )
    finally:
        plt.close(cancel)


def test_cancellation_curve_phases_badge_and_wording() -> None:
    record = _record()
    baseline = record.strategies[0]
    expected = opposing_force_cancellation(
        baseline.force_left_n,
        baseline.force_right_n,
        baseline.midhands_force_n["total"],
    )
    np.testing.assert_allclose(expected, np.full(_N, 4.0))
    fig = plot_cancellation(record)
    try:
        assert [axis.get_title() for axis in fig.axes] == [
            "Opposing-Force Cancellation",
            "Axial Squeeze",
        ]
        assert fig.axes[0].get_ylabel() == "Opposing-Force Cancellation (N)"
        assert fig.axes[1].get_ylabel() == "Axial Squeeze (N)"
        np.testing.assert_allclose(_y(fig.axes[0], "baseline"), expected)
        np.testing.assert_allclose(_y(fig.axes[0], "min squared force"), np.zeros(_N))
        np.testing.assert_allclose(_y(fig.axes[1], "baseline"), np.full(_N, 3.0))
        text = _joined(fig).lower()
        for name in ("address", "top", "impact", "follow-through"):
            assert name in text
        assert "opposing-force cancellation" in text
        assert "wasted" not in text and "optimal" not in text
        _chrome(fig, "allocation-only")
    finally:
        plt.close(fig)


def test_forward_replayed_badge_and_adapter_leaves_gaps() -> None:
    record = _record(
        strategies=(
            _loads("baseline", 8.0, 6.0, 3.0, evidence="forward-replayed"),
            _loads("min_squared_force", 5.0, 5.0, 0.0, evidence="forward-replayed"),
        )
    )
    fig = plot_cancellation(record)
    try:
        badge = next(text for text in fig.texts if text.get_gid() == "evidence-badge")
        assert badge.get_text() == "forward-replayed"
    finally:
        plt.close(fig)

    class _Source:
        time_s = _TIME
        git_sha = _SHA
        data_hash = _HASH
        objective = "Minimum squared hand force versus baseline"
        phases = (("Address", 0.0), ("Impact", 0.6), ("Follow-Through", 1.0))
        strategies = (
            {
                "strategy_id": "baseline",
                "evidence": "allocation-only",
                "force_left_n": _vec(8.0),
                "force_right_n": _vec(6.0),
                "alpha_n": np.full(_N, 3.0),
                "midhands_force_n": {"total": _vec(10.0), "ztcf": _vec(6.0)},
                "midhands_moment_nm": {"total": np.full(_N, 2.0)},
            },
            {
                "strategy_id": "min_squared_force",
                "evidence": "allocation-only",
                "force_left_n": _vec(5.0),
                "force_right_n": _vec(5.0),
                "alpha_n": np.full(_N, 0.0),
                "midhands_force_n": {"total": _vec(10.0), "ztcf": _vec(6.0)},
                "midhands_moment_nm": {"total": np.full(_N, 2.0)},
            },
        )

    missing = adapt_simulation_output(_Source()).strategies[0].midhands_force_n["zvcf"]
    assert missing.shape == (_N, 3) and np.isnan(missing).all()
    assert not np.any(missing == 0.0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"objective": "wasted energy of the grip"},
        {"objective": "Biological Optimality"},
        {"git_sha": "nope"},
        {"phases": (("Address", 0.0), ("Impact", 0.2))},
        {
            "strategies": (
                _loads("baseline", 8.0, 6.0, 3.0),
                _loads("baseline", 5.0, 5.0, 0.0),
            )
        },
    ],
)
def test_contract_rejects_bad_records(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _record(**overrides)
