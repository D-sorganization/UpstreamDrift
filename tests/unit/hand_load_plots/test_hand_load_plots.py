"""Fixture tests for the hand-load allocation dashboard (#15494, #15725)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex

from hand_load_plots import (
    HandLoadPlotInput,
    StrategyLoads,
    adapt_simulation_output,
    plot_invariant_dashboard,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

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
_OBJECTIVE = "Minimum squared hand force versus baseline"


def _vec(value: float) -> np.ndarray:
    return np.tile(np.array([value, 0.0, 0.0]), (_N, 1))


def _loads(
    strategy_id: str,
    left: float,
    right: float,
    *,
    evidence: str = "allocation-only",
    left_force: np.ndarray | None = None,
    zvcf_moment: float | None = 0.8,
) -> StrategyLoads:
    moment = np.full(_N, np.nan if zvcf_moment is None else zvcf_moment)
    return StrategyLoads(
        strategy_id=strategy_id,
        evidence=evidence,
        force_left_n=_vec(left) if left_force is None else left_force,
        force_right_n=_vec(right),
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
        "objective": _OBJECTIVE,
        "strategies": (
            _loads("baseline", 8.0, 6.0),
            _loads("min_squared_force", 5.0, 5.0),
        ),
    }
    fields.update(overrides)
    return HandLoadPlotInput(**fields)  # type: ignore[arg-type]


def _y(axis, label: str) -> np.ndarray:
    found = [line for line in axis.lines if line.get_label() == label]
    assert len(found) == 1, label
    return np.asarray(found[0].get_ydata(), dtype=float)


def _chrome(fig, badge: str) -> None:
    assert fig.get_suptitle() == f"Objective: {_OBJECTIVE}"
    texts = {text.get_gid(): text.get_text() for text in fig.texts}
    assert texts["evidence-badge"] == badge
    assert _SHA in texts["provenance-footer"] and _HASH in texts["provenance-footer"]
    for axis in fig.axes:
        assert axis.get_xlabel() == "Time (s)"
        for line in axis.lines:
            assert to_hex(line.get_color()).lower() in OKABE_ITO


def test_invariants_match_and_hand_forces_shrink() -> None:
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

        def _match(axis, kind: str, pairs: tuple[tuple[str, float], ...]) -> None:
            for component, level in pairs:
                base = _y(axis, f"{component} {kind} baseline")
                other = _y(axis, f"{component} {kind} min squared force")
                np.testing.assert_allclose(base, other)
                np.testing.assert_allclose(base, np.full(_N, level))

        _match(force, "force", (("Total", 10.0), ("ZTCF", 6.0), ("ZVCF", 4.0)))
        _match(moment, "moment", (("Total", 2.0), ("ZTCF", 1.2), ("ZVCF", 0.8)))
        assert np.all(
            _y(left, "LH force min squared force") < _y(left, "LH force baseline")
        )
        assert np.all(
            _y(right, "RH force min squared force") < _y(right, "RH force baseline")
        )
    finally:
        plt.close(fig)


def test_nan_stays_missing_and_blank_series_is_unavailable() -> None:
    left = _vec(8.0)
    left[2] = np.nan
    left[3] = 0.0
    record = _record(
        strategies=(
            _loads("baseline", 8.0, 6.0, left_force=left, zvcf_moment=None),
            _loads("min_squared_force", 5.0, 5.0),
        )
    )
    fig = plot_invariant_dashboard(record)
    try:
        drawn = _y(fig.axes[2], "LH force baseline")
        assert np.isnan(drawn[2]) and drawn[2] != 0.0
        assert drawn[3] == 0.0
        labels = [line.get_label() for line in fig.axes[1].lines]
        assert "ZVCF moment baseline" not in labels
        text = " ".join(item.get_text() for item in fig.axes[1].texts)
        assert "UNAVAILABLE" in text
    finally:
        plt.close(fig)


def test_forward_replayed_badge_and_adapter_leaves_gaps() -> None:
    record = _record(
        strategies=(
            _loads("baseline", 8.0, 6.0, evidence="forward-replayed"),
            _loads("min_squared_force", 5.0, 5.0, evidence="forward-replayed"),
        )
    )
    fig = plot_invariant_dashboard(record)
    try:
        badge = next(text for text in fig.texts if text.get_gid() == "evidence-badge")
        assert badge.get_text() == "forward-replayed"
    finally:
        plt.close(fig)

    def _mapping(strategy_id: str, left: float, right: float) -> dict[str, object]:
        return {
            "strategy_id": strategy_id,
            "evidence": "allocation-only",
            "force_left_n": _vec(left),
            "force_right_n": _vec(right),
            "midhands_force_n": {"total": _vec(10.0), "ztcf": _vec(6.0)},
            "midhands_moment_nm": {"total": np.full(_N, 2.0)},
        }

    class _Source:
        time_s = _TIME
        git_sha = _SHA
        data_hash = _HASH
        objective = _OBJECTIVE
        strategies = (
            _mapping("baseline", 8.0, 6.0),
            _mapping("min_squared_force", 5.0, 5.0),
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
        {"strategies": (_loads("baseline", 8.0, 6.0), _loads("baseline", 5.0, 5.0))},
    ],
)
def test_contract_rejects_bad_records(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _record(**overrides)
