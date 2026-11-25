"""3D fixture stills for midhands ZTCF + ZVCF = Total (#15494, #15725)."""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex

from hand_load_stills import PHASE_IDS, PHASE_LABELS, ComponentVectors, PhaseLoads
from hand_load_stills import StillRecord, adapt_simulation_output, closure_status
from hand_load_stills import render_tip_to_tail

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_PALETTE = "#000000 #e69f00 #56b4e9 #009e73 #f0e442 #0072b2 #d55e00 #cc79a7"
OKABE_ITO = set(_PALETTE.split())  # noqa: SIM905 - one line keeps the PR under 500
_SHA, _HASH = "a1b2c3d4e5f6", "0123456789abcdef"
_OBJECTIVE = "Minimum squared hand force versus baseline"
_IDS = (
    "address shaft_parallel_back top lead_arm_parallel "
    "shaft_parallel_down impact follow_through"
)
_Z, _V, _T = (0.0, -1.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 1.0)
_PARTS = {"ZTCF": "#0072b2", "ZVCF": "#d55e00", "Total": "#000000"}


def _cv(*rows: tuple[float, float, float]) -> ComponentVectors:
    return ComponentVectors(*(np.array(row, dtype=float) for row in rows))


def _phase(
    name: str, total: tuple[float, float, float] | None = None, *, blank: bool = False
) -> PhaseLoads:
    if blank:
        missing = (np.nan, np.nan, np.nan)
        return PhaseLoads(
            name, _cv(missing, missing, missing), _cv(missing, missing, missing)
        )
    force = _T if total is None else total
    moment = ((0.0, -0.4, 0.0), (0.0, 0.0, 0.4), (0.0, -0.4, 0.4))
    return PhaseLoads(name, _cv(_Z, _V, force), _cv(*moment))


def _record(**overrides: object) -> StillRecord:
    fields: dict[str, object] = {
        "git_sha": _SHA,
        "data_hash": _HASH,
        "objective": _OBJECTIVE,
        "evidence": "allocation-only",
        "phases": tuple(_phase(name) for name in PHASE_IDS),
    }
    fields.update(overrides)
    return StillRecord(**fields)  # type: ignore[arg-type]


def _ends(axis, label: str) -> tuple[np.ndarray, np.ndarray]:
    line = next(item for item in axis.lines if item.get_label() == label)
    xs, ys, zs = line.get_data_3d()
    return np.array([xs[0], ys[0], zs[0]]), np.array([xs[-1], ys[-1], zs[-1]])


def test_seven_phase_layout_closure_palette_and_footer() -> None:
    assert " ".join(PHASE_IDS) == _IDS and len(PHASE_LABELS) == 7
    fig = render_tip_to_tail(_record())
    try:
        titles = [axis.get_title() for axis in fig.axes]
        assert len(titles) == 14 and all(axis.name == "3d" for axis in fig.axes)
        for phase in PHASE_LABELS:
            assert f"{phase} Force (N)" in titles and f"{phase} Moment (N·m)" in titles
        texts = {text.get_gid(): text for text in fig.texts}
        assert texts["view-label"].get_text() == "Three-Dimensional Tip-to-Tail"
        assert texts["evidence-badge"].get_text() == "allocation-only"
        assert fig.get_suptitle() == f"Objective: {_OBJECTIVE}"
        footer = texts["provenance-footer"].get_text()
        assert _SHA in footer and _HASH in footer
        face = to_hex(texts["evidence-badge"].get_bbox_patch().get_facecolor()).lower()
        assert face in OKABE_ITO
        for axis in fig.axes:
            z_tail, z_tip = _ends(axis, "ZTCF")
            v_tail, v_tip = _ends(axis, "ZVCF")
            t_tail, t_tip = _ends(axis, "Total")
            np.testing.assert_allclose(z_tail, 0.0, atol=1e-9)
            np.testing.assert_allclose(v_tail, z_tip, atol=1e-9)
            np.testing.assert_allclose(t_tail, 0.0, atol=1e-9)
            np.testing.assert_allclose(v_tip, t_tip, atol=1e-8)
            assert "closure-failure" not in {text.get_gid() for text in axis.texts}
            for line in axis.lines:
                label = str(line.get_label())
                color = to_hex(line.get_color()).lower()
                assert color in OKABE_ITO and color == _PARTS[label]
            assert axis.get_xlabel() == "X" and axis.get_zlabel() == "Z"
    finally:
        plt.close(fig)
    replayed = render_tip_to_tail(_record(evidence="forward-replayed"))
    try:
        badge = next(
            text for text in replayed.texts if text.get_gid() == "evidence-badge"
        )
        assert badge.get_text() == "forward-replayed"
    finally:
        plt.close(replayed)


def test_closure_failure_is_shown_within_relative_tolerance() -> None:
    inside = (0.0, -1.0 - 5e-5, 1.0)
    outside = (0.0, -1.0 - 2e-4, 1.0)
    assert closure_status(_Z, _V, inside) == "closed"
    assert closure_status(_Z, _V, outside) == "open"
    fig = render_tip_to_tail(
        _record(phases=tuple(_phase(name, outside) for name in PHASE_IDS))
    )
    try:
        _, v_tip = _ends(fig.axes[0], "ZVCF")
        _, t_tip = _ends(fig.axes[0], "Total")
        assert np.linalg.norm(v_tip - t_tip) > 1e-4
        assert any(text.get_text() == "CLOSURE FAILED" for text in fig.axes[0].texts)
        assert any(line.get_label() == "_sum_tip" for line in fig.axes[0].lines)
    finally:
        plt.close(fig)


def test_nan_is_unavailable_and_zero_is_drawn() -> None:
    phases = {name: _phase(name) for name in PHASE_IDS}
    phases["impact"] = _phase("impact", blank=True)
    zero = (0.0, 0.0, 0.0)
    phases["top"] = PhaseLoads(
        "top", _cv(_Z, zero, _Z), _cv((0.0, -0.4, 0.0), zero, (0.0, -0.4, 0.0))
    )
    broken = (1.0, np.nan, 0.0)
    phases["address"] = _phase("address", broken)
    fig = render_tip_to_tail(_record(phases=tuple(phases[name] for name in PHASE_IDS)))
    try:
        address = fig.axes[0]
        top = next(axis for axis in fig.axes if axis.get_title().startswith("Top"))
        impact = [axis for axis in fig.axes if axis.get_title().startswith("Impact")]
        assert len(address.lines) == 0
        assert any(text.get_text() == "UNAVAILABLE" for text in address.texts)
        assert len(impact) == 2 and all(len(axis.lines) == 0 for axis in impact)
        _tail, z_tip = _ends(top, "ZTCF")
        v_tail, v_tip = _ends(top, "ZVCF")
        np.testing.assert_allclose(v_tail, z_tip)
        np.testing.assert_allclose(v_tip, z_tip)
        assert closure_status(broken, _V, broken) == "unavailable"
    finally:
        plt.close(fig)


def test_adapter_leaves_gaps_and_contract_rejects_bad_records() -> None:
    class _Source:
        git_sha, data_hash, objective, evidence = (
            _SHA,
            _HASH,
            _OBJECTIVE,
            "allocation-only",
        )
        phases = (
            {"name": "Address", "force_n": {"ztcf": _Z, "zvcf": _V}, "moment_nm": {}},
        )

    record = adapt_simulation_output(_Source())
    assert record.phases[0].name == "address" and len(record.phases) == 7
    assert np.isnan(record.phases[0].force_n.total).all()
    assert not np.any(record.phases[0].force_n.total == 0.0)
    assert np.isnan(record.phases[PHASE_IDS.index("impact")].moment_nm.ztcf).all()
    for overrides in (
        {"objective": "wasted energy of the grip"},
        {"objective": "Biological Optimality"},
        {"git_sha": "nope"},
        {"evidence": "replayed"},
        {"phases": tuple(_phase(name) for name in PHASE_IDS[:6])},
    ):
        with pytest.raises(ValueError):
            _record(**overrides)
    with pytest.raises(TypeError):
        render_tip_to_tail(object())  # type: ignore[arg-type]
