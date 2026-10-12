"""Tests for the swing report builder (#12325, epic #11419, policy #10382)."""

from __future__ import annotations

import math
from typing import Any
from unittest import mock

from hypothesis import given, settings
from hypothesis import strategies as st
import numpy as np
import pytest

from src.shared.python.data_io.provenance import ProvenanceInfo
from src.shared.python.reporting.swing_report import builder as builder_mod
from src.shared.python.reporting.swing_report.adapters_base import (
    EngineResult,
    QuantityData,
    result_from_reporting_mixin,
)
from src.shared.python.reporting.swing_report.builder import build_swing_report
from src.shared.python.reporting.swing_report.contract import Capture
from src.shared.python.reporting.swing_report.policy import ADVERTISED, WITHHELD
from src.shared.python.reporting.swing_report.validator import validate_swing_report

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

HASH = "a" * 64
TIME = (0.0, 0.01, 0.02)
PUBLIC = Capture(capture_id="cap-1", visibility="public", capture_sha256=HASH)
PRIVATE = Capture(capture_id="cap-2", visibility="private")
# Fixed provenance keeps tests off the (slow) git subprocess probes.
INFO = ProvenanceInfo(
    timestamp_utc="2026-10-11T12:00:00Z",
    timestamp_local="2026-10-11T08:00:00-04:00",
    git_commit_sha=None,
)


def _speed(values: tuple[float | None, ...], units: str = "m/s") -> QuantityData:
    return QuantityData(units=units, frame="world_Zup", time_s=TIME, values=values)


def _result(engine: str, **quantities: QuantityData) -> EngineResult:
    return EngineResult(
        engine=engine,
        run_id=f"run-{engine}",
        engine_version="1.0.0",
        model_file_hash=HASH,
        run_class="forward_dynamics",
        quantities=quantities,
    )


def _two_engines() -> list[EngineResult]:
    return [
        _result("mujoco", clubhead_speed=_speed((10.0, 20.0, 30.0))),
        _result("drake", clubhead_speed=_speed((10.5, 19.5, 30.0))),
    ]


def _build(results: list[EngineResult], capture: Capture = PUBLIC, scope: str = "full"):
    return build_swing_report(results, capture, scope, provenance=INFO)


def _by_id(report: Any) -> dict[str, Any]:
    return {q.quantity_id: q for q in report.quantities}


def test_two_engines_advertise_speed_with_spread() -> None:
    report = _build(_two_engines())
    speed = _by_id(report)["clubhead_speed"]
    assert speed.status == "advertised_with_uncertainty"
    assert {s.engine for s in speed.series} == {"mujoco", "drake"}
    assert speed.uncertainty is not None
    assert speed.uncertainty.budget_source == "provisional_cross_engine_spread"
    assert speed.uncertainty.lower == pytest.approx((10.0, 19.5, 30.0))
    assert speed.uncertainty.upper == pytest.approx((10.5, 20.0, 30.0))
    validate_swing_report(report.to_dict())


def test_every_withheld_quantity_is_a_stub_without_numbers() -> None:
    report = _build(_two_engines())
    by_id = _by_id(report)
    for qid in WITHHELD:
        stub = by_id[qid]
        assert stub.status == "withheld"
        assert stub.series == () and stub.uncertainty is None
        assert stub.label == "not validated" and stub.reason


def test_withheld_engine_data_is_dropped_with_reason() -> None:
    leaky = _result(
        "mujoco",
        clubhead_speed=_speed((10.0, 20.0, 30.0)),
        joint_load=_speed((1.0, 2.0, 3.0), units="N*m"),
        face_angle=_speed((4.0, 5.0, 6.0), units="deg"),
        joint_angle=_speed((7.0, 8.0, 9.0), units="rad"),
    )
    other = _result("drake", clubhead_speed=_speed((10.5, 19.5, 30.0)))
    report = _build([leaky, other])
    doc = report.to_dict()
    by_id = _by_id(report)
    for qid in ("joint_load", "face_angle", "joint_angle"):
        assert by_id[qid].series == () and by_id[qid].reason
    leaked = {"1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "9.0"}
    withheld_docs = [q for q in doc["quantities"] if q["status"] == "withheld"]
    assert leaked.isdisjoint(str(withheld_docs))
    validate_swing_report(doc)


def test_missing_channel_is_null_never_zero() -> None:
    results = [
        _result("mujoco", clubhead_speed=_speed((10.0, None, 30.0))),
        _result("drake", clubhead_speed=_speed((10.5, 19.5, 30.0))),
    ]
    speed = _by_id(_build(results))["clubhead_speed"]
    mujoco = next(s for s in speed.series if s.engine == "mujoco")
    assert mujoco.values[1] is None
    assert all(0.0 not in s.values[1:2] for s in speed.series)
    # a sample with fewer than two finite engines has no advertised value
    drake = next(s for s in speed.series if s.engine == "drake")
    assert drake.values[1] is None


def test_engine_failure_is_unavailable_not_an_exception() -> None:
    failed = EngineResult(
        engine="drake",
        run_id="run-drake",
        engine_version=None,
        model_file_hash=None,
        run_class="forward_dynamics",
        error="engine not installed",
    )
    report = _build([_result("mujoco", clubhead_speed=_speed((1.0, 2.0, 3.0))), failed])
    assert "clubhead_speed" not in _by_id(report)
    assert any("engine not installed" in (s.notes or "") for s in report.sections)
    assert {r.engine for r in report.runs} == {"mujoco", "drake"}
    validate_swing_report(report.to_dict())


def test_all_engines_unavailable_still_validates() -> None:
    failed = [
        EngineResult(
            engine=e,
            run_id=f"run-{e}",
            engine_version=None,
            model_file_hash=None,
            run_class="forward_dynamics",
            error="boom",
        )
        for e in ("mujoco", "drake")
    ]
    report = _build(failed)
    assert not ADVERTISED & set(_by_id(report))
    validate_swing_report(report.to_dict())


def test_single_engine_is_not_advertised() -> None:
    one = [_result("mujoco", clubhead_speed=_speed((10.0, 20.0, 30.0)))]
    report = _build(one)
    assert "clubhead_speed" not in _by_id(report)
    notes = " ".join(s.notes or "" for s in report.sections)
    assert "single engine: no spread" in notes
    validate_swing_report(report.to_dict())


def test_mismatched_time_grid_excludes_engine_with_reason() -> None:
    odd = QuantityData(
        units="m/s", frame="world_Zup", time_s=(0.0, 0.02, 0.04), values=(1.0, 2.0, 3.0)
    )
    results = [*_two_engines(), _result("pinocchio", clubhead_speed=odd)]
    report = _build(results)
    speed = _by_id(report)["clubhead_speed"]
    assert {s.engine for s in speed.series} == {"mujoco", "drake"}
    assert "pinocchio" in " ".join(s.notes or "" for s in report.sections)
    validate_swing_report(report.to_dict())


def test_withheld_named_channel_is_excluded() -> None:
    sneaky = QuantityData(
        units="m/s",
        frame="world_Zup",
        time_s=TIME,
        values=(1.0, 2.0, 3.0),
        channel="joint_load",
    )
    results = [_result("mujoco", clubhead_speed=sneaky), *_two_engines()[1:]]
    report = _build(results)
    validate_swing_report(report.to_dict())
    assert "clubhead_speed" not in _by_id(report)


def test_private_capture_with_hash_is_rejected() -> None:
    bad = Capture(capture_id="c", visibility="private", capture_sha256=HASH)
    with pytest.raises(ValueError, match="capture_sha256"):
        _build(_two_engines(), capture=bad)


def test_private_capture_report_has_no_capture_hash() -> None:
    doc = _build(_two_engines(), capture=PRIVATE).to_dict()
    assert "capture_sha256" not in doc["provenance"]["capture"]


@pytest.mark.parametrize("bad_scope", ["", "partial", None])
def test_invalid_scope_is_rejected(bad_scope: Any) -> None:
    with pytest.raises(ValueError, match="scope"):
        _build(_two_engines(), scope=bad_scope)


def test_empty_results_are_rejected() -> None:
    with pytest.raises(ValueError, match="results"):
        _build([])


def test_duplicate_run_ids_are_rejected() -> None:
    dup = [_result("mujoco"), _result("mujoco")]
    with pytest.raises(ValueError, match="run_id"):
        _build(dup)


def test_custom_scope_is_passed_through_and_valid() -> None:
    report = _build(_two_engines(), scope="custom")
    assert report.scope == "custom"
    validate_swing_report(report.to_dict())


def test_provenance_comes_from_provenance_info_and_engine_package_version() -> None:
    info = ProvenanceInfo(
        timestamp_utc="2026-10-11T12:00:00Z",
        timestamp_local="2026-10-11T08:00:00-04:00",
        software_name="spy-suite",
        software_version="9.9.9",
        git_commit_sha="b" * 40,
        git_is_dirty=True,
    )
    versionless = _result("mujoco", clubhead_speed=_speed((10.0, 20.0, 30.0)))
    versionless = EngineResult(**{**versionless.__dict__, "engine_version": None})
    with (
        mock.patch.object(
            builder_mod.ProvenanceInfo, "capture", return_value=info
        ) as capture_spy,
        mock.patch.object(
            builder_mod, "engine_package_version", return_value="3.3.3"
        ) as version_spy,
    ):
        report = build_swing_report([versionless, _two_engines()[1]], PUBLIC, "full")
    capture_spy.assert_called_once()
    version_spy.assert_called_once()
    prov = report.provenance
    assert (prov.software_name, prov.software_version) == ("spy-suite", "9.9.9")
    assert prov.git_commit_sha == "b" * 40 and prov.git_is_dirty is True
    assert report.generated_utc == "2026-10-11T12:00:00Z"
    assert (
        next(r for r in report.runs if r.engine == "mujoco").engine_version == "3.3.3"
    )


def test_same_input_and_provenance_is_deterministic() -> None:
    first = _build(_two_engines())
    second = _build(_two_engines())
    assert first.to_dict() == second.to_dict()


class _FakeAnalyzer:
    """Duck-typed ReportingMixin host: only the attributes it reports from."""

    times = np.array([0.0, 0.01, 0.02])
    club_head_speed = np.array([1.0, np.nan, 3.0])


def test_reporting_mixin_adapter_maps_speed_without_recomputation() -> None:
    result = result_from_reporting_mixin(
        _FakeAnalyzer(),
        engine="mujoco",
        run_id="run-m",
        engine_version="3.2.0",
        model_file_hash=HASH,
        run_class="forward_dynamics",
    )
    data = result.quantities["clubhead_speed"]
    assert data.units == "m/s" and data.time_s == TIME
    assert data.values == (1.0, None, 3.0)  # NaN becomes null, never 0


def test_reporting_mixin_adapter_without_speed_has_no_quantity() -> None:
    class _NoSpeed(_FakeAnalyzer):
        club_head_speed = None

    result = result_from_reporting_mixin(
        _NoSpeed(),
        engine="mujoco",
        run_id="run-m",
        engine_version="3.2.0",
        model_file_hash=HASH,
        run_class="forward_dynamics",
    )
    assert "clubhead_speed" not in result.quantities


_values = st.lists(
    st.one_of(st.none(), st.floats(-1e3, 1e3, allow_nan=False)), min_size=3, max_size=3
)
_engine_names = st.lists(
    st.sampled_from(["mujoco", "drake", "pinocchio", "opensim"]),
    min_size=1,
    max_size=4,
    unique=True,
)
_qids = st.sampled_from(sorted(ADVERTISED | WITHHELD))


@settings(max_examples=60, deadline=None)
@given(
    engines=_engine_names,
    data=st.data(),
    failed=st.booleans(),
    scope=st.sampled_from(["full", "custom"]),
)
def test_builder_output_always_validates(
    engines: list[str], data: st.DataObject, failed: bool, scope: str
) -> None:
    results = []
    for engine in engines:
        if failed:
            results.append(
                EngineResult(
                    engine=engine,
                    run_id=f"run-{engine}",
                    engine_version=None,
                    model_file_hash=None,
                    run_class="forward_dynamics",
                    error="down",
                )
            )
            continue
        chosen = data.draw(st.lists(_qids, max_size=6, unique=True))
        quantities = {q: _speed(tuple(data.draw(_values)), units="m/s") for q in chosen}
        results.append(_result(engine, **quantities))
    report = build_swing_report(results, PUBLIC, scope, provenance=INFO)
    validate_swing_report(report.to_dict())
    for q in report.quantities:
        for s in q.series:
            assert all(v is None or math.isfinite(v) for v in s.values)
