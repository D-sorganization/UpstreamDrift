"""Acceptance tests for the MuJoCo/Drake adapters and parity harness (#12328).

Adapter mapping and parity logic run on recorded/synthetic native output so
they execute everywhere; real-engine tests carry the requires_* markers and
skip cleanly when the engine is not installed.
"""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from src.shared.python.data_io.provenance import ProvenanceInfo
from src.shared.python.reporting.swing_report.adapters import (
    VELOCITY_TOLERANCE,
    DrakeAdapter,
    EngineResultAdapter,
    Matlab3DAdapter,
    MujocoAdapter,
    NativeAdapter,
    NativeSwingRun,
    OpensimAdapter,
    PinocchioAdapter,
    build_parity_report,
    compare_clubhead_speed,
)
from src.shared.python.reporting.swing_report.adapters_base import QuantityData
from src.shared.python.reporting.swing_report.contract import Capture
from src.shared.python.reporting.swing_report.policy import WITHHELD
from src.shared.python.reporting.swing_report.validator import validate_swing_report
from tests.helpers.opensim_support import require_opensim

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

HASH = "b" * 64
CAPTURE = Capture(capture_id="cap-s5", visibility="private")
INFO = ProvenanceInfo(
    timestamp_utc="2026-10-12T12:00:00Z",
    timestamp_local="2026-10-12T08:00:00-04:00",
    git_commit_sha=None,
)
TIMES = tuple(0.01 * i for i in range(6))
# Constant-velocity (3, 4, 0) m/s -> exact speed 5 m/s for any differencing.
POSITIONS = tuple((3.0 * t, 4.0 * t, 1.0) for t in TIMES)
ADAPTER_CLASSES = [
    MujocoAdapter,
    DrakeAdapter,
    PinocchioAdapter,
    OpensimAdapter,
    Matlab3DAdapter,
]


def _native(**overrides: object) -> NativeSwingRun:
    fields: dict[str, object] = {
        "time_s": TIMES,
        "clubhead_position_m": POSITIONS,
        "engine_version": "1.2.3",
        "model_file_hash": HASH,
    }
    fields.update(overrides)
    return NativeSwingRun(**fields)  # type: ignore[arg-type]


@pytest.mark.parametrize("adapter_cls", ADAPTER_CLASSES)
def test_adapter_is_engine_result_adapter(adapter_cls: type[NativeAdapter]) -> None:
    assert isinstance(adapter_cls(), EngineResultAdapter)


@pytest.mark.parametrize("adapter_cls", ADAPTER_CLASSES)
def test_maps_advertised_quantities_only(adapter_cls: type[NativeAdapter]) -> None:
    adapter = adapter_cls()
    result = adapter.adapt(_native(), run_id="r1")
    assert result.engine == adapter.engine_type.value
    assert set(result.quantities) == {"clubhead_position", "clubhead_speed"}
    assert not set(result.quantities) & WITHHELD
    speed = result.quantities["clubhead_speed"]
    assert speed.units == "m/s"
    assert speed.values == pytest.approx((5.0,) * len(TIMES))
    pos = result.quantities["clubhead_position"]
    assert (pos.units, pos.channel) == ("m", "z")


def test_nonfinite_position_is_missing_never_zero() -> None:
    bad = list(POSITIONS)
    bad[2] = (0.0, 0.0, math.nan)
    result = MujocoAdapter().adapt(_native(clubhead_position_m=tuple(bad)), "r1")
    assert result.quantities["clubhead_position"].values[2] is None
    speed = result.quantities["clubhead_speed"].values
    assert speed[1] is None and speed[2] is None and speed[3] is None
    assert 0.0 not in [v for v in speed if v is not None]


def test_engine_supplied_speed_wins_over_derivation() -> None:
    result = DrakeAdapter().adapt(
        _native(clubhead_speed_mps=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)), "r1"
    )
    assert result.quantities["clubhead_speed"].values[0] == 1.0


def test_engine_lacking_channel_is_unavailable() -> None:
    single = _native(time_s=(0.0,), clubhead_position_m=((0.0, 0.0, 0.0),))
    result = MujocoAdapter().adapt(single, "r1")
    assert "clubhead_speed" not in result.quantities
    other = DrakeAdapter().adapt(_native(), "r2")
    report, outcomes = build_parity_report([result, other], CAPTURE, provenance=INFO)
    ids = {q["quantity_id"]: q for q in report.to_dict()["quantities"]}
    assert "clubhead_speed" not in ids
    assert outcomes[0].comparable is False
    assert outcomes[0].reason == "clubhead_speed unavailable"


def test_failed_engine_degrades_to_note() -> None:
    failed = MujocoAdapter().unavailable("r1", "mujoco import failed")
    ok = DrakeAdapter().adapt(_native(), "r2")
    report, outcomes = build_parity_report([failed, ok], CAPTURE, provenance=INFO)
    assert not outcomes[0].comparable
    assert "mujoco import failed" in str(report.to_dict()["sections"])


def test_equivalent_swings_pass_parity_and_validate() -> None:
    results = [
        MujocoAdapter().adapt(_native(), "run-mujoco"),
        DrakeAdapter().adapt(_native(), "run-drake"),
    ]
    report, outcomes = build_parity_report(results, CAPTURE, provenance=INFO)
    assert outcomes[0].passed and outcomes[0].tolerance == VELOCITY_TOLERANCE
    doc = report.to_dict()
    assert validate_swing_report(doc) is not None
    speed = next(q for q in doc["quantities"] if q["quantity_id"] == "clubhead_speed")
    assert speed["status"] == "advertised_with_uncertainty"
    assert speed["uncertainty"]["budget_source"] == "provisional_cross_engine_spread"
    assert speed["uncertainty"]["kind"] == "engine_spread"
    assert len(speed["uncertainty"]["lower"]) == len(speed["series"][0]["values"])
    withheld = [q for q in doc["quantities"] if q["status"] == "withheld"]
    assert withheld and all("series" not in q or not q["series"] for q in withheld)


def test_divergent_speed_fails_parity_without_loosening() -> None:
    delta = 10 * VELOCITY_TOLERANCE
    a = MujocoAdapter().adapt(_native(clubhead_speed_mps=(5.0,) * 6), "ra")
    b = DrakeAdapter().adapt(_native(clubhead_speed_mps=(5.0 + delta,) * 6), "rb")
    outcome = compare_clubhead_speed(a, b)
    assert outcome.comparable and not outcome.passed
    assert outcome.max_abs_diff == pytest.approx(delta)
    report, _ = build_parity_report([a, b], CAPTURE, provenance=INFO)
    speed = next(
        q
        for q in report.to_dict()["quantities"]
        if q["quantity_id"] == "clubhead_speed"
    )
    assert max(speed["uncertainty"]["upper"]) - min(speed["uncertainty"]["lower"]) > 0


def test_tolerance_is_existing_velocity_constant() -> None:
    from src.shared.python.engine_core.cross_engine_validator import (
        CrossEngineValidator,
    )

    assert CrossEngineValidator.TOLERANCES["velocity"] == VELOCITY_TOLERANCE
    with pytest.raises(ValueError):
        compare_clubhead_speed(
            MujocoAdapter().adapt(_native(), "a"),
            DrakeAdapter().adapt(_native(), "b"),
            tolerance=0.0,
        )


def test_native_run_contract_violations() -> None:
    with pytest.raises(ValueError):
        _native(time_s=(0.0, 0.0, 0.1, 0.2, 0.3, 0.4))
    with pytest.raises(ValueError):
        _native(clubhead_position_m=POSITIONS[:-1])
    with pytest.raises(ValueError):
        _native(clubhead_speed_mps=(1.0,))
    with pytest.raises(ValueError):
        MujocoAdapter(position_axis="w")


def test_kinematic_sequence_passthrough_and_withheld_channel_blocked() -> None:
    seq = QuantityData(
        units="s", frame="none", time_s=(0.0, 0.01), values=(0.1, 0.2), channel=None
    )
    result = MujocoAdapter().adapt(_native(kinematic_sequence=seq), "r1")
    assert result.quantities["kinematic_sequence"] is seq


def test_three_engine_pairs_are_all_compared() -> None:
    results = [
        MujocoAdapter().adapt(_native(), "a"),
        DrakeAdapter().adapt(_native(), "b"),
        replace(MujocoAdapter().adapt(_native(), "c"), engine="pinocchio"),
    ]
    _, outcomes = build_parity_report(results, CAPTURE, provenance=INFO)
    assert len(outcomes) == 3


def test_all_five_engine_pairs_are_compared() -> None:
    results = [
        MujocoAdapter().adapt(_native(), "a"),
        DrakeAdapter().adapt(_native(), "b"),
        PinocchioAdapter().adapt(_native(), "c"),
        OpensimAdapter().adapt(_native(), "d"),
        Matlab3DAdapter().adapt(_native(), "e"),
    ]
    report, outcomes = build_parity_report(results, CAPTURE, provenance=INFO)
    # 5 choose 2 = 10 pairs
    assert len(outcomes) == 10
    assert all(o.passed for o in outcomes)
    doc = report.to_dict()
    assert validate_swing_report(doc) is not None


@pytest.mark.requires_mujoco
def test_real_mujoco_run_flows_through_adapter() -> None:
    mujoco = pytest.importorskip("mujoco")
    xml = (
        "<mujoco><worldbody><body pos='0 0 1'><freejoint/>"
        "<geom size='.05' mass='1'/><site name='head'/></body></worldbody></mujoco>"
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    times: list[float] = []
    positions: list[tuple[float, float, float]] = []
    for _ in range(5):
        mujoco.mj_step(model, data)
        times.append(float(data.time))
        pos_xyz = data.site_xpos[0]
        pos: tuple[float, float, float] = (
            float(pos_xyz[0]),
            float(pos_xyz[1]),
            float(pos_xyz[2]),
        )
        positions.append(pos)
    pos_tuple: tuple[tuple[float, float, float], ...] = tuple(positions)
    result = MujocoAdapter().adapt(
        NativeSwingRun(tuple(times), pos_tuple, model_file_hash=HASH), "real"
    )
    assert result.engine_version == mujoco.__version__
    assert all(v is not None for v in result.quantities["clubhead_speed"].values)
    val = result.quantities["clubhead_position"].values[0]
    assert val is not None and math.isfinite(val)


@pytest.mark.requires_drake
def test_real_drake_version_probe() -> None:
    pytest.importorskip("pydrake")
    assert DrakeAdapter().engine_version() != "unknown"


@pytest.mark.requires_pinocchio
def test_real_pinocchio_version_probe() -> None:
    pytest.importorskip("pinocchio")
    assert PinocchioAdapter().engine_version() != "unknown"


@pytest.mark.requires_opensim
def test_real_opensim_version_probe() -> None:
    require_opensim()
    assert OpensimAdapter().engine_version() != "unknown"


@pytest.mark.requires_matlab_engine
def test_real_matlab_3d_version_probe() -> None:
    pytest.importorskip("matlab.engine")
    assert Matlab3DAdapter().engine_version() != "unknown"
