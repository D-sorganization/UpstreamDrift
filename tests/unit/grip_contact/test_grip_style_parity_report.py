"""Cross-engine parity extension and report tests for grip-style pairs (#13308, HRG-GS-G10).

Covers:
1. Synthetic unit tests (fast, no engines) of:
   - per-pair normal force, per-pair tangential force, and contact-point drift in
     grip_contact/parity.py (reusing QuantityError, parity_errors, and tolerances);
   - explicit DETERMINATE vs INDETERMINATE classification and labelling citing
     GRIP_PARITY_DECISIONS.md section 19;
   - unavailable engine NaN-plus-reason path (never zero, never a pass);
   - allocated hand-hand pair NaN-plus-reason reporting;
   - JSON and Markdown report generation via scripts/grip/grip_style_parity_report.py.
2. Slow engine tests marked requires_* + slow on 0.04 s hold and 0.04 s swing across
   overlap, interlock, and ten_finger asserting DETERMINATE quantities pass for
   available engines (with strict xfail and measured numbers where an engine genuinely
   exceeds tolerance on the 0.04 s window).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.grip_contact.contact_run import ContactRun
from src.shared.python.grip_contact.contact_topology import (
    GripStyle,
    contact_topology,
)
from src.shared.python.grip_contact.parity import (
    DETERMINATE_QUANTITIES,
    INDETERMINACY_CITATION,
    INDETERMINATE_LABEL,
    INDETERMINATE_QUANTITIES,
    PAIR_PARITY_QUANTITIES,
    PARITY_QUANTITIES,
    PEAK_TOLERANCE,
    RMS_TOLERANCE,
    STYLE_PARITY_QUANTITIES,
    ClassifiedQuantityResult,
    EngineParityRow,
    GripKineticsSeries,
    ParityState,
    QuantityClassification,
    StyleParityReport,
    build_style_parity_report,
    classify_parity_quantity,
    parity_errors,
)
from src.shared.python.motion_matching.grip_contract import GripSpec

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "docs/development/full_body_models"
FIXTURES = ROOT / "tests/fixtures/club_face"
SPEC_PATH = MODELS / "full_body_spec_anthro_driver.json"
SQUEEZE_N = 1100.0
DURATION_S = 0.04


def _synthetic_series(
    scale: float = 1.0,
    internal_scale: float = 1.0,
    n: int = 21,
    engine: str = "synthetic",
) -> GripKineticsSeries:
    """Build a synthetic GripKineticsSeries with controllable net vs internal scale."""
    t = np.arange(n, dtype=float) * 0.002
    wave = (1.0 + 0.2 * np.sin(2.0 * np.pi * 10.0 * t))[:, None]
    f_net = scale * np.hstack([30.0 * wave, -10.0 * wave, 15.0 * wave])
    f_int = internal_scale * np.hstack([200.0 * wave, 50.0 * wave, -20.0 * wave])
    f_l = 0.5 * f_net + f_int
    f_r = 0.5 * f_net - f_int
    p_l = np.tile([0.0, 0.0, 1.0], (n, 1))
    p_r = np.tile([0.08, 0.0, 1.0], (n, 1))
    rot = np.tile(np.eye(3), (n, 1, 1))
    tau_net = scale * np.hstack([2.0 * wave, 4.0 * wave, -1.5 * wave])
    # Choose tau_L and tau_R so that the moment about the midpoint is tau_net
    # regardless of internal_scale:
    # (p_L - p_mid) x f_L + (p_R - p_mid) x f_R = (p_L - p_R) x f_int
    dp = p_l - p_r
    m_int = np.cross(dp, f_int)
    tau_l = 0.5 * (tau_net - m_int)
    tau_r = 0.5 * (tau_net - m_int)
    return GripKineticsSeries(
        engine=engine,
        time_s=t,
        force_on_club_n={"L": f_l, "R": f_r},
        torque_on_club_nm={"L": tau_l, "R": tau_r},
        grip_point_m={"L": p_l, "R": p_r},
        deflection_m={
            "L": internal_scale * 1e-4 * wave * np.array([1.0, 0.5, -0.2]),
            "R": internal_scale * 1e-4 * wave * np.array([-0.8, 0.4, 0.3]),
        },
        rotation_deflection_rad={
            "L": internal_scale * 1e-3 * wave[:, 0],
            "R": internal_scale * 1.2e-3 * wave[:, 0],
        },
        club_rotation=rot,
    )


def _synthetic_contact_run(
    style: GripStyle = GripStyle.OVERLAP,
    net_scale: float = 1.0,
    internal_scale: float = 1.0,
    pair_scale: float = 1.0,
    drift_scale: float = 1.0,
    n: int = 21,
    engine: str = "synthetic",
) -> ContactRun:
    """Build a synthetic ContactRun with per-pair traces for ``style``."""
    series = _synthetic_series(
        scale=net_scale, internal_scale=internal_scale, n=n, engine=engine
    )
    topo = contact_topology(GripSpec(style=style))
    t = series.time_s
    wave = 1.0 + 0.15 * np.sin(2.0 * np.pi * 8.0 * t)
    pair_normal: dict[str, np.ndarray] = {}
    pair_tangential: dict[str, np.ndarray] = {}
    pair_drift: dict[str, np.ndarray] = {}
    reasons: dict[str, str] = {}

    for idx, pair in enumerate(topo.pairs):
        if pair.is_hand_grip:
            base_fn = 40.0 + 3.0 * idx
            base_ft = 4.0 + 0.5 * idx
            base_drift = 6.0e-5 + 5.0e-6 * idx
            pair_normal[pair.name] = pair_scale * base_fn * wave
            pair_tangential[pair.name] = pair_scale * base_ft * wave
            pair_drift[pair.name] = drift_scale * base_drift * wave
        else:
            nan_arr = np.full(n, float("nan"), dtype=float)
            pair_normal[pair.name] = nan_arr.copy()
            pair_tangential[pair.name] = nan_arr.copy()
            pair_drift[pair.name] = nan_arr.copy()
            reasons[pair.name] = (
                "hand-hand pairs are allocated, not physical forces under prescribed hand motion"
            )

    normal_side = {
        "L": np.column_stack(
            [
                pair_normal[p.name]
                for p in topo.hand_grip_pairs()
                if topo.site(p.site_a).hand == "lead"
            ]
        ),
        "R": np.column_stack(
            [
                pair_normal[p.name]
                for p in topo.hand_grip_pairs()
                if topo.site(p.site_a).hand == "trail"
            ]
        ),
    }
    zeros = np.zeros(n, dtype=float)
    return ContactRun(
        series=series,
        normal_force_n=normal_side,
        roll_slip_rad={"L": zeros.copy(), "R": zeros.copy()},
        axial_slip_m={"L": zeros.copy(), "R": zeros.copy()},
        pair_normal_force_n=pair_normal,
        pair_tangential_force_n=pair_tangential,
        pair_contact_point_drift_m=pair_drift,
        channel_reasons=reasons,
    )


@pytest.mark.unit
def test_quantity_tables_and_classification_contract() -> None:
    """Every quantity is explicitly classified as DETERMINATE or INDETERMINATE."""
    assert PEAK_TOLERANCE == 0.05
    assert RMS_TOLERANCE == 0.02
    assert PAIR_PARITY_QUANTITIES == (
        "pair_normal_force",
        "pair_tangential_force",
        "contact_point_drift",
    )
    assert DETERMINATE_QUANTITIES == ("net_force", "net_moment_com")
    assert set(PAIR_PARITY_QUANTITIES).issubset(set(INDETERMINATE_QUANTITIES))
    assert set(STYLE_PARITY_QUANTITIES) == (
        set(PARITY_QUANTITIES) | {"net_moment_com"} | set(PAIR_PARITY_QUANTITIES)
    )
    assert (
        INDETERMINATE_LABEL
        == "indeterminate (hyperstatic two-hand grip; not identified by club kinematics)"
    )
    assert (
        "docs/development/full_body_models/GRIP_PARITY_DECISIONS.md"
        in INDETERMINACY_CITATION
    )
    assert "19" in INDETERMINACY_CITATION

    for q in DETERMINATE_QUANTITIES:
        assert classify_parity_quantity(q) == QuantityClassification.DETERMINATE
    for q in INDETERMINATE_QUANTITIES:
        assert classify_parity_quantity(q) == QuantityClassification.INDETERMINATE
    for prefix in PAIR_PARITY_QUANTITIES:
        assert (
            classify_parity_quantity(f"{prefix}:lead_palm_proximal_grip")
            == QuantityClassification.INDETERMINATE
        )

    with pytest.raises(TypeError, match="str"):
        classify_parity_quantity(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown parity quantity"):
        classify_parity_quantity("not_a_quantity")


@pytest.mark.unit
def test_parity_errors_extends_to_pair_forces_and_contact_point_drift() -> None:
    """parity_errors compares ContactRun including per-pair normal/tangential force and drift."""
    ref = _synthetic_contact_run(style=GripStyle.OVERLAP, engine="pinocchio")
    cand_same = _synthetic_contact_run(style=GripStyle.OVERLAP, engine="drake")
    errs_same = parity_errors(cand_same, ref)
    assert set(errs_same) == set(STYLE_PARITY_QUANTITIES)
    for name, qe in errs_same.items():
        assert qe.peak_error == pytest.approx(0.0, abs=1e-12), name
        assert qe.rms_error == pytest.approx(0.0, abs=1e-12), name
        assert qe.passes() is True

    cand_scaled = _synthetic_contact_run(
        style=GripStyle.OVERLAP,
        net_scale=1.01,
        internal_scale=1.40,
        pair_scale=1.25,
        drift_scale=1.15,
        engine="mujoco",
    )
    errs = parity_errors(cand_scaled, ref)
    assert errs["net_force"].peak_error == pytest.approx(0.01, rel=1e-6)
    assert errs["net_moment_com"].peak_error == pytest.approx(0.01, rel=1e-6)
    assert errs["net_force"].passes() is True
    assert errs["net_moment_com"].passes() is True
    assert errs["pair_normal_force"].peak_error == pytest.approx(0.25, rel=1e-6)
    assert errs["pair_tangential_force"].peak_error == pytest.approx(0.25, rel=1e-6)
    assert errs["contact_point_drift"].peak_error == pytest.approx(0.15, rel=1e-6)


@pytest.mark.unit
def test_report_builder_determinate_and_indeterminate_classification() -> None:
    """Report builder classifies determinate vs indeterminate and never counts indeterminate as pass."""
    ref = _synthetic_contact_run(style=GripStyle.INTERLOCK, engine="pinocchio")
    drake_run = _synthetic_contact_run(
        style=GripStyle.INTERLOCK,
        net_scale=1.01,
        internal_scale=1.35,
        pair_scale=1.20,
        drift_scale=1.10,
        engine="drake",
    )
    mujoco_run = _synthetic_contact_run(
        style=GripStyle.INTERLOCK,
        net_scale=1.08,
        internal_scale=1.50,
        pair_scale=1.30,
        drift_scale=1.25,
        engine="mujoco",
    )

    report = build_style_parity_report(
        style=GripStyle.INTERLOCK,
        motion="hold",
        duration_s=DURATION_S,
        reference_run=ref,
        candidate_runs={"drake": drake_run, "mujoco": mujoco_run},
        unavailable_engines={"opensim": "ImportError: No module named 'opensim'"},
    )
    assert isinstance(report, StyleParityReport)
    assert report.style == "interlock"
    assert report.motion == "hold"
    assert set(report.rows) == {"pinocchio", "opensim", "mujoco", "drake"}

    pin_row = report.row("pinocchio")
    assert pin_row.available is True
    assert pin_row.determinate_passes() is True
    for q in DETERMINATE_QUANTITIES:
        q_res = pin_row.quantity(q)
        assert q_res.classification == QuantityClassification.DETERMINATE
        assert q_res.state == ParityState.PASS
        assert q_res.passes() is True
    for q in INDETERMINATE_QUANTITIES:
        q_res = pin_row.quantity(q)
        assert q_res.classification == QuantityClassification.INDETERMINATE
        assert q_res.state == ParityState.INDETERMINATE
        assert q_res.passes() is False
        assert q_res.label == INDETERMINATE_LABEL
        assert q_res.citation == INDETERMINACY_CITATION

    drake_row = report.row("drake")
    assert drake_row.available is True
    assert drake_row.determinate_passes() is True
    assert drake_row.quantity("net_force").state == ParityState.PASS
    assert drake_row.quantity("net_moment_com").state == ParityState.PASS
    for q in INDETERMINATE_QUANTITIES:
        q_res = drake_row.quantity(q)
        assert q_res.classification == QuantityClassification.INDETERMINATE
        assert q_res.state == ParityState.INDETERMINATE
        assert q_res.passes() is False
        assert q_res.label == INDETERMINATE_LABEL
        assert math.isfinite(q_res.peak_error) and q_res.peak_error > 0.0
        assert math.isfinite(q_res.rms_error) and q_res.rms_error > 0.0

    mj_row = report.row("mujoco")
    assert mj_row.available is True
    assert mj_row.determinate_passes() is False
    assert mj_row.quantity("net_force").state == ParityState.FAIL
    assert mj_row.quantity("net_force").passes() is False
    assert set(mj_row.failing_determinate()) == {"net_force", "net_moment_com"}


@pytest.mark.unit
def test_report_builder_unavailable_nan_plus_reason_and_pair_breakdown() -> None:
    """Unavailable engines and allocated hand-hand pairs report NaN+reason, never zero or pass."""
    ref = _synthetic_contact_run(style=GripStyle.INTERLOCK, engine="pinocchio")
    drake_run = _synthetic_contact_run(
        style=GripStyle.INTERLOCK,
        net_scale=1.01,
        internal_scale=1.35,
        pair_scale=1.20,
        drift_scale=1.10,
        engine="drake",
    )
    report = build_style_parity_report(
        style=GripStyle.INTERLOCK,
        motion="hold",
        duration_s=DURATION_S,
        reference_run=ref,
        candidate_runs={"drake": drake_run},
        unavailable_engines={
            "opensim": "ImportError: No module named 'opensim'",
            "mujoco": "ImportError: No module named 'mujoco'",
        },
    )

    drake_row = report.row("drake")
    topo = contact_topology(GripSpec(style=GripStyle.INTERLOCK))
    for pair in topo.hand_grip_pairs():
        for metric in PAIR_PARITY_QUANTITIES:
            pq = drake_row.pair_quantity(pair.name, metric)
            assert pq.classification == QuantityClassification.INDETERMINATE
            assert pq.state == ParityState.INDETERMINATE
            assert pq.passes() is False
            assert pq.label == INDETERMINATE_LABEL
            assert math.isfinite(pq.peak_error)
            assert math.isfinite(pq.rms_error)
    for pair in topo.hand_hand_pairs():
        for metric in PAIR_PARITY_QUANTITIES:
            pq = drake_row.pair_quantity(pair.name, metric)
            assert pq.state == ParityState.UNAVAILABLE
            assert pq.passes() is False
            assert math.isnan(pq.peak_candidate)
            assert math.isnan(pq.peak_error)
            assert math.isnan(pq.rms_error)
            assert "allocated" in pq.reason

    osim_row = report.row("opensim")
    assert osim_row.available is False
    assert osim_row.determinate_passes() is False
    assert "ImportError" in osim_row.unavailable_reason
    for q in STYLE_PARITY_QUANTITIES:
        q_res = osim_row.quantity(q)
        assert q_res.state == ParityState.UNAVAILABLE
        assert q_res.passes() is False
        assert math.isnan(q_res.peak_candidate)
        assert math.isnan(q_res.peak_error)
        assert math.isnan(q_res.rms_error)
        assert "ImportError" in q_res.reason

    payload = report.to_dict()
    assert payload["style"] == "interlock"
    assert payload["citation"] == INDETERMINACY_CITATION
    md = report.to_markdown()
    assert "interlock" in md
    assert INDETERMINATE_LABEL in md
    assert "ImportError: No module named 'opensim'" in md
    assert "NaN" in md


@pytest.mark.unit
def test_script_write_parity_report_bundle_and_unavailable_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """scripts/grip/grip_style_parity_report.py writes JSON + Markdown and handles ImportError."""
    import scripts.grip.grip_style_parity_report as report_script

    ref = _synthetic_contact_run(style=GripStyle.OVERLAP, engine="pinocchio")
    drake_run = _synthetic_contact_run(
        style=GripStyle.OVERLAP,
        net_scale=1.005,
        internal_scale=1.20,
        pair_scale=1.10,
        drift_scale=1.05,
        engine="drake",
    )
    rep = build_style_parity_report(
        style=GripStyle.OVERLAP,
        motion="hold",
        duration_s=DURATION_S,
        reference_run=ref,
        candidate_runs={"drake": drake_run},
        unavailable_engines={
            "opensim": "ImportError: No module named 'opensim'",
            "mujoco": "ImportError: No module named 'mujoco'",
        },
    )
    json_path, md_path = report_script.write_parity_report_bundle(
        tmp_path, {"hold": [rep]}, duration_s=DURATION_S
    )
    assert json_path.is_file()
    assert md_path.is_file()
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["issue"] == 13308
    assert loaded["determinate_quantities"] == list(DETERMINATE_QUANTITIES)
    assert loaded["pair_parity_quantities"] == list(PAIR_PARITY_QUANTITIES)
    assert loaded["indeterminate_label"] == INDETERMINATE_LABEL
    assert loaded["citation"] == INDETERMINACY_CITATION
    assert "overlap" in loaded["motions"]["hold"]
    md_text = md_path.read_text(encoding="utf-8")
    assert "Cross-Engine Grip-Style Parity Report (HRG-GS-G10, #13308)" in md_text
    assert INDETERMINATE_LABEL in md_text
    assert "ImportError: No module named 'opensim'" in md_text

    def _raise_import_error(_engine: str) -> Any:
        raise ImportError("simulated missing engine")

    monkeypatch.setattr(report_script, "_import_engine_simulator", _raise_import_error)
    run_out, reason = report_script.run_candidate_engine(
        "opensim", "hold", GripStyle.OVERLAP, DURATION_S
    )
    assert run_out is None
    assert reason is not None and "ImportError: simulated missing engine" in reason


_ENGINE_RUN_CACHE: dict[tuple[str, str, str], tuple[ContactRun | None, str | None]] = {}


def _cached_engine_run(
    engine: str, style: GripStyle, motion: str
) -> tuple[ContactRun | None, str | None]:
    """Run or return cached 0.04 s ContactRun for (engine, style, motion)."""
    key = (engine, style.value, motion)
    if key not in _ENGINE_RUN_CACHE:
        import scripts.grip.grip_style_parity_report as report_script

        if engine == "pinocchio":
            try:
                sim_fn = report_script._import_engine_simulator("pinocchio")  # noqa: SLF001
            except ImportError as exc:
                _ENGINE_RUN_CACHE[key] = (None, f"{type(exc).__name__}: {exc}")
                return _ENGINE_RUN_CACHE[key]
            spec_bytes, swing, interface, topo_model = report_script._load_style_inputs(
                style
            )  # noqa: SLF001
            run = sim_fn(  # type: ignore[operator]
                spec_bytes,
                swing,
                interface=interface,
                t_end_s=DURATION_S,
                hold=(motion == "hold"),
                topology_model=topo_model,
            )
            _ENGINE_RUN_CACHE[key] = (run, None)
        else:
            _ENGINE_RUN_CACHE[key] = report_script.run_candidate_engine(
                engine, motion, style, DURATION_S
            )
    return _ENGINE_RUN_CACHE[key]


def _cached_style_report(style: GripStyle, motion: str) -> StyleParityReport:
    """Build or reuse the real 0.04 s StyleParityReport across all 4 engines."""
    ref_run, ref_reason = _cached_engine_run("pinocchio", style, motion)
    if ref_run is None:
        pytest.skip(f"pinocchio reference unavailable: {ref_reason}")
    cand_runs: dict[str, ContactRun] = {}
    unavail: dict[str, str] = {}
    for eng in ("opensim", "mujoco", "drake"):
        run, reason = _cached_engine_run(eng, style, motion)
        if run is not None:
            cand_runs[eng] = run
        elif reason is not None:
            unavail[eng] = reason
    return build_style_parity_report(
        style=style,
        motion=motion,
        duration_s=DURATION_S,
        reference_run=ref_run,
        candidate_runs=cand_runs,
        unavailable_engines=unavail,
    )


@pytest.mark.slow
@pytest.mark.requires_pinocchio
@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
@pytest.mark.parametrize("motion", ["hold", "swing"])
def test_real_style_parity_report_structure_all_engines(
    style: GripStyle, motion: str
) -> None:
    """Real 0.04 s report across Pinocchio, OpenSim, MuJoCo, and Drake classifies all 14 quantities and pair traces."""
    report = _cached_style_report(style, motion)
    assert report.style == style.value
    assert report.motion == motion
    assert set(report.rows) == {"pinocchio", "opensim", "mujoco", "drake"}
    topo = contact_topology(GripSpec(style=style))

    for eng, row in report.rows.items():
        if not row.available:
            assert row.unavailable_reason
            for q_name in STYLE_PARITY_QUANTITIES:
                q_res = row.quantity(q_name)
                assert q_res.state == ParityState.UNAVAILABLE
                assert math.isnan(q_res.peak_error)
                assert math.isnan(q_res.rms_error)
                assert q_res.passes() is False
            continue

        for q_name in DETERMINATE_QUANTITIES:
            q_res = row.quantity(q_name)
            assert q_res.classification == QuantityClassification.DETERMINATE
            assert math.isfinite(q_res.peak_error)
            assert math.isfinite(q_res.rms_error)
        for q_name in INDETERMINATE_QUANTITIES:
            q_res = row.quantity(q_name)
            assert q_res.classification == QuantityClassification.INDETERMINATE
            assert q_res.state == ParityState.INDETERMINATE
            assert q_res.passes() is False, f"{eng} {q_name} must never pass silently"
            assert q_res.label == INDETERMINATE_LABEL
            assert q_res.citation == INDETERMINACY_CITATION
            assert math.isfinite(q_res.peak_error)
            assert math.isfinite(q_res.rms_error)

        for pair in topo.hand_grip_pairs():
            for metric in PAIR_PARITY_QUANTITIES:
                pq = row.pair_quantity(pair.name, metric)
                assert pq.classification == QuantityClassification.INDETERMINATE
                assert pq.state == ParityState.INDETERMINATE
                assert pq.passes() is False
                assert pq.label == INDETERMINATE_LABEL
                assert math.isfinite(pq.peak_error)
                assert math.isfinite(pq.rms_error)
        for pair in topo.hand_hand_pairs():
            for metric in PAIR_PARITY_QUANTITIES:
                pq = row.pair_quantity(pair.name, metric)
                assert pq.state == ParityState.UNAVAILABLE
                assert pq.passes() is False
                assert math.isnan(pq.peak_error)
                assert math.isnan(pq.rms_error)
                assert "allocated" in pq.reason


def _net_moment_xfail(case: str, numbers: str) -> pytest.MarkDecorator:
    """Strict xfail for a net-moment gap exposed by the balanced t = 0 preload."""
    return pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=(
            f"{case} 0.04 s determinate net_moment_com {numbers} exceeds the "
            "5.0% peak / 2.0% RMS tolerances (#13560): since #13555 removed the "
            "unbalanced t = 0 preload the reference net moment is a 0.16-0.47 N m "
            "start-up transient instead of the 38 N m kick; the absolute error fell"
        ),
    )


_DETERMINATE_ENGINE_CASES = [
    # Drake (G9): passes all determinate quantities across all styles and motions
    pytest.param(
        "drake",
        GripStyle.OVERLAP,
        "hold",
        marks=[pytest.mark.requires_drake],
        id="drake-overlap-hold",
    ),
    pytest.param(
        "drake",
        GripStyle.INTERLOCK,
        "hold",
        marks=[pytest.mark.requires_drake],
        id="drake-interlock-hold",
    ),
    pytest.param(
        "drake",
        GripStyle.TEN_FINGER,
        "hold",
        marks=[pytest.mark.requires_drake],
        id="drake-ten_finger-hold",
    ),
    pytest.param(
        "drake",
        GripStyle.OVERLAP,
        "swing",
        marks=[pytest.mark.requires_drake],
        id="drake-overlap-swing",
    ),
    pytest.param(
        "drake",
        GripStyle.INTERLOCK,
        "swing",
        marks=[pytest.mark.requires_drake],
        id="drake-interlock-swing",
    ),
    pytest.param(
        "drake",
        GripStyle.TEN_FINGER,
        "swing",
        marks=[pytest.mark.requires_drake],
        id="drake-ten_finger-swing",
    ),
    # MuJoCo (G8): passes all determinate quantities across all styles and motions
    # (the per-row friction gain schedule closed the ten_finger hold gap, #13306)
    # except ten_finger hold net_moment_com since #13555 (#13560)
    pytest.param(
        "mujoco",
        GripStyle.OVERLAP,
        "hold",
        marks=[pytest.mark.requires_mujoco],
        id="mujoco-overlap-hold",
    ),
    pytest.param(
        "mujoco",
        GripStyle.INTERLOCK,
        "hold",
        marks=[pytest.mark.requires_mujoco],
        id="mujoco-interlock-hold",
    ),
    pytest.param(
        "mujoco",
        GripStyle.TEN_FINGER,
        "hold",
        marks=[
            pytest.mark.requires_mujoco,
            _net_moment_xfail("MuJoCo ten_finger hold", "peak 0.631% / RMS 2.266%"),
        ],
        id="mujoco-ten_finger-hold",
    ),
    pytest.param(
        "mujoco",
        GripStyle.OVERLAP,
        "swing",
        marks=[pytest.mark.requires_mujoco],
        id="mujoco-overlap-swing",
    ),
    pytest.param(
        "mujoco",
        GripStyle.INTERLOCK,
        "swing",
        marks=[pytest.mark.requires_mujoco],
        id="mujoco-interlock-swing",
    ),
    pytest.param(
        "mujoco",
        GripStyle.TEN_FINGER,
        "swing",
        marks=[pytest.mark.requires_mujoco],
        id="mujoco-ten_finger-swing",
    ),
    # OpenSim (G7/G7b): since #13555 passes ten_finger 0.04 s hold and passes
    # determinate net_force for ten_finger swing; ten_finger swing, overlap and
    # interlock exceed the tolerances on determinate net_moment_com (#13560).
    # Since #13308 the reference uses the same line-contact law as the
    # ElasticFoundationForce sphere chain, so the remaining gap is the friction
    # and patch-torque mapping (tau_n = 0, surface lever), not the capsule law.
    pytest.param(
        "opensim",
        GripStyle.OVERLAP,
        "hold",
        marks=[
            pytest.mark.requires_opensim,
            _net_moment_xfail("OpenSim overlap hold", "peak 5.451% / RMS 2.512%"),
        ],
        id="opensim-overlap-hold",
    ),
    pytest.param(
        "opensim",
        GripStyle.INTERLOCK,
        "hold",
        marks=[
            pytest.mark.requires_opensim,
            _net_moment_xfail("OpenSim interlock hold", "peak 5.451% / RMS 2.512%"),
        ],
        id="opensim-interlock-hold",
    ),
    pytest.param(
        "opensim",
        GripStyle.TEN_FINGER,
        "hold",
        marks=[pytest.mark.requires_opensim],
        id="opensim-ten_finger-hold",
    ),
    pytest.param(
        "opensim",
        GripStyle.OVERLAP,
        "swing",
        marks=[
            pytest.mark.requires_opensim,
            _net_moment_xfail("OpenSim overlap swing", "peak 0.391% / RMS 4.059%"),
        ],
        id="opensim-overlap-swing",
    ),
    pytest.param(
        "opensim",
        GripStyle.INTERLOCK,
        "swing",
        marks=[
            pytest.mark.requires_opensim,
            _net_moment_xfail("OpenSim interlock swing", "peak 0.391% / RMS 4.059%"),
        ],
        id="opensim-interlock-swing",
    ),
    pytest.param(
        "opensim",
        GripStyle.TEN_FINGER,
        "swing",
        marks=[
            pytest.mark.requires_opensim,
            _net_moment_xfail(
                "OpenSim ten_finger swing",
                "peak 2.066% / RMS 4.104% (19.5 mN m RMS against a 0.475 N m "
                "reference peak; net_force passes at 0.706% / 1.011%, #13548)",
            ),
        ],
        id="opensim-ten_finger-swing",
    ),
]


@pytest.mark.slow
@pytest.mark.requires_pinocchio
@pytest.mark.timeout(600)
@pytest.mark.parametrize(("engine", "style", "motion"), _DETERMINATE_ENGINE_CASES)
def test_real_determinate_parity_passes_per_engine(
    engine: str, style: GripStyle, motion: str
) -> None:
    """Every DETERMINATE quantity (net_force, net_moment_com) must pass for each available engine."""
    report = _cached_style_report(style, motion)
    row = report.row(engine)
    if not row.available:
        pytest.skip(f"{engine} unavailable: {row.unavailable_reason}")
    failing = row.failing_determinate()
    assert not failing, (
        f"Determinate parity failed for engine={engine} style={style.value} "
        f"motion={motion}: "
        + ", ".join(
            f"{k}: peak={v.peak_error * 100:.3f}% (tol {PEAK_TOLERANCE * 100:.1f}%), "
            f"rms={v.rms_error * 100:.3f}% (tol {RMS_TOLERANCE * 100:.1f}%)"
            for k, v in failing.items()
        )
    )
