"""Unit and parity tests for Simscape physical grip contact (HRG-GS-M12, #13334).

Covers:
1. Canonical Python -> MATLAB codegen kernel generator and ``--check`` mode.
2. Constitutive golden vectors (randomized + separated, just touching,
   near-collinear capsules, stick, slip, zero relative velocity, coincident
   axis, rebounding clamp) and ``test_gs3dx_grip_pair_wrench.m`` contract.
3. Scenario input exporter (``hold`` and ``swing`` across ``overlap``,
   ``interlock``, ``ten_finger``) and pure-NumPy vs Pinocchio FK equivalence.
4. Series loader DbC validation (NaN-with-reason, schema, R2025b receipt,
   conversion to ``TopologyKineticsSeries`` and ``GripKineticsSeries``).
5. Non-circular determinate net-club-wrench parity against Pinocchio G6 from
   committed R2025b fixtures under ``exploratory_gs3dx/receipts/grip_contact/``
   (skips with a clear reason until R2025b fixtures are committed; never passes
   vacuously).
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.grip_contact import (
    DETERMINATE_QUANTITIES,
    INDETERMINATE_QUANTITIES,
    PEAK_TOLERANCE,
    RMS_TOLERANCE,
    GripStyle,
    build_topology_model,
    classify_parity_quantity,
    compare_topology_series,
    contact_topology,
    load_coordinate_swing,
    parity_errors,
)
from src.shared.python.grip_contact.matlab_kernel import (
    DEFAULT_GOLDEN_JSON_PATH,
    DEFAULT_KERNEL_M_PATH,
    DEFAULT_PARAMS_JSON_PATH,
    GOLDEN_SCHEMA_VERSION,
    KERNEL_SCHEMA_VERSION,
    MAX_PHYSICAL_PAIRS,
    build_golden_vectors_payload,
    build_kernel_params_payload,
    evaluate_kernel_reference,
    export_matlab_kernel_bundle,
    render_matlab_kernel_source,
    validate_golden_vectors_payload,
    validate_kernel_params_payload,
)
from src.shared.python.grip_contact.simscape_grip_contact import (
    DEFAULT_RECEIPT_DIR,
    DEFAULT_SCENARIOS_JSON_PATH,
    RECEIPT_SCHEMA_VERSION,
    SCENARIOS_SCHEMA_VERSION,
    SERIES_SCHEMA_VERSION,
    build_scenarios_payload,
    export_scenarios_json,
    load_simscape_grip_contact_receipt,
    load_simscape_grip_contact_series,
    simscape_series_to_kinetics_series,
    simscape_series_to_topology_series,
    spec_weld_club_state,
    synthesize_reference_simscape_series,
    validate_scenarios_payload,
    validate_simscape_grip_contact_receipt,
    validate_simscape_grip_contact_series,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
GS3DX_ROOT = (
    REPO_ROOT
    / "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/exploratory_gs3dx"
)
SPEC_PATH = (
    REPO_ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
)
SWING_NPZ = REPO_ROOT / "tests/fixtures/club_face/swing_q_driver.npz"
POSES_JSON = REPO_ROOT / "tests/fixtures/club_face/address_poses.json"


def _has_pinocchio() -> bool:
    try:
        import pinocchio  # noqa: F401
    except ImportError:
        return False
    # The CI unit lane may mock pinocchio; only a real module has a Model class.
    if type(pinocchio).__module__ == "unittest.mock":
        return False
    model_cls = getattr(pinocchio, "Model", None)
    return isinstance(model_cls, type) and model_cls.__module__ != "unittest.mock"


requires_pinocchio = pytest.mark.skipif(
    not _has_pinocchio(), reason="pinocchio not installed"
)


# ---------------------------------------------------------------------------
# 1. Canonical kernel -> MATLAB & golden vectors
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_kernel_params_payload_and_matlab_source_contract() -> None:
    """Generated MATLAB source and params JSON obey %#codegen and DRY rules."""
    params = build_kernel_params_payload()
    validate_kernel_params_payload(params)
    assert params["schema_version"] == KERNEL_SCHEMA_VERSION
    assert params["max_physical_pairs"] == MAX_PHYSICAL_PAIRS
    assert set(params["styles"]) == {"overlap", "interlock", "ten_finger"}

    src = render_matlab_kernel_source(params)
    assert "%#codegen" in src
    assert (
        "function [F_club_w, M_club_w, F_hand_L_w, M_hand_L_w, F_hand_R_w, M_hand_R_w,"
        in src
    )
    assert (
        "pair_moment_club_w, pair_contact_point_w] = gs3dx_grip_contact_kernel(" in src
    )
    assert "z_state" not in src
    assert "dz_dt" not in src
    # Codegen prohibitions: no persistent state, no containers.Map, no eval/feval.
    for forbidden in ("persistent ", "containers.Map", "eval(", "feval("):
        assert forbidden not in src

    # Every law constant in params must appear verbatim in the generated MATLAB.
    law = params["law"]
    assert f"{law['normal_stiffness_n_m']:.17g}" in src
    assert f"{law['static_friction']:.17g}" in src
    assert f"{law['dynamic_friction']:.17g}" in src
    assert f"{law['patch_radius_m']:.17g}" in src


@pytest.mark.unit
def test_export_matlab_kernel_bundle_check_and_drift_detection(tmp_path: Path) -> None:
    """Committed generated files are fresh and --check catches any mutation."""
    assert export_matlab_kernel_bundle(check=True) == 0

    m_path = tmp_path / "gs3dx_grip_contact_kernel.m"
    p_path = tmp_path / "gs3dx_grip_contact_params.json"
    g_path = tmp_path / "gs3dx_grip_pair_golden.json"
    assert (
        export_matlab_kernel_bundle(
            kernel_m_path=m_path,
            params_json_path=p_path,
            golden_json_path=g_path,
            check=False,
        )
        == 0
    )
    assert (
        export_matlab_kernel_bundle(
            kernel_m_path=m_path,
            params_json_path=p_path,
            golden_json_path=g_path,
            check=True,
        )
        == 0
    )

    # Mutate generated .m -> --check must fail with return code 1.
    m_path.write_text(
        m_path.read_text(encoding="utf-8") + "\n% drift\n", encoding="utf-8"
    )
    assert (
        export_matlab_kernel_bundle(
            kernel_m_path=m_path,
            params_json_path=p_path,
            golden_json_path=g_path,
            check=True,
        )
        == 1
    )


@pytest.mark.unit
def test_golden_vectors_cover_all_styles_and_edge_cases() -> None:
    """Golden vectors cover all styles, required edge cases, and match pair_contact."""
    assert DEFAULT_GOLDEN_JSON_PATH.exists()
    payload = json.loads(DEFAULT_GOLDEN_JSON_PATH.read_text(encoding="utf-8"))
    validate_golden_vectors_payload(payload)
    assert payload["schema_version"] == GOLDEN_SCHEMA_VERSION

    cases = payload["cases"]
    assert len(cases) >= 36

    styles_seen = {c["style"] for c in cases}
    assert styles_seen == {"overlap", "interlock", "ten_finger"}

    categories = {c["category"] for c in cases}
    required_categories = {
        "separated",
        "just_touching",
        "near_collinear_capsules",
        "stick",
        "slip",
        "zero_relative_velocity",
        "coincident_axis",
        "rebounding_clamp",
        "randomized",
    }
    assert required_categories.issubset(categories)

    params = build_kernel_params_payload()
    for case in cases:
        inp = case["inputs"]
        exp = case["expected"]
        assert "z_state" not in inp
        assert "dz_dt" not in exp
        out = evaluate_kernel_reference(params, case["style"], inp)
        assert "dz_dt" not in out
        np.testing.assert_allclose(
            out["f_club_w_n"], exp["f_club_w_n"], rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            out["m_club_w_nm"], exp["m_club_w_nm"], rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            out["f_hand_l_w_n"], exp["f_hand_l_w_n"], rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            out["f_hand_r_w_n"], exp["f_hand_r_w_n"], rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            out["pair_normal_n"], exp["pair_normal_n"], rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            out["pair_tangent_n"], exp["pair_tangent_n"], rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            out["pair_spin_nm"], exp["pair_spin_nm"], rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            out["pair_slip_speed_m_s"],
            exp["pair_slip_speed_m_s"],
            rtol=1e-12,
            atol=1e-12,
        )
        # Newton's third law: club net force equals minus sum of hand net forces.
        np.testing.assert_allclose(
            out["f_club_w_n"],
            -(out["f_hand_l_w_n"] + out["f_hand_r_w_n"]),
            rtol=1e-12,
            atol=1e-12,
        )


@pytest.mark.unit
def test_kernel_capsule_pairs_use_shared_line_contact_quadrature() -> None:
    """Capsule rows integrate along the hand capsule like ``capsule_capsule`` (#13308).

    The closest-point law snapped the little-finger contact ~8 mm once the
    finger tilted past ``sin^2(theta) = 1e-10``; the line quadrature keeps the
    contact point and club force continuous across that tilt.
    """
    from src.shared.python.grip_contact.pair_contact import line_contact_node_count

    params = build_kernel_params_payload()
    tbl = np.asarray(params["styles"]["ten_finger"]["pair_table"], dtype=float)
    caps = [k for k, row in enumerate(tbl) if row[0] > 0.0 and row[2] == 2.0]
    assert caps, "ten_finger must carry at least one capsule pair"
    for k in caps:
        row = tbl[k]
        half_a = 0.5 * float(np.linalg.norm(row[6:9] - row[3:6]))
        assert row[17] == line_contact_node_count(half_a, float(row[9]))

    golden = json.loads(DEFAULT_GOLDEN_JSON_PATH.read_text(encoding="utf-8"))
    (case,) = [
        c
        for c in golden["cases"]
        if c["style"] == "ten_finger" and c["category"] == "near_collinear_capsules"
    ]
    k = caps[0]
    points, forces = [], []
    for angle in (0.0, 5.0e-6, 2.0e-5, 1.0e-4):
        c, s = np.cos(angle), np.sin(angle)
        tilt = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        inp = dict(case["inputs"])
        inp["r_hand_r"] = (np.asarray(inp["r_hand_r"]) @ tilt).tolist()
        out = evaluate_kernel_reference(params, "ten_finger", inp)
        assert out["pair_normal_n"][k] > 0.0
        points.append(out["pair_contact_point_w_m"][k])
        forces.append(out["f_club_w_n"])
    jumps = np.linalg.norm(np.diff(np.asarray(points), axis=0), axis=1)
    assert jumps.max() < 1.0e-4, jumps
    f = np.asarray(forces)
    assert np.linalg.norm(f[1] - f[0]) < 1.0e-3 * np.linalg.norm(f[0])


@pytest.mark.unit
def test_matlab_pair_wrench_unittest_file_contract() -> None:
    """MATLAB unittest file checks R2025b and 1e-10 rel / 1e-9 N abs tolerances."""
    test_m = GS3DX_ROOT / "tests/test_gs3dx_grip_pair_wrench.m"
    assert test_m.exists()
    content = test_m.read_text(encoding="utf-8")
    assert "matlab.unittest.TestCase" in content
    assert "2025b" in content
    assert "1e-10" in content
    assert "1e-9" in content
    assert "gs3dx_grip_pair_golden.json" in content
    assert "gs3dx_grip_contact_kernel" in content
    assert "z_state" not in content
    assert "dz_dt" not in content


# ---------------------------------------------------------------------------
# 2. Scenario inputs exporter & pure-NumPy FK parity
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_export_scenarios_json_check_and_structure(tmp_path: Path) -> None:
    """Scenario input JSON is fresh and matches Pinocchio/MuJoCo G6 parameters."""
    assert export_scenarios_json(DEFAULT_SCENARIOS_JSON_PATH, check=True) == 0
    payload = json.loads(DEFAULT_SCENARIOS_JSON_PATH.read_text(encoding="utf-8"))
    validate_scenarios_payload(payload)
    assert payload["schema_version"] == SCENARIOS_SCHEMA_VERSION
    assert payload["squeeze_per_hand_n"] == pytest.approx(1100.0)
    assert set(payload["styles"]) == {"overlap", "interlock", "ten_finger"}
    assert set(payload["scenarios"]) == {"hold", "short_swing"}

    for style in ("overlap", "interlock", "ten_finger"):
        style_meta = payload["styles"][style]
        assert style_meta["topology_mapping_hash"].startswith("sha256:")

    out_tmp = tmp_path / "scenarios.json"
    assert export_scenarios_json(out_tmp, check=False) == 0
    assert export_scenarios_json(out_tmp, check=True) == 0
    drifted = json.loads(out_tmp.read_text(encoding="utf-8"))
    drifted["squeeze_per_hand_n"] = 1101.0
    out_tmp.write_text(json.dumps(drifted), encoding="utf-8")
    assert export_scenarios_json(out_tmp, check=True) == 1


@requires_pinocchio
@pytest.mark.unit
def test_pure_numpy_weld_club_kinematics_matches_pinocchio() -> None:
    """Pure-NumPy spec_weld_club_state matches Pinocchio WeldClubKinematics to 1e-14."""
    from src.engines.physics_engines.pinocchio.python.grip_bushing import (
        WeldClubKinematics,
    )

    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    names = list(spec["coordinate_order"])
    swing = load_coordinate_swing(SWING_NPZ, POSES_JSON, "driver", names)
    kin = WeldClubKinematics(spec, names)

    for idx in (0, 5, 15, 30, 50):
        q = swing.q[idx]
        qdot = (
            (swing.q[idx + 1] - swing.q[idx]) / 0.002
            if idx + 1 < len(swing.q)
            else np.zeros_like(q)
        )
        pin_state = kin.state(q, qdot)
        np_state = spec_weld_club_state(spec, names, q, qdot)
        np.testing.assert_allclose(
            np_state.rotation, pin_state.rotation, rtol=1e-14, atol=1e-14
        )
        np.testing.assert_allclose(
            np_state.position_m, pin_state.position_m, rtol=1e-14, atol=1e-14
        )
        np.testing.assert_allclose(
            np_state.velocity_m_s, pin_state.velocity_m_s, rtol=1e-14, atol=1e-14
        )
        np.testing.assert_allclose(
            np_state.omega_rad_s, pin_state.omega_rad_s, rtol=1e-14, atol=1e-14
        )


# ---------------------------------------------------------------------------
# 3. Python series loader & DbC validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_simscape_series_loader_dbc_and_nan_reason_contract(tmp_path: Path) -> None:
    """Series loader enforces DbC preconditions and NaN-with-reason rules."""
    sample = synthesize_reference_simscape_series(
        style="overlap", scenario="hold", n_samples=6
    )
    assert "z_state_m" not in sample
    validate_simscape_grip_contact_series(sample)

    series_path = tmp_path / "series.json"
    series_path.write_text(json.dumps(sample), encoding="utf-8")
    loaded = load_simscape_grip_contact_series(series_path)
    assert not hasattr(loaded, "z_state_m")
    topo_series = simscape_series_to_topology_series(loaded)
    kin_series = simscape_series_to_kinetics_series(loaded)

    assert topo_series.engine == "simscape_contact"
    assert topo_series.model_id == "GS3DX_GripContact"
    assert topo_series.time_s.shape == (6,)
    assert kin_series.force_on_club_n["L"].shape == (6, 3)
    assert kin_series.force_on_club_n["R"].shape == (6, 3)

    # Indeterminate vs determinate classification per §19.
    assert classify_parity_quantity("net_force").value == "determinate"
    assert classify_parity_quantity("net_moment_com").value == "determinate"
    assert classify_parity_quantity("force_L").value == "indeterminate"
    assert classify_parity_quantity("pair_normal_force").value == "indeterminate"
    assert "net_force" in DETERMINATE_QUANTITIES
    assert "net_moment_com" in DETERMINATE_QUANTITIES
    assert "force_L" in INDETERMINATE_QUANTITIES
    assert "pair_normal_force" in INDETERMINATE_QUANTITIES

    # Wrong release raises ValueError.
    bad_rel = copy.deepcopy(sample)
    bad_rel["matlab_release"] = "2026a"
    with pytest.raises(ValueError, match="2025b"):
        validate_simscape_grip_contact_series(bad_rel)

    # NaN without reason raises ValueError.
    bad_nan = copy.deepcopy(sample)
    bad_nan["net_club_force_w_n"][0][0] = float("nan")
    with pytest.raises(ValueError, match="unavailable_reasons"):
        validate_simscape_grip_contact_series(bad_nan)

    # NaN WITH non-empty reason is accepted by DbC contract.
    ok_nan = copy.deepcopy(bad_nan)
    ok_nan["unavailable_reasons"]["net_club_force_w_n"] = (
        "probe sample masked for NaN-reason unit test"
    )
    validate_simscape_grip_contact_series(ok_nan)


@pytest.mark.unit
def test_simscape_receipt_validation_contract() -> None:
    """Receipt validator checks release, hashes, convergence, save/reload, probes, budget."""
    valid_receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "matlab_release": "2025b",
        "model_name": "GS3DX_GripContact",
        "model_sha256": "a" * 64,
        "input_sha256": "b" * 64,
        "kernel_sha256": "c" * 64,
        "solver": {
            "solver_name": "ode15s",
            "max_step_s": 5.0e-4,
            "rel_tol": 1.0e-5,
            "abs_tol": 1.0e-7,
        },
        "block_budget": {
            "uncompiled_total": 72,
            "compiled_total": 96,
            "license_ceiling": 1000,
        },
        "timestep_convergence": {
            "baseline_max_step_s": 5.0e-4,
            "refined_max_step_s": 2.5e-4,
            "max_rel_force_diff": 1.2e-4,
            "passed": True,
        },
        "save_reload_check": {
            "max_abs_force_diff_n": 0.0,
            "max_abs_moment_diff_nm": 0.0,
            "passed": True,
        },
        "frame_probes": {
            "max_pos_err_m": 0.0,
            "max_rot_err_fro": 0.0,
            "passed": True,
        },
        "force_sign_checks": {
            "min_normal_push_dot_n": 120.0,
            "passed": True,
        },
    }
    validate_simscape_grip_contact_receipt(valid_receipt)

    bad_budget = copy.deepcopy(valid_receipt)
    bad_budget["block_budget"]["compiled_total"] = 1005
    with pytest.raises(ValueError, match="1000"):
        validate_simscape_grip_contact_receipt(bad_budget)

    bad_sign = copy.deepcopy(valid_receipt)
    bad_sign["force_sign_checks"]["passed"] = False
    with pytest.raises(ValueError, match="force_sign_checks"):
        validate_simscape_grip_contact_receipt(bad_sign)


# ---------------------------------------------------------------------------
# 4. MATLAB builder, runner & deferred catalog contracts
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_matlab_builder_runner_and_deferred_catalog_contracts() -> None:
    """Verify gs3dx_names, builder, runner, and deferred-catalog-bundle entries."""
    names_m = (GS3DX_ROOT / "tools/gs3dx_names.m").read_text(encoding="utf-8")
    assert "'grip_contact'" in names_m
    assert '"GS3DX_GripContact"' in names_m

    builder_m = GS3DX_ROOT / "tools/gs3dx_build_grip_contact.m"
    runner_m = GS3DX_ROOT / "tools/gs3dx_grip_contact_run.m"
    assert builder_m.exists()
    assert runner_m.exists()

    builder_src = builder_m.read_text(encoding="utf-8")
    assert "2025b" in builder_src
    assert "gs3dx_block_budget" in builder_src
    assert "External Force and Torque" in builder_src
    assert "Transform Sensor" in builder_src
    assert "Integrator" not in builder_src
    assert "add_block('sm_lib/Joints/Weld Joint'" not in builder_src

    runner_src = runner_m.read_text(encoding="utf-8")
    assert "2025b" in runner_src
    assert "timestep_convergence" in runner_src
    assert "save_reload_check" in runner_src
    assert "frame_probes" in runner_src
    assert "force_sign_checks" in runner_src

    catalog_path = REPO_ROOT / "docs/development/deferred-catalog-bundle.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    items = catalog.get("deferred_items", [])
    m12_items = [item for item in items if item.get("id") == "HRG-GS-M12"]
    assert len(m12_items) == 1
    assert m12_items[0]["issue"] == 13334
    assert m12_items[0]["epic_issue"] == 13297
    assert m12_items[0]["status"] == "deferred"
    assert "C:/Program Files/MATLAB/R2025b/bin/matlab.exe" in m12_items[0]["executable"]


# ---------------------------------------------------------------------------
# 5. Committed R2025b fixture parity test against Pinocchio reference
# ---------------------------------------------------------------------------


@pytest.mark.unit
@requires_pinocchio
@pytest.mark.parametrize("style", ["overlap", "interlock", "ten_finger"])
@pytest.mark.parametrize("scenario", ["hold", "short_swing"])
def test_simscape_r2025b_net_wrench_parity_against_pinocchio(
    style: str, scenario: str
) -> None:
    """Compare committed R2025b Simscape series against independent Pinocchio G6 reference.

    Skips cleanly until the R2025b execution fixtures exist under
    ``exploratory_gs3dx/receipts/grip_contact/``; never passes vacuously.
    """
    receipt_path = DEFAULT_RECEIPT_DIR / "gs3dx_grip_contact_receipt.json"
    series_path = DEFAULT_RECEIPT_DIR / f"gs3dx_grip_contact_{style}_{scenario}.json"
    if not receipt_path.exists() or not series_path.exists():
        pytest.skip(
            f"R2025b execution fixture not yet committed at {series_path.relative_to(REPO_ROOT)}; "
            "run gs3dx_grip_contact_run(repo_root, out_dir) on MATLAB R2025b."
        )

    receipt = load_simscape_grip_contact_receipt(receipt_path)
    validate_simscape_grip_contact_receipt(receipt)

    raw_series = load_simscape_grip_contact_series(series_path)
    sim_topo = simscape_series_to_topology_series(raw_series)
    from src.shared.python.grip_contact import (
        ClubDynamics,
        CoordinateSwing,
        GripInterface,
        simscape_series_to_contact_run,
    )
    from src.shared.python.grip_contact.net_wrench_parity import (
        failing_net_wrench,
        net_club_wrench_errors,
    )
    from src.shared.python.motion_matching.grip_contract import GripSpec

    # Independent Pinocchio G6 reference (non-circular).
    from src.engines.physics_engines.pinocchio.python.grip_contact_sim import (
        simulate_grip_contact,
    )

    spec_bytes = SPEC_PATH.read_bytes()
    spec = json.loads(spec_bytes)
    names = list(spec["coordinate_order"])
    swing = load_coordinate_swing(SWING_NPZ, POSES_JSON, "driver", names)
    interface = GripInterface.from_spec(spec)
    club = ClubDynamics.from_spec(spec)
    topo = contact_topology(GripSpec(style=GripStyle(style)))
    topo_model = build_topology_model(interface, topo, squeeze_per_hand_n=1100.0)
    duration = float(sim_topo.time_s[-1])
    if scenario == "hold":
        n_hold = int(round(duration / 0.002)) + 1
        run_swing = CoordinateSwing(
            swing.names,
            np.arange(n_hold) * 0.002,
            np.repeat(swing.q[:1], n_hold, axis=0),
            swing.sha256,
        )
    else:
        run_swing = swing

    pin_run = simulate_grip_contact(
        spec_bytes,
        run_swing,
        interface=interface,
        t_end_s=duration,
        hold=(scenario == "hold"),
        topology_model=topo_model,
    )
    sim_run = simscape_series_to_contact_run(raw_series, interface=interface)

    # Section 19: only the net club wrench is determinate; the per-hand split
    # and squeeze are not acceptance quantities.
    errors = net_club_wrench_errors(sim_run.series, pin_run.series, club, interface)
    assert not failing_net_wrench(errors), failing_net_wrench(errors)

    # Negative control: a 10 % error in the Simscape club force must fail.
    forces = sim_run.series.force_on_club_n
    scaled = dataclasses.replace(
        sim_run.series, force_on_club_n={k: 1.1 * v for k, v in forces.items()}
    )
    bad = net_club_wrench_errors(scaled, pin_run.series, club, interface)
    assert "net_force" in failing_net_wrench(bad)


@pytest.mark.unit
def test_r2025b_receipt_hashes_match_committed_kernel_and_inputs() -> None:
    """The committed R2025b evidence was produced from the committed kernel and inputs.

    A regenerated kernel (for example after a shared-law change) or changed
    scenario inputs make this fail until the R2025b run is repeated.
    """
    receipt_path = DEFAULT_RECEIPT_DIR / "gs3dx_grip_contact_receipt.json"
    if not receipt_path.exists():
        pytest.skip("R2025b receipt not committed yet")
    receipt = load_simscape_grip_contact_receipt(receipt_path)
    generated = GS3DX_ROOT / "tools/generated"
    for key, name in (
        ("kernel_sha256", "gs3dx_grip_contact_kernel.m"),
        ("input_sha256", "gs3dx_grip_contact_scenarios.json"),
    ):
        digest = hashlib.sha256((generated / name).read_bytes()).hexdigest()
        assert receipt[key] == digest, f"{name} changed since the R2025b run"
