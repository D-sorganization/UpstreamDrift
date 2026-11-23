"""Strict prototype bundles and separate playback/forward comparison."""

from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from src.shared.python.motion_matching.same_input.solution_bundle_prototype import (
    SolutionBundle,
    load_solution,
    recorded_playback,
    save_solution,
    verify_forward_resimulation,
)

pytestmark = pytest.mark.unit


def _fixture():
    model = b'{"synthetic":true}'
    manifest = {
        "schema_version": "research-solution-prototype/v4",
        "research_only": True,
        "runner": {"mode": "unavailable"},
        "solution_id": "synthetic",
        "parent_id": None,
        "git_sha": "a" * 40,
        "model_sha256": hashlib.sha256(model).hexdigest(),
        "asset_sha256": {},
        "conventions": {
            "units": "SI",
            "frame": "world",
            "quaternion": "none",
            "joint_order": ["joint"],
        },
        "controller": {
            "kind": "fixed_pd_bspline",
            "hold": "zoh",
            "holds_per_step": 1,
            "saturation": "declared",
            "internal_state": {},
        },
        "integrator": {
            "kind": "test",
            "runtime": "synthetic",
            "runtime_sha256": "c" * 64,
        },
        "environment": {},
        "impact": {"status": "none"},
        "aero": {},
        "optimizer": {"seed": 7},
        "flight_metrics": {"status": "unavailable"},
        "feasibility": {"status": "unavailable"},
        "assumptions": ["synthetic example"],
    }
    arrays = {
        "reference_time_s": np.array([0.0, 1.0]),
        "q_ref": np.array([[0.0], [1.0]]),
        "qd_ref": np.array([[1.0], [1.0]]),
        "tau_ff_ref": np.zeros((2, 1)),
        "spline_knots_s": np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
        "spline_coefficients": np.zeros((4, 1)),
        "kp": np.array([1.0]),
        "kd": np.array([1.0]),
        "q0": np.array([0.0]),
        "qd0": np.array([1.0]),
        "recorded_time_s": np.array([0.0, 1.0]),
        "recorded_hold_time_s": np.array([0.0]),
        "recorded_q": np.array([[0.0], [1.0]]),
        "recorded_qd": np.array([[1.0], [1.0]]),
        "commanded_tau": np.array([[2.0]]),
        "ideal_input_tau": np.array([[1.0]]),
        "saturation_lower": np.array([-10.0]),
        "saturation_upper": np.array([10.0]),
        "spline_scale": np.array([1.0]),
    }
    return SolutionBundle(manifest, arrays, model)


def test_round_trip_model_identity_and_independent_recorded_playback(tmp_path):
    b = _fixture()
    manifest_hash = save_solution(tmp_path / "solution.json", b)
    result = load_solution(
        tmp_path / "solution.json",
        expected_model_sha256=b.manifest["model_sha256"],
        expected_runtime_sha256="c" * 64,
        expected_manifest_sha256=manifest_hash,
    )
    assert recorded_playback(result).q.tolist() == [[0.0], [1.0]]
    with pytest.raises(ValueError, match="model"):
        load_solution(
            tmp_path / "solution.json",
            expected_model_sha256="b" * 64,
            expected_runtime_sha256="c" * 64,
            expected_manifest_sha256=manifest_hash,
        )
    with pytest.raises(ValueError, match="runtime"):
        load_solution(
            tmp_path / "solution.json",
            expected_model_sha256=b.manifest["model_sha256"],
            expected_runtime_sha256="d" * 64,
            expected_manifest_sha256=manifest_hash,
        )


@pytest.mark.parametrize(
    "bad", ["nonfinite", "time", "shape", "version", "asset", "state", "metadata"]
)
def test_rejects_incompatible_or_incomplete_evidence(bad):
    b = _fixture()
    manifest, arrays = deepcopy(b.manifest), {k: v.copy() for k, v in b.arrays.items()}
    if bad == "nonfinite":
        arrays["q_ref"][0, 0] = np.nan
    if bad == "time":
        arrays["reference_time_s"][:] = 0
    if bad == "shape":
        arrays["ideal_input_tau"] = np.zeros((2, 1))
    if bad == "version":
        manifest["schema_version"] = "9.0"
    if bad == "asset":
        manifest["asset_sha256"] = {"asset": "not-a-hash"}
    if bad == "state":
        arrays["recorded_q"][0, 0] = 1
    if bad == "metadata":
        del manifest["integrator"]
    with pytest.raises(ValueError):
        SolutionBundle(manifest, arrays, b.model_bytes)


def test_tampered_npz_is_rejected(tmp_path):
    b = _fixture()
    manifest_hash = save_solution(tmp_path / "solution.json", b)
    path = next(tmp_path.glob("solution.*.npz"))
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash"):
        load_solution(
            tmp_path / "solution.json",
            expected_model_sha256=b.manifest["model_sha256"],
            expected_runtime_sha256="c" * 64,
            expected_manifest_sha256=manifest_hash,
        )


def test_asset_hashes_must_match_the_callers_runtime_inventory(tmp_path):
    b = _fixture()
    manifest = deepcopy(b.manifest)
    manifest["asset_sha256"] = {"declared-asset": "d" * 64}
    b = SolutionBundle(manifest, b.arrays, b.model_bytes)
    path = tmp_path / "solution.json"
    digest = save_solution(path, b)
    kwargs = {
        "expected_manifest_sha256": digest,
        "expected_model_sha256": manifest["model_sha256"],
        "expected_runtime_sha256": "c" * 64,
    }
    with pytest.raises(ValueError, match="assets"):
        load_solution(path, **kwargs)
    assert (
        load_solution(
            path, expected_assets={"declared-asset": "d" * 64}, **kwargs
        ).manifest["asset_sha256"]
        == manifest["asset_sha256"]
    )


def test_manifest_tamper_rejected_before_reading_untrusted_path(tmp_path):
    b = _fixture()
    path = tmp_path / "solution.json"
    manifest_hash = save_solution(path, b)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["npz_file"] = "../../outside.npz"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest hash"):
        load_solution(
            path,
            expected_manifest_sha256=manifest_hash,
            expected_model_sha256=b.manifest["model_sha256"],
            expected_runtime_sha256="c" * 64,
        )
    changed_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="manifest fields"):
        load_solution(
            path,
            expected_manifest_sha256=changed_hash,
            expected_model_sha256=b.manifest["model_sha256"],
            expected_runtime_sha256="c" * 64,
        )


def test_failure_to_publish_manifest_never_exposes_a_partial_bundle(
    tmp_path, monkeypatch
):
    import os

    b = _fixture()

    def fail(*args):
        raise OSError("synthetic publication failure")

    monkeypatch.setattr(os, "link", fail)
    with pytest.raises(OSError, match="publication"):
        save_solution(tmp_path / "solution.json", b)
    assert not (tmp_path / "solution.json").exists()


def test_duplicate_manifest_keys_rejected_even_with_matching_digest(tmp_path):
    b = _fixture()
    path = tmp_path / "solution.json"
    save_solution(path, b)
    raw = path.read_bytes().replace(
        b'"schema_version":', b'"schema_version": "fake", "schema_version":', 1
    )
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="duplicate"):
        load_solution(
            path,
            expected_manifest_sha256=hashlib.sha256(raw).hexdigest(),
            expected_model_sha256=b.manifest["model_sha256"],
            expected_runtime_sha256="c" * 64,
        )


def test_duplicate_npz_members_rejected_even_with_updated_digests(tmp_path):
    import io
    import warnings
    import zipfile

    b = _fixture()
    path = tmp_path / "solution.json"
    save_solution(path, b)
    record = json.loads(path.read_text(encoding="utf-8"))
    original = next(tmp_path.glob("solution.*.npz")).read_bytes()
    archive = io.BytesIO(original)
    with warnings.catch_warnings(), zipfile.ZipFile(archive, "a") as zf:
        warnings.simplefilter("ignore", UserWarning)
        zf.writestr("q_ref.npy", zf.read("q_ref.npy"))
    raw = archive.getvalue()
    record["npz_sha256"] = hashlib.sha256(raw).hexdigest()
    (tmp_path / f"solution.{record['npz_sha256']}.npz").write_bytes(raw)
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        load_solution(
            path,
            expected_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            expected_model_sha256=b.manifest["model_sha256"],
            expected_runtime_sha256="c" * 64,
        )


def test_overflowing_time_differences_fail_closed():
    b = _fixture()
    arrays = {k: v.copy() for k, v in b.arrays.items()}
    arrays["reference_time_s"] = np.array([-1e308, 1e308])
    with pytest.raises(ValueError, match="time"):
        SolutionBundle(b.manifest, arrays, b.model_bytes)


def test_real_impact_evidence_is_not_accepted_by_prototype():
    b = _fixture()
    manifest = deepcopy(b.manifest)
    manifest["impact"] = {"status": "detected"}
    with pytest.raises(ValueError, match="impact"):
        SolutionBundle(manifest, b.arrays, b.model_bytes)


def test_grid_aligned_recorded_force_event_is_preserved_but_not_certified(tmp_path):
    b = _fixture()
    manifest = deepcopy(b.manifest)
    manifest["impact"] = {
        "status": "recorded_latched_force",
        "time_s": 0.0,
        "duration_s": 0.1,
        "frame": "face",
        "point_in_frame_m": [0.0, 0.0, 0.0],
        "axis_in_frame": [1.0, 0.0, 0.0],
        "force_world_n": [-1.0, 0.0, 0.0],
        "event_q": [0.0],
        "event_qd": [1.0],
        "contact_verified": False,
    }
    bundle = SolutionBundle(manifest, b.arrays, b.model_bytes)
    path = tmp_path / "impact.json"
    digest = save_solution(path, bundle)
    loaded = load_solution(
        path,
        expected_manifest_sha256=digest,
        expected_model_sha256=manifest["model_sha256"],
        expected_runtime_sha256="c" * 64,
    )
    assert loaded.manifest["impact"] == manifest["impact"]


@pytest.mark.parametrize(
    "bad",
    [
        "near_edge",
        "state",
        "outside",
        "verified",
        "extra",
        "nonfinite",
        "tiny_duration",
    ],
)
def test_recorded_force_event_rejects_inconsistent_evidence(bad):
    b = _fixture()
    manifest = deepcopy(b.manifest)
    event = {
        "status": "recorded_latched_force",
        "time_s": 0.0,
        "duration_s": 0.1,
        "frame": "face",
        "point_in_frame_m": [0.0, 0.0, 0.0],
        "axis_in_frame": [1.0, 0.0, 0.0],
        "force_world_n": [-1.0, 0.0, 0.0],
        "event_q": [0.0],
        "event_qd": [1.0],
        "contact_verified": False,
    }
    if bad == "near_edge":
        event["time_s"] = 5e-13
    if bad == "state":
        event["event_q"] = [0.01]
    if bad == "outside":
        event["duration_s"] = 1.01
    if bad == "verified":
        event["contact_verified"] = True
    if bad == "extra":
        event["undocumented"] = 1
    if bad == "nonfinite":
        event["force_world_n"] = [float("nan"), 0, 0]
    if bad == "tiny_duration":
        event["time_s"], event["duration_s"] = 0.5, 5e-13
    manifest["impact"] = event
    with pytest.raises(ValueError):
        SolutionBundle(manifest, b.arrays, b.model_bytes)


def test_off_grid_force_event_is_allowed_without_claiming_grid_interpolation():
    b = _fixture()
    manifest = deepcopy(b.manifest)
    manifest["impact"] = {
        "status": "recorded_latched_force",
        "time_s": 0.5,
        "duration_s": 0.1,
        "frame": "face",
        "point_in_frame_m": [0.0, 0.0, 0.0],
        "axis_in_frame": [1.0, 0.0, 0.0],
        "force_world_n": [-1.0, 0.0, 0.0],
        "event_q": [0.123],
        "event_qd": [0.987],
        "contact_verified": False,
    }
    assert (
        SolutionBundle(manifest, b.arrays, b.model_bytes).manifest["impact"]["time_s"]
        == 0.5
    )


def test_unavailable_runner_and_old_research_schema_fail_closed():
    from src.shared.python.motion_matching.same_input.solution_bundle_prototype import (
        run_mujoco_ideal,
    )

    b = _fixture()
    with pytest.raises(ValueError, match="runner"):
        run_mujoco_ideal(b)
    old = deepcopy(b.manifest)
    old["schema_version"] = "research-solution-prototype/v2"
    with pytest.raises(ValueError, match="migration"):
        SolutionBundle(old, b.arrays, b.model_bytes)


def test_explicit_v2_migration_preserves_single_hold_values():
    from src.shared.python.motion_matching.same_input.solution_bundle_prototype import (
        migrate_research_v2,
    )

    b = _fixture()
    old, arrays = deepcopy(b.manifest), {k: v.copy() for k, v in b.arrays.items()}
    old["schema_version"] = "research-solution-prototype/v2"
    del old["controller"]["holds_per_step"]
    del arrays["recorded_hold_time_s"]
    migrated = migrate_research_v2(old, arrays, b.model_bytes)
    assert migrated.manifest["controller"]["holds_per_step"] == 1
    np.testing.assert_array_equal(migrated.arrays["recorded_hold_time_s"], [0.0])
    np.testing.assert_array_equal(
        migrated.arrays["ideal_input_tau"], arrays["ideal_input_tau"]
    )
    with pytest.raises(ValueError, match="v2"):
        migrate_research_v2(b.manifest, arrays, b.model_bytes)


def test_v3_requires_explicit_no_impact_migration():
    from src.shared.python.motion_matching.same_input.solution_bundle_prototype import (
        migrate_research_v3,
    )

    b = _fixture()
    old = deepcopy(b.manifest)
    old["schema_version"] = "research-solution-prototype/v3"
    with pytest.raises(ValueError, match="migration"):
        SolutionBundle(old, b.arrays, b.model_bytes)
    migrated = migrate_research_v3(old, b.arrays, b.model_bytes)
    assert migrated.manifest["impact"] == {"status": "none"}
    assert migrated.manifest["schema_version"].endswith("/v4")
    old["impact"] = {"status": "detected"}
    with pytest.raises(ValueError, match="impact"):
        migrate_research_v3(old, b.arrays, b.model_bytes)


def test_real_mujoco_ideal_forward_replay_uses_actual_integrator(tmp_path):
    pytest.importorskip("mujoco")
    from pathlib import Path
    from src.shared.python.motion_matching.same_input.integrator import ROOT_COORDINATES
    from src.shared.python.motion_matching.same_input.solution_bundle_mujoco_replay import (
        measured_runtime_fingerprint,
    )
    from src.shared.python.motion_matching.same_input.solution_bundle_prototype import (
        run_mujoco_ideal,
    )

    model = (
        Path(__file__).resolve().parents[4]
        / "docs/development/full_body_models/full_body_spec_anthro_driver_v2.json"
    ).read_bytes()
    names = json.loads(model)["coordinate_order"]
    n, dt = len(names), 0.0001
    times = np.arange(3, dtype=float) * dt
    root = [names.index(name) for name in ROOT_COORDINATES]
    lower, upper = np.full(n, -10.0), np.full(n, 10.0)
    lower[root] = upper[root] = 0.0
    seed = _fixture()
    manifest = deepcopy(seed.manifest)
    manifest["model_sha256"] = hashlib.sha256(model).hexdigest()
    manifest["conventions"]["joint_order"] = names
    manifest["controller"]["saturation"] = "ideal_clip_total"
    manifest["integrator"] = {
        "kind": "rk4_zoh",
        "runtime": "partial_measured_v1",
        "runtime_sha256": measured_runtime_fingerprint(),
    }
    manifest["runner"] = {
        "mode": "mujoco_ideal_zoh",
        "dt_s": dt,
        "substeps": 2,
        "project": False,
        "trust_radius": 1.0,
        "min_spacing_s": dt,
    }
    arrays = {
        "reference_time_s": times,
        "q_ref": np.zeros((3, n)),
        "qd_ref": np.zeros((3, n)),
        "tau_ff_ref": np.zeros((3, n)),
        "spline_knots_s": np.r_[np.repeat(times[0], 4), np.repeat(times[-1], 4)],
        "spline_coefficients": np.zeros((4, n)),
        "kp": np.zeros(n),
        "kd": np.zeros(n),
        "q0": np.zeros(n),
        "qd0": np.zeros(n),
        "recorded_time_s": times,
        "recorded_hold_time_s": times[:-1],
        "recorded_q": np.zeros((3, n)),
        "recorded_qd": np.zeros((3, n)),
        "commanded_tau": np.zeros((2, n)),
        "ideal_input_tau": np.zeros((2, n)),
        "saturation_lower": lower,
        "saturation_upper": upper,
        "spline_scale": np.ones(n),
    }
    active = next(i for i in range(n) if i not in root)
    arrays["tau_ff_ref"][:, active] = 2.0
    arrays["saturation_upper"][active] = 1.0
    initial = SolutionBundle(manifest, arrays, model)
    actual = run_mujoco_ideal(initial)
    assert np.any(actual.q[1:] != 0.0)  # Real dynamics, not recorded-state playback.
    assert np.all(actual.commanded_tau[:, active] == 2.0)
    assert np.all(actual.ideal_input_tau[:, active] == 1.0)
    arrays.update(
        recorded_q=actual.q,
        recorded_qd=actual.qd,
        commanded_tau=actual.commanded_tau,
        ideal_input_tau=actual.ideal_input_tau,
    )
    bundle = SolutionBundle(manifest, arrays, model)
    path = tmp_path / "real.json"
    manifest_hash = save_solution(path, bundle)
    loaded = load_solution(
        path,
        expected_manifest_sha256=manifest_hash,
        expected_model_sha256=manifest["model_sha256"],
        expected_runtime_sha256=manifest["integrator"]["runtime_sha256"],
    )
    replay = run_mujoco_ideal(loaded)
    for key in ("q", "qd", "commanded_tau", "ideal_input_tau"):
        np.testing.assert_allclose(
            getattr(replay, key), getattr(actual, key), rtol=1e-12, atol=1e-12
        )
    checked = verify_forward_resimulation(
        loaded,
        run_mujoco_ideal,
        dict.fromkeys(("q", "qd", "commanded_tau", "ideal_input_tau"), (1e-12, 1e-12)),
        runtime_sha256=manifest["integrator"]["runtime_sha256"],
    )
    assert checked.passed and not checked.independently_verified
    altered = {k: v.copy() for k, v in arrays.items()}
    altered["recorded_q"][-1, active] += 0.01
    mismatch = SolutionBundle(manifest, altered, model)
    assert not verify_forward_resimulation(
        mismatch,
        run_mujoco_ideal,
        dict.fromkeys(("q", "qd", "commanded_tau", "ideal_input_tau"), (1e-12, 1e-12)),
        runtime_sha256=manifest["integrator"]["runtime_sha256"],
    ).passed

    wrong_runtime = deepcopy(manifest)
    wrong_runtime["integrator"]["runtime_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="runtime fingerprint"):
        run_mujoco_ideal(SolutionBundle(wrong_runtime, arrays, model))
    unsupported = deepcopy(manifest)
    unsupported["controller"]["internal_state"] = {"memory": [1.0]}
    with pytest.raises(ValueError, match="state"):
        run_mujoco_ideal(SolutionBundle(unsupported, arrays, model))
    unsupported = deepcopy(manifest)
    unsupported["environment"] = {"wind": [1, 0, 0]}
    with pytest.raises(ValueError, match="environment"):
        run_mujoco_ideal(SolutionBundle(unsupported, arrays, model))
    wrong_grid = {k: v.copy() for k, v in arrays.items()}
    wrong_grid["recorded_time_s"][1] += dt / 10
    with pytest.raises(ValueError, match="time grid"):
        run_mujoco_ideal(SolutionBundle(manifest, wrong_grid, model))
    wrong_root = {k: v.copy() for k, v in arrays.items()}
    wrong_root["saturation_lower"][root[0]] = -1.0
    with pytest.raises(ValueError, match="root"):
        run_mujoco_ideal(SolutionBundle(manifest, wrong_root, model))
    unsupported = deepcopy(manifest)
    unsupported["integrator"]["adaptive"] = True
    with pytest.raises(ValueError, match="integrator"):
        run_mujoco_ideal(SolutionBundle(unsupported, arrays, model))
    wrong_model_doc = json.loads(model)
    wrong_model_doc["coordinate_order"] = []
    wrong_model = json.dumps(wrong_model_doc).encode()
    wrong_manifest = deepcopy(manifest)
    wrong_manifest["model_sha256"] = hashlib.sha256(wrong_model).hexdigest()
    with pytest.raises(ValueError, match="coordinate inventory"):
        run_mujoco_ideal(SolutionBundle(wrong_manifest, arrays, wrong_model))


def test_off_grid_recorded_force_replays_actual_impact_state(tmp_path):
    pytest.importorskip("mujoco")
    from pathlib import Path
    from src.shared.python.motion_matching import club_face_target as face
    from src.shared.python.motion_matching.same_input.integrator import ROOT_COORDINATES
    from src.shared.python.motion_matching.same_input.solution_bundle_mujoco_replay import (
        measured_runtime_fingerprint,
        run_mujoco_ideal_with_trace,
    )
    from src.shared.python.motion_matching.same_input.solution_bundle_prototype import (
        run_mujoco_ideal,
    )

    model = (
        Path(__file__).resolve().parents[4]
        / "docs/development/full_body_models/full_body_spec_anthro_driver_v2.json"
    ).read_bytes()
    spec = json.loads(model)
    names, dt = spec["coordinate_order"], 0.0001
    n = len(names)
    root = [names.index(name) for name in ROOT_COORDINATES]
    times = np.arange(3, dtype=float) * dt
    lower, upper = np.full(n, -10.0), np.full(n, 10.0)
    lower[root] = upper[root] = 0.0
    seed = _fixture()
    manifest = deepcopy(seed.manifest)
    manifest["model_sha256"] = hashlib.sha256(model).hexdigest()
    manifest["conventions"]["joint_order"] = names
    manifest["controller"].update(saturation="ideal_clip_total", holds_per_step=2)
    manifest["integrator"] = {
        "kind": "rk4_zoh",
        "runtime": "partial_measured_v1",
        "runtime_sha256": measured_runtime_fingerprint(),
    }
    manifest["runner"] = {
        "mode": "mujoco_ideal_zoh",
        "dt_s": dt,
        "substeps": 4,
        "project": False,
        "trust_radius": 1.0,
        "min_spacing_s": dt,
    }
    manifest["impact"] = {
        "status": "recorded_latched_force",
        "time_s": dt / 4,
        "duration_s": dt,
        "frame": face.FACE_FRAME,
        "point_in_frame_m": face.face_centre_in_frame(spec).tolist(),
        "axis_in_frame": face.face_axis_in_frame(spec).tolist(),
        "force_world_n": [-10.0, 0.0, 0.0],
        "event_q": np.zeros(n).tolist(),
        "event_qd": np.zeros(n).tolist(),
        "contact_verified": False,
    }
    arrays = {
        "reference_time_s": times,
        "q_ref": np.zeros((3, n)),
        "qd_ref": np.zeros((3, n)),
        "tau_ff_ref": np.zeros((3, n)),
        "spline_knots_s": np.r_[np.repeat(times[0], 4), np.repeat(times[-1], 4)],
        "spline_coefficients": np.zeros((4, n)),
        "kp": np.zeros(n),
        "kd": np.zeros(n),
        "q0": np.zeros(n),
        "qd0": np.zeros(n),
        "recorded_time_s": times,
        "recorded_hold_time_s": np.arange(4, dtype=float) * (dt / 2),
        "recorded_q": np.zeros((3, n)),
        "recorded_qd": np.zeros((3, n)),
        "commanded_tau": np.zeros((4, n)),
        "ideal_input_tau": np.zeros((4, n)),
        "saturation_lower": lower,
        "saturation_upper": upper,
        "spline_scale": np.ones(n),
    }
    actual = run_mujoco_ideal(SolutionBundle(manifest, arrays, model))
    traced, recorder = run_mujoco_ideal_with_trace(
        SolutionBundle(manifest, arrays, model)
    )
    np.testing.assert_array_equal(actual.q, traced.q)
    np.testing.assert_array_equal(actual.qd, traced.qd)
    assert recorder.outcome.status == "complete"
    assert any(
        np.linalg.norm(stage.applied_external) > 0
        for step in recorder.steps
        for stage in step.stages
    )
    assert all(
        stage.effort_semantics == "ideal_direct_generalized_no_saturation"
        for step in recorder.steps
        for stage in step.stages
    )
    assert actual.impact_event is not None and actual.impact_event[0] == dt / 4
    assert np.any(
        actual.impact_event[2] != 0.0
    )  # Real pre-force integration, not grid interpolation.
    stronger = deepcopy(manifest)
    stronger["impact"]["force_world_n"] = [-20.0, 0.0, 0.0]
    strong_replay = run_mujoco_ideal(SolutionBundle(stronger, arrays, model))
    np.testing.assert_array_equal(strong_replay.impact_event[1], actual.impact_event[1])
    np.testing.assert_array_equal(strong_replay.impact_event[2], actual.impact_event[2])
    assert np.max(np.abs(strong_replay.q[-1] - actual.q[-1])) > 1e-12
    arrays.update(
        recorded_q=actual.q,
        recorded_qd=actual.qd,
        commanded_tau=actual.commanded_tau,
        ideal_input_tau=actual.ideal_input_tau,
    )
    manifest["impact"]["event_q"] = actual.impact_event[1].tolist()
    manifest["impact"]["event_qd"] = actual.impact_event[2].tolist()
    bundle = SolutionBundle(manifest, arrays, model)
    path = tmp_path / "impact.json"
    digest = save_solution(path, bundle)
    loaded = load_solution(
        path,
        expected_manifest_sha256=digest,
        expected_model_sha256=manifest["model_sha256"],
        expected_runtime_sha256=manifest["integrator"]["runtime_sha256"],
    )
    tolerances = dict.fromkeys(
        ("q", "qd", "commanded_tau", "ideal_input_tau"), (1e-12, 1e-12)
    )
    report = verify_forward_resimulation(
        loaded,
        run_mujoco_ideal,
        tolerances,
        runtime_sha256=manifest["integrator"]["runtime_sha256"],
    )
    assert report.passed and not report.independently_verified
    assert report.max_absolute_error["impact_q"] <= 1e-12
    altered = deepcopy(manifest)
    altered["impact"]["event_q"][0] += 0.01
    bad = SolutionBundle(altered, arrays, model)
    assert not verify_forward_resimulation(
        bad,
        run_mujoco_ideal,
        tolerances,
        runtime_sha256=manifest["integrator"]["runtime_sha256"],
    ).passed


def test_multiple_holds_preserve_control_clock_and_replay(tmp_path):
    pytest.importorskip("mujoco")
    from pathlib import Path
    from src.shared.python.motion_matching.same_input.integrator import ROOT_COORDINATES
    from src.shared.python.motion_matching.same_input.solution_bundle_mujoco_replay import (
        measured_runtime_fingerprint,
    )
    from src.shared.python.motion_matching.same_input.solution_bundle_prototype import (
        run_mujoco_ideal,
    )

    model = (
        Path(__file__).resolve().parents[4]
        / "docs/development/full_body_models/full_body_spec_anthro_driver_v2.json"
    ).read_bytes()
    names = json.loads(model)["coordinate_order"]
    n, dt = len(names), 0.0002
    root = [names.index(name) for name in ROOT_COORDINATES]
    active = next(i for i in range(n) if i not in root)
    b = _fixture()
    manifest = deepcopy(b.manifest)
    manifest["model_sha256"] = hashlib.sha256(model).hexdigest()
    manifest["conventions"]["joint_order"] = names
    manifest["controller"].update(saturation="ideal_clip_total", holds_per_step=2)
    manifest["integrator"] = {
        "kind": "rk4_zoh",
        "runtime": "partial_measured_v1",
        "runtime_sha256": measured_runtime_fingerprint(),
    }
    manifest["runner"] = {
        "mode": "mujoco_ideal_zoh",
        "dt_s": dt,
        "substeps": 4,
        "project": False,
        "trust_radius": 1.0,
        "min_spacing_s": dt,
    }
    t = np.arange(3) * dt
    lower, upper = np.full(n, -10.0), np.full(n, 10.0)
    lower[root] = upper[root] = 0.0
    ff = np.zeros((3, n))
    ff[:, active] = [0.0, 2.0, 4.0]
    arrays = {
        "reference_time_s": t,
        "q_ref": np.zeros((3, n)),
        "qd_ref": np.zeros((3, n)),
        "tau_ff_ref": ff,
        "spline_knots_s": np.r_[np.repeat(t[0], 4), np.repeat(t[-1], 4)],
        "spline_coefficients": np.zeros((4, n)),
        "kp": np.zeros(n),
        "kd": np.zeros(n),
        "q0": np.zeros(n),
        "qd0": np.zeros(n),
        "recorded_time_s": t,
        "recorded_hold_time_s": np.arange(4) * (dt / 2),
        "recorded_q": np.zeros((3, n)),
        "recorded_qd": np.zeros((3, n)),
        "commanded_tau": np.zeros((4, n)),
        "ideal_input_tau": np.zeros((4, n)),
        "saturation_lower": lower,
        "saturation_upper": upper,
        "spline_scale": np.ones(n),
    }
    initial = SolutionBundle(manifest, arrays, model)
    actual = run_mujoco_ideal(initial)
    np.testing.assert_array_equal(actual.hold_time_s, arrays["recorded_hold_time_s"])
    np.testing.assert_allclose(
        actual.commanded_tau[:, active], [0, 1, 2, 3], atol=1e-12
    )
    arrays.update(
        recorded_q=actual.q,
        recorded_qd=actual.qd,
        commanded_tau=actual.commanded_tau,
        ideal_input_tau=actual.ideal_input_tau,
    )
    bundle = SolutionBundle(manifest, arrays, model)
    digest = save_solution(tmp_path / "multi.json", bundle)
    restored = load_solution(
        tmp_path / "multi.json",
        expected_manifest_sha256=digest,
        expected_model_sha256=manifest["model_sha256"],
        expected_runtime_sha256=manifest["integrator"]["runtime_sha256"],
    )
    result = verify_forward_resimulation(
        restored,
        run_mujoco_ideal,
        dict.fromkeys(("q", "qd", "commanded_tau", "ideal_input_tau"), (1e-12, 1e-12)),
        runtime_sha256=manifest["integrator"]["runtime_sha256"],
    )
    assert result.passed and not result.independently_verified
    invalid_policy = deepcopy(manifest)
    invalid_policy["runner"]["substeps"] = 3
    with pytest.raises(ValueError, match="policy"):
        run_mujoco_ideal(SolutionBundle(invalid_policy, arrays, model))


@pytest.mark.parametrize(
    "bad", ["count", "clock", "offset", "noninteger", "fraction", "huge"]
)
def test_hold_schedule_fails_closed(bad):
    b = _fixture()
    manifest = deepcopy(b.manifest)
    arrays = {k: v.copy() for k, v in b.arrays.items()}
    if bad == "count":
        manifest["controller"]["holds_per_step"] = 2
    elif bad == "clock":
        arrays["recorded_hold_time_s"][0] = 0.01
    elif bad == "offset":
        arrays["recorded_time_s"] += 0.1
        arrays["recorded_hold_time_s"] += 0.1
    elif bad == "noninteger":
        manifest["controller"]["holds_per_step"] = True
    elif bad == "fraction":
        manifest["controller"]["holds_per_step"] = 1.5
    else:
        manifest["controller"]["holds_per_step"] = 10**100
    with pytest.raises(ValueError, match="hold|controller"):
        SolutionBundle(manifest, arrays, b.model_bytes)


def test_forward_comparison_rejects_wrong_hold_clock():
    b = _fixture()

    def wrong_clock(bundle):
        playback = recorded_playback(bundle)
        return playback._replace(hold_time_s=playback.hold_time_s + 0.01)

    with pytest.raises(ValueError, match="time grid"):
        verify_forward_resimulation(
            b,
            wrong_clock,
            dict.fromkeys(("q", "qd", "commanded_tau", "ideal_input_tau"), (0.0, 0.0)),
            runtime_sha256="c" * 64,
        )


@pytest.mark.parametrize("field", ["feasibility", "flight_metrics"])
def test_prototype_cannot_claim_results_it_does_not_verify(field):
    b = _fixture()
    manifest = deepcopy(b.manifest)
    manifest[field] = {"status": "pass"}
    with pytest.raises(ValueError, match="unavailable"):
        SolutionBundle(manifest, b.arrays, b.model_bytes)


def test_rejects_malformed_spline_configuration():
    b = _fixture()
    arrays = {k: v.copy() for k, v in b.arrays.items()}
    arrays["spline_knots_s"] = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.4, 0.2, 1.0, 1.0, 1.0, 1.0]
    )
    arrays["spline_coefficients"] = np.zeros((6, 1))
    with pytest.raises(ValueError, match="spline"):
        SolutionBundle(b.manifest, arrays, b.model_bytes)


def test_forward_resimulation_is_distinct_and_checks_each_quantity():
    b = _fixture()
    calls = []

    def forward(bundle):
        calls.append(True)
        p = recorded_playback(bundle)
        return p

    tolerances = dict.fromkeys(
        ("q", "qd", "commanded_tau", "ideal_input_tau"), (0.0, 0.0)
    )
    result = verify_forward_resimulation(
        b, forward, tolerances, runtime_sha256="c" * 64
    )
    assert result.passed
    assert not result.independently_verified
    assert result.runtime_identity_is_caller_asserted
    assert calls == [True]

    def drift(bundle):
        p = recorded_playback(bundle)
        return p._replace(ideal_input_tau=p.ideal_input_tau + 0.1)

    assert not verify_forward_resimulation(
        b, drift, tolerances, runtime_sha256="c" * 64
    ).passed
    with pytest.raises(ValueError, match="tolerance"):
        verify_forward_resimulation(
            b, forward, {"q": (0.0, 0.0)}, runtime_sha256="c" * 64
        )
    with pytest.raises(ValueError, match="runtime"):
        verify_forward_resimulation(b, forward, tolerances, runtime_sha256="d" * 64)


@pytest.mark.parametrize("overflow", ["difference", "tolerance"])
def test_forward_comparison_rejects_nonfinite_derived_arithmetic(overflow):
    b = _fixture()
    arrays = {k: v.copy() for k, v in b.arrays.items()}
    arrays["recorded_q"][-1, 0] = 1e308
    b = SolutionBundle(b.manifest, arrays, b.model_bytes)

    def runner(bundle):
        playback = recorded_playback(bundle)
        if overflow == "difference":
            playback.q[-1, 0] = -1e308
        return playback

    tolerances = dict.fromkeys(
        ("q", "qd", "commanded_tau", "ideal_input_tau"), (0.0, 0.0)
    )
    if overflow == "tolerance":
        tolerances["q"] = (0.0, 2.0)
    with pytest.raises(ValueError, match="nonfinite"):
        verify_forward_resimulation(b, runner, tolerances, runtime_sha256="c" * 64)
