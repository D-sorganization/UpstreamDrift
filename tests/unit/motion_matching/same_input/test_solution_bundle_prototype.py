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
        "schema_version": "research-solution-prototype/v1",
        "research_only": True,
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
        "recorded_q": np.array([[0.0], [1.0]]),
        "recorded_qd": np.array([[1.0], [1.0]]),
        "commanded_tau": np.array([[2.0]]),
        "applied_tau": np.array([[1.0]]),
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
        arrays["applied_tau"] = np.zeros((2, 1))
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

    tolerances = dict.fromkeys(("q", "qd", "commanded_tau", "applied_tau"), (0.0, 0.0))
    result = verify_forward_resimulation(
        b, forward, tolerances, runtime_sha256="c" * 64
    )
    assert result.passed
    assert not result.independently_verified
    assert result.runtime_identity_is_caller_asserted
    assert calls == [True]

    def drift(bundle):
        p = recorded_playback(bundle)
        return p._replace(applied_tau=p.applied_tau + 0.1)

    assert not verify_forward_resimulation(
        b, drift, tolerances, runtime_sha256="c" * 64
    ).passed
    with pytest.raises(ValueError, match="tolerance"):
        verify_forward_resimulation(
            b, forward, {"q": (0.0, 0.0)}, runtime_sha256="c" * 64
        )
    with pytest.raises(ValueError, match="runtime"):
        verify_forward_resimulation(b, forward, tolerances, runtime_sha256="d" * 64)
