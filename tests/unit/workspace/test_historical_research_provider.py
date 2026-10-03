"""SDK-free sanitized research export contracts; no scientific promotion."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.shared.python.workspace.historical_research import (
    ResearchAuditPin,
    build_historical_research_package,
    export_historical_research_package,
)

pytestmark = pytest.mark.unit
COMMIT = "a" * 40


def digest(payload):
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def save_json(path, value):
    path.write_bytes(json.dumps(value, sort_keys=True).encode())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fit_inputs(tmp_path):
    source = tmp_path / "original.mp4"
    source.write_bytes(b"original downloaded bytes, not rendered")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    model = tmp_path / "model.xml"
    model.write_bytes(b"<mujoco/>")
    capture = tmp_path / "capture.zip"
    capture.write_bytes(b"bound capture archive")
    frames = [
        {
            "pts_ticks": n,
            "timebase_numerator": 1,
            "timebase_denominator": 30,
            "physical_time_s": None,
            "is_timing_exact": True,
            "schema_version": "shadow-tracker/frame/1.1.0",
            "asset_id": "source-" + source_hash,
            "shot_id": "shot",
            "swing_id": "swing",
            "camera_id": "camera",
            "frame_id": f"frame-{n}",
            "physical_time_reason": "Unknown historical playback scale",
            "frame_sha256": "f" * 64,
            "timing_mode": "container_pts",
            "clock_evidence": "Original container",
            "decoder_name": "pyav",
            "decoder_version": "19.0.0",
            "pixel_format": "bgr24",
        }
        for n in [30, 31]
    ]
    original = {
        "rms_pixels": 2.0,
        "dense_rms_pixels": 3.0,
        "held_out_rms_pixels": 4.0,
        "converged": False,
        "optimizer_ran": True,
        "frame_indices": [0, 1],
    }
    stamp = {"source_commit": COMMIT, "source_sha256": "sha256:" + "c" * 64}
    fit_record = {
        "model_id": "model",
        "capture_id": "capture",
        "model_hash": digest(model.read_bytes()),
        "capture_hash": digest(capture.read_bytes()),
        "qualification": "monocular_research_hypothesis",
        "physical_time_qualified": False,
        "dynamics_replayed": False,
        "frames": frames,
        "frame_indices": [0, 1],
        "provenance": {
            "execution_stamp": stamp,
            "request_options": {
                "unknown_visibility_weight": 0.5,
                "config": {"bound": "retained"},
            },
        },
        "evidence": {"original_fit": original},
    }
    fit = tmp_path / "fit.json"
    save_json(fit, fit_record)
    return source, source_hash, model, capture, fit, fit_record, stamp


def _audit_records(tmp_path, source, source_hash, model, capture, fit, stamp):
    audit = {
        "schema": "necromatcher/independent-bounded-trial-audit/1",
        "fit_id": "fit",
        "fit_hash": digest(fit.read_bytes()),
        "model_id": "model",
        "model_hash": digest(model.read_bytes()),
        "capture_id": "capture",
        "capture_hash": digest(capture.read_bytes()),
        "source_video_identity": {"content_sha256": source_hash},
        "producer_execution_stamp": stamp,
        "audit_source_runtime_unchanged": True,
        "observed_original_rms_pixels": 2.0,
        "dense_original_rms_pixels": 3.0,
        "maximum_sampled_grip_gap_m": 0.001,
        "maximum_sampled_grip_rotation_rad": 0.01,
        "maximum_sampled_ground_penetration_m": 0.0002,
        "source_frame_count": 2,
        "dense_observed_point_count": 26,
        "added_image_objective_points": 0,
        "optimization_performed": False,
        "saved_optimizer_ran": True,
        "saved_optimizer_converged": False,
        "scientific_acceptance": False,
        "physical_time_qualified": False,
        "continuous_nonlinear_certified": False,
    }
    audit_path = tmp_path / "audit.json"
    audit_hash = save_json(audit_path, audit)
    restart = {
        "training_rms_pixels": 2.0,
        "held_out_rms_pixels": 4.0,
        "training_frame_count": 2,
        "dense_frame_count": 2,
        "training_observed_point_count": 26,
        "dense_observed_point_count": 26,
        "unknown_visibility_weight": 0.5,
        "initial_spline_six_fields_exact": True,
    }
    summary = {
        "schema": "necromatcher/geometry-weight-v14-independent-assessment/1",
        "producer": COMMIT,
        "both_runs_verified": True,
        "source_runtime_parent_artifact_evidence_brackets_verified": True,
        "optimization_performed": False,
        "scientific_acceptance": False,
        "continuous_certified": False,
        "physical_time_qualified": False,
        "runs": [
            {
                "fit_id": "fit",
                "fit_sha256": audit["fit_hash"],
                "metrics": {
                    k: v
                    for k, v in audit.items()
                    if k
                    not in [
                        "schema",
                        "source_video_identity",
                        "producer_execution_stamp",
                        "audit_source_runtime_unchanged",
                    ]
                },
                "fresh_full_audit": {
                    "path": str(audit_path),
                    "sha256": "sha256:" + audit_hash,
                },
                "fresh_strict_restart": restart,
                "optimizer_converged": False,
                "targets": {
                    "checks": {
                        "dense_rms_nonregression": True,
                        "sampled_nonpenetration": True,
                    },
                    "scientific_acceptance": False,
                },
            }
        ],
    }
    summary_path = tmp_path / "assessment.json"
    summary_hash = save_json(summary_path, summary)
    pin = ResearchAuditPin(
        "fit", summary_path, summary_hash, audit_path, audit_hash, source
    )
    return pin, summary, audit


@pytest.fixture
def research_case(tmp_path):
    source, source_hash, model, capture, fit, fit_record, stamp = _fit_inputs(tmp_path)
    assets = {}
    for name, path, kind in [
        ("fit", fit, "kinematic_fit"),
        ("model", model, "native_model"),
        ("capture", capture, "image_capture"),
    ]:
        assets[name] = SimpleNamespace(
            dataset_id=name,
            session_id="swing",
            kind=kind,
            path=str(path),
            metadata={
                "hash": digest(path.read_bytes()),
                "source_sha256": source_hash,
                "frame_count": 2,
            },
        )

    class Library:
        def load_asset(self, name):
            asset = assets[name]
            if digest(Path(asset.path).read_bytes()) != asset.metadata["hash"]:
                raise ValueError("asset hash mismatch")
            return asset

        def load_fit(self, name):
            return json.loads(Path(self.load_asset(name).path).read_bytes())

        def players(self):
            return [
                SimpleNamespace(subject_id="third-player", display_name="Third Player")
            ]

        def swings(self, player_id=None):
            return [SimpleNamespace(session_id="swing", subject_id="third-player")]

    pin, summary, audit = _audit_records(
        tmp_path, source, source_hash, model, capture, fit, stamp
    )
    return Library(), pin, summary, audit, fit_record


def test_generic_package_has_exact_pins_and_separate_statuses(research_case):
    library, pin, _, _, _ = research_case
    payload = build_historical_research_package(library, (pin,), "b" * 40)
    value = json.loads(payload)
    record = value["records"][0]
    assert record["player_name"] == "Third Player"
    assert record["scientific_acceptance"] == "rejected"
    assert all(target["passed"] for target in record["targets"])
    assert value["provider"]["source_commit"] == "b" * 40
    assert record["producer_commit"] == COMMIT
    assert record["metrics"]["grip_gap_mm"] == 1.0
    assert record["source_clock"]["start"] == {"numerator": 1, "denominator": 1}
    assert record["source_clock"]["end"] == {"numerator": 31, "denominator": 30}
    assert str(pin.assessment_path).encode() not in payload
    assert b"original downloaded bytes" not in payload
    assert build_historical_research_package(library, (pin,), "b" * 40) == payload


@pytest.mark.parametrize(
    "which", ["assessment", "audit", "source", "fit", "model", "capture"]
)
def test_changed_bound_input_refuses_export(research_case, which):
    library, pin, _, _, _ = research_case
    paths = {
        "assessment": pin.assessment_path,
        "audit": pin.full_audit_path,
        "source": pin.source_video_path,
    }
    path = paths.get(which)
    if path is None:
        path = Path(library.load_asset(which).path)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        build_historical_research_package(library, (pin,), COMMIT)


@pytest.mark.parametrize(
    "field,value",
    [
        ("fit_hash", "sha256:" + "0" * 64),
        ("scientific_acceptance", True),
        ("saved_optimizer_converged", True),
        ("observed_original_rms_pixels", 42.0),
        ("dense_original_rms_pixels", float("nan")),
    ],
)
def test_rehashed_audit_cannot_contradict_canonical_evidence(
    research_case, field, value
):
    library, pin, summary, audit, _ = research_case
    audit[field] = value
    audit_sha = save_json(pin.full_audit_path, audit)
    summary["runs"][0]["metrics"][field] = value
    summary["runs"][0]["fresh_full_audit"]["sha256"] = "sha256:" + audit_sha
    summary_sha = save_json(pin.assessment_path, summary)
    pin = replace(pin, full_audit_sha256=audit_sha, assessment_sha256=summary_sha)
    with pytest.raises(ValueError):
        build_historical_research_package(library, (pin,), COMMIT)


def test_export_never_overwrites_existing_output(research_case, tmp_path):
    library, pin, _, _, _ = research_case
    target = tmp_path / "research.json"
    receipt = export_historical_research_package(library, (pin,), COMMIT, target)
    assert receipt["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert receipt["bytes"] == len(target.read_bytes())
    with pytest.raises(FileExistsError):
        export_historical_research_package(library, (pin,), COMMIT, target)
    assert receipt["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "field,value",
    [
        ("player_name", "C:/Users/private/secret"),
        ("source_time", True),
        ("training_count", 99),
        ("image_points", 1),
    ],
)
def test_research_export_rejects_unsafe_or_contradictory_metadata(
    research_case, field, value
):
    library, pin, summary, audit, fit = research_case
    if field == "player_name":
        library.players = lambda: [
            SimpleNamespace(subject_id="third-player", display_name=value)
        ]
    elif field == "source_time":
        fit["frames"][0]["physical_time_s"] = 1.0
        asset = library.load_asset("fit")
        save_json(Path(asset.path), fit)
        asset.metadata["hash"] = digest(Path(asset.path).read_bytes())
        audit["fit_hash"] = asset.metadata["hash"]
        summary["runs"][0]["fit_sha256"] = asset.metadata["hash"]
        summary["runs"][0]["metrics"]["fit_hash"] = asset.metadata["hash"]
    elif field == "training_count":
        summary["runs"][0]["fresh_strict_restart"]["training_frame_count"] = value
    else:
        audit["added_image_objective_points"] = value
        summary["runs"][0]["metrics"]["added_image_objective_points"] = value
    audit_sha = save_json(pin.full_audit_path, audit)
    summary["runs"][0]["fresh_full_audit"]["sha256"] = "sha256:" + audit_sha
    summary_sha = save_json(pin.assessment_path, summary)
    pin = replace(pin, full_audit_sha256=audit_sha, assessment_sha256=summary_sha)
    with pytest.raises(ValueError):
        build_historical_research_package(library, (pin,), COMMIT)


@pytest.mark.parametrize("schema_unavailable", [False, True])
def test_schema_copy_and_cold_public_import_are_sdk_free(schema_unavailable):
    import os
    import subprocess
    import sys
    from src.shared.python.workspace.historical_research import (
        historical_research_schema_bytes,
    )

    assert hashlib.sha256(historical_research_schema_bytes()).hexdigest() == (
        "a194bf0c8145e417262f460c9bc4ea817b73d491b8468be7dfcd17a88334cb30"
    )
    root = Path(__file__).resolve().parents[3]
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            map(str, [root, root / "src", root / "vendor/ud-tools"])
        ),
    )
    script = """
import importlib.abc, sys
class BlockSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mujoco', 'PyQt6', 'pydrake'} or (schema_unavailable and fullname.split('.')[0] == 'jsonschema'):
            raise ImportError('Scientific export has no SDK/UI dependency')
sys.meta_path.insert(0, BlockSDK())
from src.shared.python.workspace import ResearchAuditPin, build_historical_research_package
from src.shared.python.workspace.historical_research import historical_research_schema_bytes
assert historical_research_schema_bytes()
assert not any(n.split('.')[0] in {'mujoco','PyQt6','pydrake'} for n in sys.modules)
"""
    script = f"schema_unavailable = {schema_unavailable!r}\n" + script
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr


def test_v12_assessment_adapter_preserves_same_generic_boundary(research_case):
    library, pin, summary, _, _ = research_case
    summary["schema"] = "necromatcher/controlled-v12-independent-assessment/1"
    summary["all_four_verified"] = summary.pop("both_runs_verified")
    summary["source_runtime_driver_artifact_reference_brackets_verified"] = summary.pop(
        "source_runtime_parent_artifact_evidence_brackets_verified"
    )
    run = summary["runs"][0]
    run["initial_restart_and_images"] = run.pop("fresh_strict_restart")
    run["targets"] = {"fixed_historical_V10_targets": run["targets"]}
    pin = replace(pin, assessment_sha256=save_json(pin.assessment_path, summary))
    record = json.loads(build_historical_research_package(library, (pin,), COMMIT))[
        "records"
    ][0]
    assert record["image_counts"]["training_observation_count"] == 26
    assert record["scientific_acceptance"] == "rejected"


@pytest.mark.parametrize(
    "change", ["unknown_protocol", "unverified", "foreign_fit", "duplicate_fit"]
)
def test_assessment_protocol_and_selected_run_must_be_unambiguous(
    research_case, change
):
    library, pin, summary, _, _ = research_case
    if change == "unknown_protocol":
        summary["schema"] = "unknown/1"
    elif change == "unverified":
        summary["both_runs_verified"] = False
    elif change == "foreign_fit":
        summary["runs"][0]["fit_id"] = "other-player-fit"
    else:
        summary["runs"].append(summary["runs"][0])
    pin = replace(pin, assessment_sha256=save_json(pin.assessment_path, summary))
    with pytest.raises(ValueError):
        build_historical_research_package(library, (pin,), COMMIT)


@pytest.mark.parametrize("change", ["empty", "duplicate", "source_commit", "rights"])
def test_public_export_preconditions_fail_closed(research_case, change):
    library, pin, _, _, _ = research_case
    pins, commit = (pin,), COMMIT
    if change == "empty":
        pins = ()
    elif change == "duplicate":
        pins = (pin, pin)
    elif change == "source_commit":
        commit = "main"
    else:
        with pytest.raises(ValueError):
            replace(pin, rights_status="public_distribution_authorized")
        return
    with pytest.raises(ValueError):
        build_historical_research_package(library, pins, commit)


@pytest.mark.parametrize("malformed", [None, 17, {"sha256": "not-a-digest"}])
def test_rehashed_external_digest_requires_a_string_identity(research_case, malformed):
    library, pin, summary, audit, _ = research_case
    audit["fit_hash"] = malformed
    audit_sha = save_json(pin.full_audit_path, audit)
    summary["runs"][0]["metrics"]["fit_hash"] = malformed
    summary["runs"][0]["fit_sha256"] = malformed
    summary["runs"][0]["fresh_full_audit"]["sha256"] = "sha256:" + audit_sha
    summary_sha = save_json(pin.assessment_path, summary)
    pin = replace(pin, full_audit_sha256=audit_sha, assessment_sha256=summary_sha)
    with pytest.raises(ValueError, match="SHA-256 identity must be a string"):
        build_historical_research_package(library, (pin,), COMMIT)


@pytest.mark.parametrize(
    "field,value",
    [
        ("training_observed_point_count", 27),
        ("dense_frame_count", 3),
        ("training_frame_count", 3),
    ],
)
def test_rehashed_assessment_cannot_inflate_image_counts(research_case, field, value):
    library, pin, summary, _, _ = research_case
    summary["runs"][0]["fresh_strict_restart"][field] = value
    pin = replace(pin, assessment_sha256=save_json(pin.assessment_path, summary))
    with pytest.raises(ValueError, match="image evidence/counts"):
        build_historical_research_package(library, (pin,), COMMIT)


def test_schema_validation_dependency_is_required_without_fallback(
    research_case, monkeypatch
):
    import builtins

    library, pin, _, _, _ = research_case
    original = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "jsonschema" or name.startswith("jsonschema."):
            raise ImportError("Dependency deliberately unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    with pytest.raises(RuntimeError, match="requires.*jsonschema"):
        build_historical_research_package(library, (pin,), COMMIT)


def test_v15_assessment_adapter_preserves_generic_pins_and_rejection(research_case):
    library, pin, summary, _, _ = research_case
    summary["schema"] = "necromatcher/normalized-coverage-v15-independent-assessment/1"
    pin = replace(pin, assessment_sha256=save_json(pin.assessment_path, summary))
    payload = build_historical_research_package(library, (pin,), "b" * 40)
    record = json.loads(payload)["records"][0]
    assert record["player_name"] == "Third Player"
    assert record["producer_commit"] == COMMIT
    assert record["scientific_acceptance"] == "rejected"
    assert record["source_clock"]["physical_time"] == "unknown"
    assert record["continuous_certified"] is False
    assert record["image_counts"]["training_observation_count"] == 26
    assert build_historical_research_package(library, (pin,), "b" * 40) == payload


@pytest.mark.parametrize(
    "field",
    ["both_runs_verified", "source_runtime_parent_artifact_evidence_brackets_verified"],
)
@pytest.mark.parametrize("value", [None, False])
def test_v15_assessment_requires_both_explicit_verification_flags(
    research_case, field, value
):
    library, pin, summary, _, _ = research_case
    summary["schema"] = "necromatcher/normalized-coverage-v15-independent-assessment/1"
    if value is None:
        summary.pop(field)
    else:
        summary[field] = value
    pin = replace(pin, assessment_sha256=save_json(pin.assessment_path, summary))
    with pytest.raises(ValueError, match="unverified independent assessment"):
        build_historical_research_package(library, (pin,), COMMIT)


@pytest.mark.parametrize(
    "field",
    [
        "scientific_acceptance",
        "continuous_certified",
        "physical_time_qualified",
        "optimization_performed",
    ],
)
def test_v15_rehashed_assessment_cannot_promote_research_states(research_case, field):
    library, pin, summary, _, _ = research_case
    summary["schema"] = "necromatcher/normalized-coverage-v15-independent-assessment/1"
    summary[field] = True
    pin = replace(pin, assessment_sha256=save_json(pin.assessment_path, summary))
    with pytest.raises(ValueError, match=f"cannot promote {field}"):
        build_historical_research_package(library, (pin,), COMMIT)


def test_v15_rehashed_assessment_fit_digest_cannot_override_saved_parent(research_case):
    library, pin, summary, _, _ = research_case
    summary["schema"] = "necromatcher/normalized-coverage-v15-independent-assessment/1"
    summary["runs"][0]["fit_sha256"] = "sha256:" + "0" * 64
    pin = replace(pin, assessment_sha256=save_json(pin.assessment_path, summary))
    with pytest.raises(ValueError, match="fit identity differs"):
        build_historical_research_package(library, (pin,), COMMIT)
