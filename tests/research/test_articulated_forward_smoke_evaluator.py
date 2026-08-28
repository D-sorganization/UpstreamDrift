"""Frozen-source native evaluation for the #9153 serial smoke."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.articulated_forward_attribution_runner import (
    NativeEngineUnavailable,
    build_registered_cases,
)
from scripts.research.proximal_distal_energy.articulated_forward_attribution_study import (
    ForwardAttributionStudyPlan,
)
from scripts.research.proximal_distal_energy.articulated_forward_smoke_evaluator import (
    evaluate_rigid_smoke_case,
    require_registered_native_engine,
    run_registered_rigid_smoke,
)


SOURCE_REVISION = "a" * 40
SOURCE_DATA_SHA256 = "9fa4364571ba5535995c63226289c0711ee1ebf37c58b7a3b4e4d14a98561779"


def _manifest() -> dict[str, object]:
    return ForwardAttributionStudyPlan(
        source_revision=SOURCE_REVISION,
        source_data_sha256=SOURCE_DATA_SHA256,
    ).to_manifest()


def test_wrong_pinocchio_package_is_a_typed_unavailable_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class WrongPinocchio:
        __version__ = "0.1"

    monkeypatch.setattr(
        "scripts.research.proximal_distal_energy."
        "articulated_forward_smoke_evaluator.import_module",
        lambda _name: WrongPinocchio(),
    )

    with pytest.raises(NativeEngineUnavailable, match="robotics Pinocchio"):
        require_registered_native_engine("pinocchio")


def test_native_dll_initialization_failure_is_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_import(_name: str) -> object:
        raise OSError("planted native DLL initialization failure")

    monkeypatch.setattr(
        "scripts.research.proximal_distal_energy."
        "articulated_forward_smoke_evaluator.import_module",
        fail_import,
    )

    with pytest.raises(NativeEngineUnavailable, match="native DLL"):
        require_registered_native_engine("mujoco")


def test_source_hash_mismatch_fails_before_native_evaluation() -> None:
    manifest = _manifest()
    identity = manifest["identity"]
    assert isinstance(identity, dict)
    identity["source_data_sha256"] = "0" * 64
    case = build_registered_cases(manifest)[0]

    with pytest.raises(ValueError, match="source-data SHA-256"):
        evaluate_rigid_smoke_case(case, manifest)


def test_mujoco_smoke_case_returns_json_safe_attribution_and_outcomes() -> None:
    manifest = _manifest()
    case = next(
        item
        for item in build_registered_cases(manifest)
        if item.engine == "mujoco"
        and item.variant == "nominal"
        and item.time_step_s == 0.0005
    )

    result = evaluate_rigid_smoke_case(case, manifest)

    assert result["source_state"] == {
        "source_case_index": 4,
        "source_sample_index": 6,
        "source_time_s": 0.12,
    }
    assert result["engine"]["name"] == "mujoco"
    assert result["estimand"] == "same_trajectory_descriptive_attribution"
    assert result["closure"]["momentum_relative_residual"] <= 0.02
    assert result["closure"]["work_relative_residual"] <= 0.01
    assert result["closure"]["failure_codes"] == []
    assert result["closure"]["passes_registered_tolerances"] is True
    assert result["outcomes"]["clubhead_speed_m_s"] > 0.0
    assert len(result["outcomes"]["clubhead_direction_world"]) == 3
    assert result["outcomes"]["face_path_proxy_definition"] == (
        "angle between club local +x axis and clubhead velocity in the world frame"
    )
    assert result["events"] == {
        "model": "bilateral_always_active_rigid_attachment",
        "count": 0,
        "times_s": [],
    }
    json.dumps(result, allow_nan=False)


def test_registered_smoke_wrapper_uses_atomic_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    monkeypatch.setattr(
        "scripts.research.proximal_distal_energy."
        "articulated_forward_smoke_evaluator.evaluate_rigid_smoke_case",
        lambda case, _manifest: {"case_key": case.case_key},
    )

    checkpoints = run_registered_rigid_smoke(
        manifest=manifest,
        execution_revision="c" * 40,
        checkpoint_dir=tmp_path,
    )

    assert len(checkpoints) == 42
    assert all(item.status == "completed" for item in checkpoints)
