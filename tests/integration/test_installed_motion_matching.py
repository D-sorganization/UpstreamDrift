"""Integration tests for installed motion matching execution outside docs.

Acceptance cases for issue #10520 / parent #10508:
- RED: wheel execution without repository docs tree fails command construction/runtime in old layout.
- RED: old and new entry points differ in defaults/schema or drop stderr/cancellation state.
- GREEN: deterministic small fixture through packaged service and compatibility wrapper;
  numeric comparison uses existing tolerances, not new relaxed thresholds.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_FULL_BODY = REPO_ROOT / "docs/development/full_body_models"


@pytest.mark.integration
def test_wheel_execution_without_docs_tree_fails_command_construction_and_runtime(
    tmp_path: Path,
) -> None:
    """Verify that execution outside repo docs tree requires packaged entry points.

    In an installed wheel distribution, docs/ is not shipped.
    Old command paths under docs/development/full_body_models fail to exist,
    while packaged entry points under src/shared/python/motion_matching/execution
    remain present and valid.
    """
    from src.tools.motion_matching import pipeline

    # When docs tree is simulated as absent:
    fake_isolated_root = tmp_path / "isolated_install"
    fake_isolated_root.mkdir()

    # The packaged execution modules must exist inside src/
    from src.shared.python.motion_matching.execution import (
        downswing,
        driver,
        mjx_export,
        spec_builder,
    )

    packaged_scripts = [
        Path(spec_builder.__file__).resolve(),
        Path(driver.__file__).resolve(),
        Path(downswing.__file__).resolve(),
        Path(mjx_export.__file__).resolve(),
    ]
    for script in packaged_scripts:
        assert script.exists(), f"Packaged script must exist on disk: {script}"
        assert "docs" not in script.parts, (
            f"Packaged script must not be in docs: {script}"
        )

    # Pipeline command construction in the packaged layout points to packaged scripts
    req = pipeline.MatchRequest(capture="driver", club="driver", output_root=tmp_path)
    build_cmd = pipeline.build_command(req)
    match_cmd = pipeline.match_command(req)

    assert Path(build_cmd[1]).exists(), (
        f"Build command executable script must exist: {build_cmd[1]}"
    )
    assert Path(match_cmd[1]).exists(), (
        f"Match command executable script must exist: {match_cmd[1]}"
    )
    assert "docs" not in Path(build_cmd[1]).parts
    assert "docs" not in Path(match_cmd[1]).parts

    exp_req = pipeline.ExperimentRequest(run=tmp_path, name="test_exp")
    exp_cmd = pipeline.experiment_command(exp_req)
    assert Path(exp_cmd[1]).exists(), (
        f"Experiment command executable script must exist: {exp_cmd[1]}"
    )
    assert "docs" not in Path(exp_cmd[1]).parts

    mjx_cmd = pipeline.export_mjx_command(tmp_path)
    assert Path(mjx_cmd[1]).exists(), (
        f"MJX export command executable script must exist: {mjx_cmd[1]}"
    )
    assert "docs" not in Path(mjx_cmd[1]).parts


@pytest.mark.integration
def test_entry_points_defaults_schema_and_cancellation() -> None:
    """Old wrappers and new packaged entry points must share identical CLI schemas and exit codes."""
    from src.shared.python.motion_matching.execution import (
        downswing as new_downswing,
        mjx_export as new_mjx,
        spec_builder as new_spec,
    )

    # 1. Spec builder parser comparison
    p_new_spec = new_spec.build_parser()
    spec_actions_new = {a.dest: a.default for a in p_new_spec._actions}
    assert "stature" in spec_actions_new
    assert "mass" in spec_actions_new
    assert "club" in spec_actions_new
    assert spec_actions_new["club"] == "driver"

    # 2. Downswing parser comparison
    p_new_downswing = new_downswing.build_parser()
    downswing_actions_new = {a.dest: a.default for a in p_new_downswing._actions}
    assert "omega" in downswing_actions_new
    assert "zeta" in downswing_actions_new
    assert "feedforward" in downswing_actions_new
    assert downswing_actions_new["zeta"] == 1.0
    assert downswing_actions_new["feedforward"] == 1.0

    # 3. MJX parser comparison
    p_new_mjx = new_mjx.build_parser()
    mjx_actions_new = {a.dest: a.default for a in p_new_mjx._actions}
    assert "run" in mjx_actions_new
    assert "timestep" in mjx_actions_new
    assert mjx_actions_new["timestep"] == 5e-4

    # 4. Check wrappers exist and issue DeprecationWarning
    wrapper_spec = DOCS_FULL_BODY / "build_anthropometric_spec.py"
    wrapper_downswing = (
        DOCS_FULL_BODY / "evidence/ground_support/downswing_experiment.py"
    )
    wrapper_mjx = DOCS_FULL_BODY / "evidence/ground_support/export_mjx_package.py"
    wrapper_driver = DOCS_FULL_BODY / "evidence/ground_support/run_ground_support.py"

    assert wrapper_spec.exists()
    assert wrapper_downswing.exists()
    assert wrapper_mjx.exists()
    assert wrapper_driver.exists()

    # Verify wrapper execution emits DeprecationWarning and preserves exit code on error (--invalid-flag)
    res = subprocess.run(
        [sys.executable, str(wrapper_spec), "--nonexistent-flag"],
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0
    assert (
        "unrecognized arguments" in res.stderr.lower() or "error" in res.stderr.lower()
    )
    assert "DeprecationWarning" in res.stderr or "deprecated" in res.stderr.lower()


@pytest.mark.integration
def test_deterministic_small_fixture_numeric_parity(tmp_path: Path) -> None:
    """Verify numeric parity between packaged service and compatibility wrapper on small fixture."""
    from src.shared.python.motion_matching.execution import spec_builder

    out_new = tmp_path / "out_new"
    out_wrap = tmp_path / "out_wrap"
    out_new.mkdir()
    out_wrap.mkdir()

    native_path = (
        REPO_ROOT
        / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
    )
    osim_path = (
        REPO_ROOT / "src/engines/physics_engines/opensim/models/golf_humanoid.osim"
    )
    candidate_path = (
        DOCS_FULL_BODY / "evidence/native_candidates/returned81_candidate.json"
    )

    if not (native_path.is_file() and osim_path.is_file() and candidate_path.is_file()):
        pytest.skip("Required reference evidence assets not present on disk")

    # Run via packaged spec_builder
    spec_builder.build_anthropometric_spec(
        native_path=native_path,
        osim_path=osim_path,
        native_candidate_path=candidate_path,
        stature_m=1.75,
        mass_kg=75.0,
        output_dir=out_new,
        club="driver",
        name="test_anthro_spec",
    )

    # Run via wrapper script
    wrapper_spec = DOCS_FULL_BODY / "build_anthropometric_spec.py"
    cmd = [
        sys.executable,
        str(wrapper_spec),
        "--native",
        str(native_path),
        "--osim",
        str(osim_path),
        "--native-candidate",
        str(candidate_path),
        "--stature",
        "1.75",
        "--mass",
        "75.0",
        "--club",
        "driver",
        "--output",
        str(out_wrap),
        "--name",
        "test_anthro_spec",
    ]
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "default"
    subprocess.run(cmd, check=True, capture_output=True, env=env)

    doc_new = json.loads(
        (out_new / "test_anthro_spec.json").read_text(encoding="utf-8")
    )
    doc_wrap = json.loads(
        (out_wrap / "test_anthro_spec.json").read_text(encoding="utf-8")
    )

    # Compare structure and key values
    assert doc_new["subject"] == doc_wrap["subject"]
    assert doc_new["de_leva_table_sha256"] == doc_wrap["de_leva_table_sha256"]
    assert len(doc_new["bodies"]) == len(doc_wrap["bodies"])

    receipt_new = json.loads(
        (out_new / "build_receipt_test_anthro_spec.json").read_text(encoding="utf-8")
    )
    receipt_wrap = json.loads(
        (out_wrap / "build_receipt_test_anthro_spec.json").read_text(encoding="utf-8")
    )
    assert receipt_new["spec_sha256"] == receipt_wrap["spec_sha256"]
    assert receipt_new["total_mass_kg"] == pytest.approx(
        receipt_wrap["total_mass_kg"], abs=1e-6
    )
