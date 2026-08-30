"""Hosted structural-campaign workflow resource contracts."""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.unit


def test_structural_campaign_retains_partial_artifacts_with_safe_timeout() -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/heavy-tests-opt-in.yml").read_text(encoding="utf-8")
    )
    job = workflow["jobs"]["structural-runtime-audit"]

    assert job["timeout-minutes"] == 90
    upload = next(
        step
        for step in job["steps"]
        if step.get("name") == "Upload Structural Campaign Checkpoints"
    )
    assert "always()" in upload["if"]
    assert upload["with"]["if-no-files-found"] == "error"


def test_structural_campaign_replays_runtime_contract_before_execution() -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/heavy-tests-opt-in.yml").read_text(encoding="utf-8")
    )
    job = workflow["jobs"]["structural-runtime-audit"]

    assert job["env"] == {
        "BLIS_NUM_THREADS": "1",
        "MKL_DYNAMIC": "FALSE",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_DYNAMIC": "FALSE",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_CORETYPE": "Haswell",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    names = [step.get("name") for step in job["steps"]]
    registration = names.index("Validate Registered Recovery Slice")
    observed = names.index("Generate Current-Run Runtime Audit")
    replay = names.index("Compare Qualified and Current Runtime")
    retain = names.index("Upload Structural Runtime Replay Evidence")
    execute = names.index("Run Registered Structural Campaign Slice")
    assert registration < observed < replay < retain < execute
    registration_step = job["steps"][registration]
    assert (
        "structural_factorial_recovery_registration validate-slice"
        in (registration_step["run"])
    )
    assert "structural_case_start" in registration_step["run"]
    assert "structural_case_stop" in registration_step["run"]
    upload = job["steps"][retain]
    assert "always()" in upload["if"]
    assert upload["with"]["if-no-files-found"] == "error"
