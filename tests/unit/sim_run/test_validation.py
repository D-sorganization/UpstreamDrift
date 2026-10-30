"""Unit tests for simulation result validation and numerical integrity (Epic #14085 / #14089).

Verifies that:
- Result envelope provenance (repo SHA, engine version, input sha256) is strictly validated.
- NaN / Inf values in data or analysis_results force validation failure with diagnostics.
- Schema-invalid worker outputs force run status to 'failed'.
- Falsifying test: A fixture engine emitting NaN CANNOT reach 'succeeded'.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest

from src.api.models.requests import SimulationRequest
from src.sim_run.models import Provenance, ResultEnvelope, RunRecord, RunRequest
from src.sim_run.store import SimulationRunStore
from src.sim_run.validation import (
    validate_numerical_integrity,
    validate_provenance,
    validate_result,
)
from src.sim_run.worker import SimulationWorkerPool, get_repo_sha

pytestmark = pytest.mark.unit


def _make_provenance(**kwargs: Any) -> Provenance:
    defaults = {
        "repo_sha": "a" * 40,
        "engine_type": "golf_swing_pendulum",
        "engine_version": "1.0.0",
        "input_sha256": "b" * 64,
        "created_at": "2026-10-30T00:00:00Z",
        "completed_at": "2026-10-30T00:01:00Z",
        "host": "test-runner",
    }
    defaults.update(kwargs)
    return Provenance.model_validate(defaults)


def _make_envelope(**kwargs: Any) -> ResultEnvelope:
    defaults = {
        "schema_version": "1.0.0",
        "run_id": "run_test_0001",
        "provenance": _make_provenance(),
        "validation_status": "unvalidated",
        "validation_errors": [],
        "success": True,
        "data": {"trajectory": [1.0, 2.0, 3.0]},
        "analysis_results": None,
        "artifacts": [],
        "error": None,
    }
    defaults.update(kwargs)
    return ResultEnvelope.model_validate(defaults)


def test_validate_provenance_clean() -> None:
    prov = _make_provenance()
    errors = validate_provenance(prov)
    assert errors == []


def test_validate_provenance_rejects_empty_or_unknown_sha() -> None:
    prov = _make_provenance(repo_sha="")
    errors = validate_provenance(prov)
    assert any("repo_sha" in e for e in errors)

    prov2 = _make_provenance(repo_sha="unknown")
    errors2 = validate_provenance(prov2)
    assert any("repo_sha" in e for e in errors2)


def test_validate_provenance_rejects_invalid_input_sha256() -> None:
    prov = _make_provenance(input_sha256="not_a_sha")
    errors = validate_provenance(prov)
    assert any("input_sha256" in e for e in errors)


def test_validate_numerical_integrity_clean() -> None:
    data = {
        "scalars": {"energy": 42.5, "step": 10},
        "times": [0.0, 0.01, 0.02],
        "nested": [{"val": 1.23}, {"val": -4.56}],
    }
    errors = validate_numerical_integrity(data, path="data")
    assert errors == []


def test_validate_numerical_integrity_detects_nan() -> None:
    data = {
        "trajectory": [
            {"time": 0.0, "q": 1.0},
            {"time": 0.1, "q": float("nan")},
        ]
    }
    errors = validate_numerical_integrity(data, path="data")
    assert len(errors) == 1
    assert "NaN" in errors[0]
    assert "data.trajectory[1].q" in errors[0]


def test_validate_numerical_integrity_detects_inf() -> None:
    data = {"velocity": float("inf"), "torque": float("-inf")}
    errors = validate_numerical_integrity(data, path="data")
    assert len(errors) == 2
    assert any("Inf" in e and "velocity" in e for e in errors)
    assert any("-Inf" in e and "torque" in e for e in errors)


def test_validate_result_valid_envelope() -> None:
    env = _make_envelope(data={"steps": [1.0, 2.0]})
    is_valid, out_env, errors = validate_result(env)
    assert is_valid is True
    assert errors == []
    assert out_env is not None
    assert out_env.validation_status == "valid"
    assert out_env.validation_errors == []


def test_validate_result_invalid_envelope_with_nan() -> None:
    env = _make_envelope(data={"steps": [1.0, float("nan")]})
    is_valid, out_env, errors = validate_result(env)
    assert is_valid is False
    assert len(errors) > 0
    assert any("NaN" in e for e in errors)
    assert out_env is not None
    assert out_env.validation_status == "invalid"
    assert out_env.validation_errors == errors


def test_validate_result_schema_invalid_dict() -> None:
    raw = {"invalid_key": 123}
    is_valid, out_env, errors = validate_result(raw)
    assert is_valid is False
    assert out_env is None
    assert len(errors) > 0
    assert any("schema" in e.lower() or "validation" in e.lower() for e in errors)


def test_falsifying_fixture_engine_emitting_nan_cannot_reach_succeeded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Falsifying test: A fixture engine emitting NaN cannot reach succeeded (#14089)."""
    store = SimulationRunStore(tmp_path / "runs", max_bytes=10_000_000)
    pool = SimulationWorkerPool(store)

    req = RunRequest.model_validate(
        SimulationRequest(
            duration=1.0,
            timestep=0.01,
            engine_type="golf_swing_pendulum",
        ).model_dump()
    )
    rec = store.create_run(req)
    assert rec.status == "queued"

    # Define a fixture engine worker script that emits an output envelope containing NaN
    script = tmp_path / "nan_engine_worker.py"
    script.write_text(
        """import sys, json, math
from pathlib import Path
from src.sim_run.models import ResultEnvelope, Provenance

out_file = Path(sys.argv[1])
now = "2026-10-30T00:00:00Z"
envelope = ResultEnvelope(
    schema_version="1.0.0",
    run_id="run_nan_test",
    provenance=Provenance(
        repo_sha="0" * 40,
        engine_type="golf_swing_pendulum",
        engine_version="1.0.0",
        input_sha256="1" * 64,
        created_at=now,
        completed_at=now,
        host="test-node",
    ),
    validation_status="unvalidated",
    validation_errors=[],
    success=True,
    data={"trajectory": [1.0, float("nan"), 3.0]},
    analysis_results=None,
    artifacts=[],
    error=None,
)
out_file.write_text(json.dumps(envelope.model_dump()), encoding="utf-8")
""",
        encoding="utf-8",
    )

    # Execute with worker_cmd invoking the nan_engine_worker
    out_path = store.get_run_dir(rec.run_id) / "output.json"
    worker_cmd = [sys.executable, str(script), str(out_path)]

    finished_rec = pool.execute_run(rec.run_id, worker_cmd=worker_cmd)

    # Assert that the run MUST NOT reach succeeded
    assert finished_rec.status != "succeeded", "Run with NaN output reached succeeded!"
    assert finished_rec.status == "failed"

    # Diagnostics verification
    assert finished_rec.result is not None
    assert finished_rec.result.validation_status == "invalid"
    assert any("NaN" in err for err in finished_rec.result.validation_errors)

    assert finished_rec.error is not None
    assert finished_rec.error.code == "validation_failed"
    assert finished_rec.error.stage == "validation"
    assert "NaN" in finished_rec.error.message


def test_get_repo_sha_format(tmp_path: Path) -> None:
    sha = get_repo_sha()
    assert isinstance(sha, str)
    assert len(sha) == 40
    assert all(c in "0123456789abcdefABCDEF" for c in sha)
