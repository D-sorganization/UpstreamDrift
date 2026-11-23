"""Fail-closed tests for the v2 baseline forward-tracking diagnostic."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts import audit_15493_baseline as audit

pytestmark = pytest.mark.unit


def bundle():
    return SimpleNamespace(
        spec_bytes=b'{"coordinate_order":["TranslationX","joint"]}',
        coordinate_order=("TranslationX", "joint"),
        target_time_s=np.array([3.0, 4.0]),
        target_q=np.array([[0.0, 0.0], [1.0, 2.0]]),
        steps=2,
        dt_s=0.5,
        step_policy=lambda: "policy",
    )


def test_missing_target_and_model_mismatch_do_not_run(monkeypatch):
    b = bundle()
    monkeypatch.setattr(audit, "closed_loop", lambda *a, **k: pytest.fail("ran"))
    b.target_time_s = None
    result = audit.audit_bundle(b, {"coordinate_order": ["TranslationX", "joint"]})
    assert result["status"] == "unqualified"
    assert result["tracking"]["status"] == "unavailable"
    b.target_time_s = np.array([3.0, 4.0])
    result = audit.audit_bundle(b, {"coordinate_order": ["other"]})
    assert result["tracking"]["status"] == "unavailable"


def test_forward_tracking_is_diagnostic_and_unqualified(monkeypatch):
    def fake_closed_loop(*args, **kwargs):
        assert kwargs["impact"] is None
        assert kwargs["policy"] == "policy"
        return SimpleNamespace(
            time_s=np.array([0.0, 0.5, 1.0]),
            q=np.array([[0.0, 0.0], [0.6, 0.8], [1.0, 2.0]]),
            failure=None,
        )

    monkeypatch.setattr(audit, "closed_loop", fake_closed_loop)
    result = audit.audit_bundle(
        bundle(), {"coordinate_order": ["TranslationX", "joint"]}
    )
    assert result["status"] == "unqualified"
    assert result["tracking"]["status"] == "measured"
    assert result["tracking"]["per_coordinate"]["TranslationX"]["unit"] == "m"
    assert result["tracking"]["per_coordinate"]["joint"]["unit"] == "rad"
    assert result["tracking"]["per_coordinate"]["joint"]["max_abs"] == pytest.approx(
        0.2
    )
    assert all(gate == "unverified" for gate in result["gates"].values())


def test_cli_unqualified_receipt_has_nonzero_exit(monkeypatch, tmp_path):
    import json
    import sys

    result = {"status": "unqualified", "tracking": {"status": "unavailable"}}
    monkeypatch.setattr(audit, "audit_bundle", lambda *args: result)
    monkeypatch.setattr(audit.InputBundle, "load", lambda path: bundle())
    out = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["audit", "unused.npz", "--out", str(out)])
    with pytest.raises(SystemExit) as error:
        audit.main()
    assert error.value.code == 2
    assert json.loads(out.read_text(encoding="utf-8")) == result


@pytest.mark.requires_mujoco
def test_synthetic_v2_bundle_runs_existing_mujoco_tracker():
    import json

    pytest.importorskip("mujoco")
    from scipy.optimize import least_squares

    from src.shared.python.motion_matching.same_input import InputBundle, VectorPlant

    spec_bytes = audit.V2_SPEC.read_bytes()
    spec = json.loads(spec_bytes)
    names = tuple(spec["coordinate_order"])
    plant = VectorPlant("mujoco", spec_bytes)
    closure = least_squares(
        plant.closure_pose_residual,
        np.zeros(len(names)),
        max_nfev=100,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    assert closure.success and np.max(np.abs(closure.fun)) < 1e-8
    q = closure.x
    target = np.tile(q, (3, 1))
    velocity = np.zeros((3, len(names)))
    synthetic = InputBundle(
        spec_bytes=spec_bytes,
        coordinate_order=names,
        dt_s=0.001,
        q0=q,
        v0=velocity[0],
        efforts=np.zeros((2, len(names))),
        reference_q=target,
        reference_v=velocity,
        reference_engine="mujoco",
        target_time_s=np.array([0.0, 0.001, 0.002]),
        target_q=target,
    )
    report = audit.audit_bundle(synthetic, spec)
    assert report["status"] == "unqualified"
    assert report["tracking"]["status"] == "measured"
    assert report["tracking"]["samples"] == 3
    assert report["tracking"]["rollout_failure"] is None
