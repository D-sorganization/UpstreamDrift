"""Canonical-v2 club-only trial and versioned threshold contract (T2-WP1, Refs #15787)."""

from __future__ import annotations

import copy
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.model_appearance import canonical_golfer_gate as gg
from src.shared.python.model_appearance.player_models import build_player_spec
from src.shared.python.motion_matching.club_only import canonical_v2_trial as cv2
from src.shared.python.motion_matching.club_only import native_g1_gates as g1
from src.shared.python.motion_matching.club_only import observation as obs_m

pytestmark = pytest.mark.unit
_CTRL = {
    "kp": [200.0] * 3,
    "kd": [20.0] * 3,
    "knots": [0.0, 0.0, 0.125, 0.25, 0.25],
    "control": {"hold": "zoh"},
    "integrator": {"name": "rk4", "dt": 0.004},
}


@lru_cache(maxsize=1)
def _tiger() -> dict[str, Any]:
    return build_player_spec("tiger_woods")


def _run(
    obs: obs_m.ClubObservation,
    *,
    dx: float = 0.0,
    mid: str = "full_body_mujoco",
    mut: tuple[str, str, Any] | None = None,
) -> cv2.CanonicalV2TrialResult:
    spec = copy.deepcopy(_tiger())
    if mut is not None:
        spec[mut[0]][mut[1]] = mut[2]
    pred = {
        "grip_xyz_m": obs.mid_hands_xyz.copy(),
        "clubhead_xyz_m": obs.face_xyz.copy(),
        "grip_quat_wxyz": obs.mid_hands_quat.copy(),
        "clubhead_quat_wxyz": obs.face_quat.copy(),
        "grip_closure_residual_m": np.zeros(obs.native_time_s.shape[0]),
    }
    pred["clubhead_xyz_m"][:, 0] += dx
    return cv2.run_canonical_v2_trial(
        spec=spec,
        observation=obs,
        prediction=pred,
        controller=_CTRL,
        target_model_id=mid,
    )


def _assert_contract(
    res: cv2.CanonicalV2TrialResult, obs: obs_m.ClubObservation, *, ok: bool, mid: str
) -> None:
    assert isinstance(res, cv2.CanonicalV2TrialResult)
    assert res.schema == cv2.CANONICAL_V2_TRIAL_SCHEMA
    assert res.trial_id == "TW_ProV1"
    assert res.target_model_id == mid
    assert (res.qualified, res.native_g1_pass, res.claims_native_qualification) == (
        ok,
    ) * 3
    assert (len(res.blockers) == 0) is ok
    g1.validate_native_g1_claim_contract(
        native_g1_pass=res.native_g1_pass,
        claims_native_qualification=res.claims_native_qualification,
        qualification_blockers=res.blockers,
    )
    n = obs.native_time_s.shape[0]
    assert res.position_error_m.shape == (n,)
    assert res.orientation_error_rad.shape == (n,)


def test_non_canonical_golfer_spec_raises_before_trial() -> None:
    obs = obs_m.build_calibrated_observation_fixture("TW_ProV1")
    for sec, key, val, part in (
        ("subject", "hand_visual_backend", "capsule", "capsule"),
        ("subject", "body_visual_backend", "loft", "loft"),
        ("spec_revision", "configuration", "legacy", "spec_revision"),
        ("club", "shaft", {"model": "rigid"}, "shaft"),
    ):
        with pytest.raises(gg.NonCanonicalGolferError) as exc:
            _run(obs, mut=(sec, key, val))
        assert exc.value.part == part


def test_threshold_fixture_loads_deterministically_with_sha256_pin(
    tmp_path: Path,
) -> None:
    first = cv2.load_canonical_v2_thresholds()
    assert isinstance(first, cv2.CanonicalV2Thresholds)
    assert first == cv2.load_canonical_v2_thresholds()
    assert first.schema_version == cv2.THRESHOLDS_SCHEMA
    assert first.content_sha256 == cv2.EXPECTED_THRESHOLDS_SHA256
    assert (first.governing_issue, first.panel_id) == (15787, "th_18425773591c")
    raw = json.loads(cv2.DEFAULT_THRESHOLDS_PATH.read_text(encoding="utf-8"))
    for fname, val, msg, pin in (
        ("mut.json", 0.099, "SHA-256", False),
        ("zero.json", 0.0, "> 0", True),
    ):
        raw["kinematic_tolerances"]["grip_position_rmse_m"] = val
        data = (json.dumps(raw, indent=2) + "\n").encode("utf-8")
        out = tmp_path / fname
        out.write_bytes(data)
        kw = {"expected_sha256": hashlib.sha256(data).hexdigest()} if pin else {}
        with pytest.raises(ValueError, match=msg):
            cv2.load_canonical_v2_thresholds(out, **kw)


def test_failed_tw_prov1_trial_preserves_full_residuals_and_blockers() -> None:
    obs = obs_m.build_calibrated_observation_fixture("TW_ProV1")
    res = _run(obs, dx=0.25, mid="full_body_mujoco_tiger_woods")
    _assert_contract(res, obs, ok=False, mid="full_body_mujoco_tiger_woods")
    assert np.allclose(res.position_error_m, 0.25)
    assert np.allclose(res.orientation_error_rad, 0.0)
    assert res.summary_stats["clubhead_position_rmse_m"] == pytest.approx(0.25)
    assert res.summary_stats["grip_position_rmse_m"] == pytest.approx(0.0)


def test_passing_synthetic_tw_prov1_trial_qualifies_honestly() -> None:
    obs = obs_m.build_calibrated_observation_fixture("TW_ProV1")
    res = _run(obs)
    _assert_contract(res, obs, ok=True, mid="full_body_mujoco")
    assert res.summary_stats["clubhead_position_rmse_m"] == pytest.approx(0.0)
    assert res.summary_stats["grip_closure_max_m"] == pytest.approx(0.0)


def test_private_workbook_loading_without_capture_data_dir_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CAPTURE_DATA_DIR", raising=False)
    with pytest.raises(FileNotFoundError, match="CAPTURE_DATA_DIR"):
        cv2.load_tw_prov1_via_capture_dir()
