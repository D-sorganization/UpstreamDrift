"""Unit tests for Track C shared sliding-window Tiger Woods fitter (Refs #15837)."""

import json
import numpy as np
import pytest

from src.shared.python.shadow_tracker import tiger_trajectory_fitter as ttf

pytestmark = [pytest.mark.unit]
_PH = ("address", "backswing", "top", "downswing", "impact", "finish")
_fit, _gates = ttf.fit_tiger_trajectory, ttf.evaluate_trajectory_gates
_cfg = ttf.SlidingWindowFitterConfig


def _obs(
    mode: ttf.FitMode, grip: float = 0.0, heel: float = 0.06, trunk: float = 0.25
) -> ttf.FitterObservationInput:
    """Create deterministic synthetic observations for a 6-frame golf swing."""
    n = len(_PH)
    a, f = np.zeros((n, 12)), np.linspace(0.0, 1.0, n)
    a[:, 0], a[:, 1], a[:, 2] = 0.2 * f, 0.1 * f, trunk * (1.0 - 2.0 * f)
    a[:, 3], a[:, 4], a[:, 5], a[:, 6] = 0.4 * f, 0.3, 0.4 * f + grip, 0.3
    a[3:, 11] = heel
    fids, rows = (
        tuple(f"f-{i}" for i in range(n)),
        tuple(tuple(map(float, r)) for r in a),
    )
    return ttf.FitterObservationInput(mode, fids, _PH, rows)


def test_adr0059_coupled_trunk_default_and_bruno_ratios() -> None:
    """Verify ADR-0059 coupled trunk default and Bruno 2015 regional split ratios."""
    cfg = _cfg()
    assert cfg.trunk_coupled and cfg.model_id == "mujoco-tiger-woods-2000"
    assert cfg.lumbar_split_ratios == (0.51, 0.27, 0.17)
    assert cfg.thoracic_split_ratios == (0.49, 0.73, 0.83)
    res = _fit(_obs("fused_AB"), cfg)
    for lum, tho in zip(res.lumbar_angles, res.thoracic_angles, strict=True):
        tot = np.asarray(lum) + np.asarray(tho)
        np.testing.assert_allclose(lum, tot * [0.51, 0.27, 0.17], atol=1e-6)
        np.testing.assert_allclose(tho, tot * [0.49, 0.73, 0.83], atol=1e-6)


def test_rigid_club_two_hand_grip_separation_violation_fails_gate() -> None:
    """Verify two-hand rigid-club grip separation beyond tolerance fails model-space check."""
    cfg = _cfg(grip_tol_m=0.015)
    res = _fit(_obs("fused_AB"), cfg)
    assert res.grip_gate_passed
    bad = np.asarray(res.fitted_angles)
    bad[:, 5] += 0.45
    m = _gates(bad, _PH, cfg)
    assert m["max_grip_separation_m"] > cfg.grip_tol_m and not m["grip_gate_passed"]


def test_phase_specific_foot_contact_slip_vs_legal_trail_heel_lift() -> None:
    """Verify planted foot slip fails gate while bounded trail-heel lift passes late phases."""
    cfg = _cfg(foot_slip_tol_m=0.02, max_trail_heel_lift_m=0.12)
    res = _fit(_obs("fused_AB", heel=0.08), cfg)
    assert res.foot_contact_gate_passed and 0.0 <= res.max_trail_heel_lift_m <= 0.12
    for idx, col, val in ((1, 7, 0.09), (0, 11, 0.06)):
        bad = np.asarray(res.fitted_angles)
        bad[idx, col] = val
        assert not _gates(bad, _PH, cfg)["foot_contact_gate_passed"]


def test_ablation_comparison_enforces_identical_config_sha256() -> None:
    """Verify A_only, B_only, and fused_AB share identical config_sha256 in ablation."""
    cfg = _cfg(window_size=4, stride=2, smoothness_weight=0.05)
    a_in, b_in = _obs("A_only", trunk=0.22), _obs("B_only", trunk=0.28)
    rep = ttf.run_ablation_comparison(a_in, b_in, cfg)
    hs = {rep.config_sha256, rep.a_only.config_sha256, rep.b_only.config_sha256}
    assert hs == {cfg.config_sha256} and rep.fused_ab.config_sha256 == cfg.config_sha256
    assert rep.best_mode in ("A_only", "B_only", "fused_AB")


def test_overlay_manifest_and_hash_pinned_scorecard_honesty(tmp_path) -> None:
    """Verify overlay manifest and scorecard retain unverified timebase and honest claims."""
    rep = ttf.run_ablation_comparison(_obs("A_only"), _obs("B_only"), _cfg())
    man = ttf.build_side_by_side_overlay_manifest(rep)
    exp = ("unverified", False, "image_observations_only")
    assert man.schema_version == "markerless-scorecard/1.0.0"
    assert (man.timebase, man.physical_time_verified, man.qualification) == exp
    assert len(man.frame_entries) == len(_PH) and len(man.manifest_sha256) == 64
    path = tmp_path / "scorecard.json"
    payload = ttf.export_scorecard_json(rep.fused_ab, path)
    d = json.loads(path.read_text(encoding="utf-8"))
    assert d == payload and len(d["scorecard_sha256"]) == 64
    assert (d["timebase"], d["physical_time_verified"], d["qualification"]) == exp
    assert d["claim_label"] == "kinematic_3d_plausible"
    c_bad = _cfg(reprojection_gate=0.05)
    assert _fit(_obs("A_only", grip=0.8), c_bad).claim_label == "below_gate"
    c_2d = _cfg(grip_tol_m=0.001, reprojection_gate=0.5, max_nfev=1)
    assert _fit(_obs("A_only", grip=0.1), c_2d).claim_label == "kinematic_2d_match"
