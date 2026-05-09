"""Smoke tests for the Drake visualisation parity package (issue #4126).

The viz modules are deliberately engine-agnostic at the Python level
(they only consume the documented :class:`DrakeFitResult` and
:class:`ClubTarget` fields) so these tests run on every CI node --
``pydrake`` is not required for the matplotlib + dataclass paths.

Tests that exercise the live Meshcat HTML pathway are additionally
gated on ``requires_drake`` so they are skipped on the standard CI
runner.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from src.engines.physics_engines.drake.python.motion_matching.viz import (
    DrakeFitResult,
    OverlayArtifacts,
    render_error_timecourse,
    render_fit_quality_card,
    render_trajectory_overlay,
)
from src.engines.physics_engines.drake.python.motion_matching.viz.render_fit_quality_card import (
    compute_fit_quality_metrics,
)
from src.shared.python.motion_matching.club_target import (
    ClubTarget,
    SourceProvenance,
)

# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _synthetic_target(n: int = 32) -> ClubTarget:
    """A short, smooth synthetic ``ClubTarget`` good enough for smoke tests.

    The trajectory is a small circular arc traced out by the clubhead
    around the butt; the orientation is held identity for the duration
    so the orientation-error panel renders a flat zero line.
    """
    t = np.linspace(0.0, 0.3, n, dtype=float)
    butt = np.zeros((n, 3), dtype=float)
    radius = 0.5
    omega = 2.0 * np.pi * 1.5  # ~1.5 Hz
    head = np.stack(
        [
            radius * np.cos(omega * t),
            radius * np.sin(omega * t),
            0.5 + 0.05 * t,
        ],
        axis=1,
    )
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    return ClubTarget(
        time=t,
        butt=butt,
        clubhead=head,
        club_quat=quat,
        impact_idx=n // 2 + 1,  # 1-based per CLUB_IK_SPEC
        source=SourceProvenance(
            filename="synthetic.csv",
            format="synthetic",
            subject_id="UNIT",
            trial_id="VIZ",
            sha256="0" * 64,
        ),
    )


def _synthetic_fit(target: ClubTarget, *, with_tau: bool = True) -> DrakeFitResult:
    """A synthetic ``DrakeFitResult`` whose paths drift slightly from target.

    A small constant offset on the clubhead and a low-amplitude wobble
    on the grip exercise the position-error and orientation-error
    panels without violating the validators.
    """
    grip = np.asarray(target.butt) + np.array([0.005, 0.0, 0.0])
    head = np.asarray(target.clubhead) + np.array([0.0, 0.005, 0.002])
    quat = np.asarray(target.club_quat).copy()
    # Tiny rotation around z for a non-zero orientation error.
    half = 1.0e-3 / 2.0
    quat[:, 0] = np.cos(half)
    quat[:, 3] = np.sin(half)
    norms = np.linalg.norm(quat, axis=1, keepdims=True)
    quat = quat / np.maximum(norms, 1.0e-12)
    n_joints = 5
    tau = (
        0.1
        * np.sin(
            2.0
            * np.pi
            * np.arange(n_joints).reshape(1, -1)
            * target.time.reshape(-1, 1)
        )
        if with_tau
        else None
    )
    coeffs = np.zeros(n_joints * 7, dtype=float)
    return DrakeFitResult(
        time=target.time.copy(),
        grip=grip,
        clubhead=head,
        club_quat=quat,
        coefficients=coeffs,
        final_loss=1.234e-3,
        tau=tau,
        solver_status="success",
        wall_clock_s=2.5,
        n_iterations=42,
        swing_id="SYNTHETIC_TW",
        commit_hash="abcdef0",
        branch="feat/issue-4126",
    )


# ---------------------------------------------------------------------------
# Dataclass validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_drake_fit_result_rejects_mismatched_shapes() -> None:
    """``DrakeFitResult.__post_init__`` is the DbC line-of-defence."""
    with pytest.raises(ValueError, match="grip"):
        DrakeFitResult(
            time=np.linspace(0.0, 0.3, 8),
            grip=np.zeros((4, 3)),  # wrong N
            clubhead=np.zeros((8, 3)),
            club_quat=np.tile([1.0, 0.0, 0.0, 0.0], (8, 1)),
            coefficients=np.zeros(7),
            final_loss=0.0,
        )


@pytest.mark.unit
def test_drake_fit_result_rejects_bad_solver_status() -> None:
    """Only ``success`` / ``warning`` / ``failed`` survive validation."""
    with pytest.raises(ValueError, match="solver_status"):
        DrakeFitResult(
            time=np.linspace(0.0, 0.3, 8),
            grip=np.zeros((8, 3)),
            clubhead=np.zeros((8, 3)),
            club_quat=np.tile([1.0, 0.0, 0.0, 0.0], (8, 1)),
            coefficients=np.zeros(7),
            final_loss=0.0,
            solver_status="weird",
        )


# ---------------------------------------------------------------------------
# View 1 -- trajectory overlay
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.requires_drake
def test_render_trajectory_overlay_writes_png(tmp_path: Path) -> None:
    """The PNG fallback is *always* produced, even without pydrake.

    The mark is ``requires_drake`` per the issue acceptance criteria;
    the body itself only relies on matplotlib so the test is also
    valid as a smoke test on a Drake-equipped runner.
    """
    target = _synthetic_target()
    fit = _synthetic_fit(target)

    artefacts = render_trajectory_overlay(fit, target, tmp_path, title="UnitTest")

    assert isinstance(artefacts, OverlayArtifacts)
    assert artefacts.png_path.exists()
    assert artefacts.png_path.stat().st_size > 0
    # HTML is best-effort: present iff pydrake imported successfully.
    if artefacts.html_path is not None:
        assert artefacts.html_path.exists()
        assert artefacts.html_path.stat().st_size > 0


@pytest.mark.unit
@pytest.mark.requires_drake
def test_render_trajectory_overlay_rejects_mismatched_grids(tmp_path: Path) -> None:
    """Mismatched timegrids are a hard fail (loaders/sim guarantee parity)."""
    target = _synthetic_target(n=32)
    fit = _synthetic_fit(_synthetic_target(n=16))
    with pytest.raises(ValueError, match="timegrid"):
        render_trajectory_overlay(fit, target, tmp_path)


# ---------------------------------------------------------------------------
# View 2 -- error timecourse
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.requires_drake
def test_render_error_timecourse_writes_png(tmp_path: Path) -> None:
    """The 2D matplotlib timecourse is produced and non-empty."""
    target = _synthetic_target()
    fit = _synthetic_fit(target)
    out = render_error_timecourse(fit, target, tmp_path)
    assert out.exists()
    assert out.suffix == ".png"
    assert out.stat().st_size > 0


@pytest.mark.unit
@pytest.mark.requires_drake
def test_render_error_timecourse_works_without_torque(tmp_path: Path) -> None:
    """If ``tau`` is ``None`` the joint-torque panel is skipped (no crash)."""
    target = _synthetic_target()
    fit = _synthetic_fit(target, with_tau=False)
    out = render_error_timecourse(fit, target, tmp_path)
    assert out.exists()


# ---------------------------------------------------------------------------
# View 3 -- fit quality summary card
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.requires_drake
def test_render_fit_quality_card_writes_png(tmp_path: Path) -> None:
    """The summary card writes a PNG of non-zero size."""
    target = _synthetic_target()
    fit = _synthetic_fit(target)
    out = render_fit_quality_card(fit, target, tmp_path, title="Smoke")
    assert out.exists()
    assert out.suffix == ".png"
    assert out.stat().st_size > 0


@pytest.mark.unit
@pytest.mark.requires_drake
def test_compute_fit_quality_metrics_round_trip() -> None:
    """The headline metrics are real, finite, and shape-stable."""
    target = _synthetic_target()
    fit = _synthetic_fit(target)
    metrics = compute_fit_quality_metrics(fit, target)
    assert metrics.clubhead_rmse_mm >= 0.0
    assert metrics.butt_rmse_mm >= 0.0
    assert 0.0 <= metrics.mean_orient_err_deg < 360.0
    # Synthetic offsets were tiny but non-zero -> RMS error is positive.
    assert metrics.clubhead_rmse_mm > 0.0
    assert metrics.butt_rmse_mm > 0.0
