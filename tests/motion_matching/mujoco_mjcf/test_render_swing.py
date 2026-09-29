"""Smoke tests for the MuJoCo motion-matching thin renderer (issue #4125).

The renderer itself only depends on numpy + matplotlib, but it lives under
the mujoco engine package so the import-path test gates it behind the
``requires_mujoco`` marker. The render_swing module deliberately does not
import mujoco at module scope (it's a thin renderer, not a sim wrapper),
so these tests exercise the full PNG-emitting code path even when MuJoCo
is unavailable; the marker is honoured for environment selection in CI.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
from src.engines.physics_engines.mujoco.python.motion_matching.viz.render_swing import (
    FitResult,
    VizOptions,
    render_error_timecourse,
    render_fit_quality_card,
    render_trajectory_overlay,
)
from src.shared.python.motion_matching.club_target import (
    ClubTarget,
    SourceProvenance,
)

# ``requires_mujoco`` is the engine-selection marker. The actual code under
# test does not import mujoco, so the body still runs in environments
# without it; the marker is here so CI matrices that filter by engine
# (e.g. ``-m requires_mujoco``) include the suite.
pytestmark = pytest.mark.requires_mujoco


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_dimensions(path: Path) -> tuple[int, int]:
    """Return (width, height) by reading the PNG IHDR chunk."""
    data = path.read_bytes()
    if not data.startswith(_PNG_SIGNATURE):
        raise AssertionError(f"{path} is not a PNG (bad signature)")
    # IHDR chunk starts at byte 8: 4-byte length, 4-byte type 'IHDR',
    # then 4-byte width, 4-byte height (big-endian).
    if data[12:16] != b"IHDR":
        raise AssertionError(f"{path} is not a PNG (missing IHDR chunk)")
    width, height = struct.unpack(">II", data[16:24])
    return width, height


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _synthetic_target(n: int = 64) -> ClubTarget:
    """Build a tiny ClubTarget along a circular clubhead path."""
    rng = np.random.default_rng(seed=20260506)
    t = np.linspace(0.0, 0.3, n, dtype=np.float64)
    # Butt at origin (slowly drifting), clubhead orbits around.
    butt = np.zeros((n, 3), dtype=np.float64)
    butt[:, 2] = 1.2 + 0.05 * np.sin(2.0 * np.pi * t / 0.3)
    angle = np.linspace(-np.pi, 0.0, n)
    clubhead = np.stack(
        [
            0.9 * np.cos(angle),
            0.9 * np.sin(angle),
            butt[:, 2] - 0.3,
        ],
        axis=1,
    )
    # Build unit quaternions by integrating a tiny rotation rate.
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    quat[:, 0] = np.cos(angle / 4.0)
    quat[:, 1] = np.sin(angle / 4.0)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    # Add a touch of jitter so the figure isn't degenerate.
    butt += rng.normal(scale=1e-4, size=butt.shape)
    clubhead += rng.normal(scale=1e-4, size=clubhead.shape)
    provenance = SourceProvenance(
        filename="synthetic.csv",
        format="synthetic",
        subject_id="TEST",
        trial_id="render_swing",
        sha256="0" * 64,
    )
    return ClubTarget(
        time=t,
        butt=butt,
        clubhead=clubhead,
        club_quat=quat,
        impact_idx=int(0.85 * n),
        source=provenance,
    )


def _synthetic_result(target: ClubTarget) -> FitResult:
    """Build a FitResult that drifts slightly from the target."""
    t = target.time.copy()
    grip = target.butt + np.array([2e-3, -1e-3, 5e-4])
    head = target.clubhead + np.array([3e-3, 4e-3, -2e-3])
    quat = target.club_quat.copy()
    # Small angular perturbation to make the orientation panel non-trivial.
    quat[:, 1] += 1e-3
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    # Synthetic torques: 4 actuated joints worth of low-frequency content.
    tau = 0.5 * np.sin(2.0 * np.pi * t[:, None] / 0.3 + np.arange(4))
    return FitResult(
        time=t,
        grip=grip,
        clubhead=head,
        club_quat=quat,
        tau=tau.astype(np.float64),
        solver_status="ok",
        duration_s=4.2,
        metadata={
            "swing_id": "TEST_TW_ProV1",
            "solver": "synthetic",
            "iterations": 17,
            "commit": "abcdef0",
            "branch": "feat/issue-4125",
        },
    )


@pytest.fixture()
def fit_pair(tmp_path):
    """A (result, target, options) triple writing into ``tmp_path``."""
    target = _synthetic_target()
    result = _synthetic_result(target)
    options = VizOptions(output_dir=tmp_path, dpi=120)
    return result, target, options


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_render_trajectory_overlay_writes_png(fit_pair):
    result, target, options = fit_pair
    out = render_trajectory_overlay(result, target, options)
    assert out.suffix == ".png"
    assert out.is_file()
    assert out.stat().st_size > 0
    width, height = _png_dimensions(out)
    # 12.0 in x 6.0 in @ 120 dpi -> ~1440 x ~720 (with bbox-tight padding).
    assert width >= 800
    assert height >= 400


def test_render_error_timecourse_writes_png(fit_pair):
    result, target, options = fit_pair
    out = render_error_timecourse(result, target, options)
    assert out.suffix == ".png"
    assert out.is_file()
    assert out.stat().st_size > 0
    width, height = _png_dimensions(out)
    # 10x10 in @ 120 dpi -> ~1200 x ~1200.
    assert width >= 600
    assert height >= 600


def test_render_fit_quality_card_writes_png(fit_pair):
    result, target, options = fit_pair
    out = render_fit_quality_card(result, target, options)
    assert out.suffix == ".png"
    assert out.is_file()
    assert out.stat().st_size > 0
    width, height = _png_dimensions(out)
    assert width >= 600
    assert height >= 400


def test_render_overlay_honours_explicit_output_path(tmp_path, fit_pair):
    result, target, _ = fit_pair
    explicit = tmp_path / "subdir" / "my_overlay.png"
    options = VizOptions(output_path=explicit, dpi=100)
    out = render_trajectory_overlay(result, target, options)
    assert out == explicit
    assert explicit.is_file()
    assert _png_dimensions(out)[0] > 0


def test_fitresult_from_simout_duck_type():
    """FitResult.from_simout adapts an arbitrary SimOut-shaped object."""

    class _FakeSimOut:
        def __init__(self):
            self.time = np.linspace(0.0, 0.3, 8)
            self.grip = np.zeros((8, 3))
            self.clubhead = np.zeros((8, 3))
            self.club_quat = np.tile([1.0, 0.0, 0.0, 0.0], (8, 1))
            self.tau = np.zeros((8, 3))
            self.solver_status = "ok"
            self.wall_clock_s = 0.123

    fr = FitResult.from_simout(_FakeSimOut(), swing_id="duck")
    assert fr.metadata["swing_id"] == "duck"
    assert fr.tau is not None and fr.tau.shape == (8, 3)
    assert fr.wall_clock_s == pytest.approx(0.123)
