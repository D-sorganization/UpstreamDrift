"""Tests for Pinocchio visualisation + leaderboard JSON writer (issue #4133).

The leaderboard tests run unit-style without ``pinocchio`` installed; the
viz smoke tests are gated on ``requires_pinocchio`` so CI tiers without
the engine extras simply skip them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the engine python tree is importable as a top-level package; the
# repo's primary ``src`` package layout otherwise hides it from collection.
_PIN_PY = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "engines"
    / "physics_engines"
    / "pinocchio"
    / "python"
)
if str(_PIN_PY) not in sys.path:
    sys.path.insert(0, str(_PIN_PY))

# Headless-safe matplotlib backend before any module imports it.
import matplotlib  # noqa: E402

matplotlib.use("Agg")

from motion_matching import FitResult, write_leaderboard_entry  # noqa: E402
from motion_matching.leaderboard_writer import SCHEMA_KEYS  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_FRAMES = 64


class _FakeTarget:
    """ClubTargetLike duck-type with the minimum surface viz needs."""

    def __init__(self, n: int = N_FRAMES) -> None:
        t = np.linspace(0.0, 0.3, n)
        self.time = t
        # A simple arc for the clubhead, near-straight for the butt.
        self.butt = np.column_stack([0.05 * np.sin(t * 10), 0.0 * t, 1.0 + 0.0 * t])
        self.clubhead = np.column_stack(
            [0.5 * np.sin(t * 10), 0.5 * np.cos(t * 10), 0.5 + 0.0 * t]
        )
        # Identity quaternion for every frame, wxyz.
        self.club_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
        self.impact_idx = n // 2


@pytest.fixture
def fake_target() -> _FakeTarget:
    return _FakeTarget()


@pytest.fixture
def fake_result(fake_target: _FakeTarget) -> FitResult:
    rng = np.random.default_rng(seed=0)
    n = fake_target.time.shape[0]
    return FitResult(
        trial_id="TW_ProV1_test",
        solver="lm-analytical-jacobian",
        butt_sim=fake_target.butt
        + rng.normal(scale=0.001, size=fake_target.butt.shape),
        clubhead_sim=fake_target.clubhead
        + rng.normal(scale=0.002, size=fake_target.clubhead.shape),
        club_quat_sim=fake_target.club_quat.copy(),
        time=fake_target.time.copy(),
        joint_torques=rng.normal(size=(n, 4)),
        clubhead_speed_mph=np.linspace(0.0, 110.0, n),
        grip_rmse_mm=1.8,
        clubhead_rmse_mm=2.3,
        orientation_rmse_deg=0.41,
        total_work_J=284.0,
        wall_clock_s=4.2,
        n_iterations=247,
        commit="abc1234",
    )


# ---------------------------------------------------------------------------
# Leaderboard writer — schema/round-trip (no pinocchio needed)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_leaderboard_writes_canonical_schema(
    fake_target: _FakeTarget, fake_result: FitResult, tmp_path: Path
) -> None:
    out = write_leaderboard_entry(fake_result, fake_target, tmp_path)
    assert out.exists()
    assert out.name == "pinocchio.json"
    assert out.parent.name == "TW_ProV1_test"

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(payload.keys()) == set(SCHEMA_KEYS)
    assert payload["engine"] == "pinocchio"
    assert payload["trial"] == "TW_ProV1_test"
    assert payload["solver"] == "lm-analytical-jacobian"
    assert payload["grip_rmse_mm"] == pytest.approx(1.8)
    assert payload["clubhead_rmse_mm"] == pytest.approx(2.3)
    assert payload["total_work_J"] == pytest.approx(284.0)
    assert payload["wall_clock_s"] == pytest.approx(4.2)
    assert payload["commit"] == "abc1234"
    # ISO-8601 datetime sanity check (parses without exception).
    from datetime import datetime

    datetime.fromisoformat(payload["run_at"])


@pytest.mark.unit
def test_leaderboard_round_trip(
    fake_target: _FakeTarget, fake_result: FitResult, tmp_path: Path
) -> None:
    out = write_leaderboard_entry(fake_result, fake_target, tmp_path)
    payload = json.loads(out.read_text(encoding="utf-8"))
    # Numeric fields must round-trip exactly through JSON.
    for k in ("grip_rmse_mm", "clubhead_rmse_mm", "total_work_J", "wall_clock_s"):
        assert isinstance(payload[k], float)


@pytest.mark.unit
def test_leaderboard_rejects_non_finite(
    fake_target: _FakeTarget, fake_result: FitResult, tmp_path: Path
) -> None:
    fake_result.grip_rmse_mm = float("nan")
    with pytest.raises(ValueError, match="grip_rmse_mm"):
        write_leaderboard_entry(fake_result, fake_target, tmp_path)


@pytest.mark.unit
def test_leaderboard_rejects_empty_trial(
    fake_target: _FakeTarget, fake_result: FitResult, tmp_path: Path
) -> None:
    fake_result.trial_id = "  "
    with pytest.raises(ValueError, match="trial_id"):
        write_leaderboard_entry(fake_result, fake_target, tmp_path)


@pytest.mark.unit
def test_leaderboard_resolves_commit_when_unknown(
    fake_target: _FakeTarget, fake_result: FitResult, tmp_path: Path
) -> None:
    """An empty/unknown commit triggers a best-effort git lookup or 'unknown'."""
    fake_result.commit = ""
    out = write_leaderboard_entry(fake_result, fake_target, tmp_path)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(payload["commit"], str)
    assert payload["commit"]  # never empty


# ---------------------------------------------------------------------------
# Viz smoke tests — require_pinocchio so non-engine CI lanes skip cleanly.
# ---------------------------------------------------------------------------


def _skip_if_no_pinocchio() -> None:
    """Skip the calling test if the Pinocchio bindings are not importable."""
    try:
        import pinocchio  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"pinocchio not installed: {exc}")


@pytest.mark.requires_pinocchio
@pytest.mark.unit
def test_plot_trajectory_overlay_smoke(
    fake_target: _FakeTarget, fake_result: FitResult, tmp_path: Path
) -> None:
    _skip_if_no_pinocchio()
    from motion_matching.viz import plot_trajectory_overlay

    out = tmp_path / "traj.png"
    fig = plot_trajectory_overlay(fake_target, fake_result, out_path=out)
    assert out.exists() and out.stat().st_size > 0
    import matplotlib.pyplot as plt

    plt.close(fig)


@pytest.mark.requires_pinocchio
@pytest.mark.unit
def test_plot_error_timecourse_smoke(
    fake_target: _FakeTarget, fake_result: FitResult, tmp_path: Path
) -> None:
    _skip_if_no_pinocchio()
    from motion_matching.viz import plot_error_timecourse

    out = tmp_path / "err.png"
    fig = plot_error_timecourse(fake_target, fake_result, out_path=out)
    assert out.exists() and out.stat().st_size > 0
    import matplotlib.pyplot as plt

    plt.close(fig)


@pytest.mark.requires_pinocchio
@pytest.mark.unit
def test_plot_fit_quality_card_smoke(
    fake_target: _FakeTarget, fake_result: FitResult, tmp_path: Path
) -> None:
    _skip_if_no_pinocchio()
    from motion_matching.viz import plot_fit_quality_card

    out = tmp_path / "card.png"
    fig = plot_fit_quality_card(fake_target, fake_result, out_path=out)
    assert out.exists() and out.stat().st_size > 0
    import matplotlib.pyplot as plt

    plt.close(fig)


@pytest.mark.requires_pinocchio
@pytest.mark.unit
def test_visualize_fit_emits_three_views(
    fake_target: _FakeTarget, fake_result: FitResult, tmp_path: Path
) -> None:
    _skip_if_no_pinocchio()
    from motion_matching.viz import visualize_fit

    artefacts = visualize_fit(fake_target, fake_result, out_dir=tmp_path)
    assert {"trajectory_overlay", "error_timecourse", "fit_quality_card"} <= set(
        artefacts.keys()
    )
    for key in ("trajectory_overlay", "error_timecourse", "fit_quality_card"):
        path = artefacts[key]
        assert isinstance(path, Path) and path.exists() and path.stat().st_size > 0
