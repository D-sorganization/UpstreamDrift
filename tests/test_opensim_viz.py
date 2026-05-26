"""Smoke tests for the three canonical OpenSim visualisation figures (#4130).

The matplotlib paths are headless-safe and run unconditionally — they do
not import the ``opensim`` wheel, only ``matplotlib`` + ``numpy``. The
optional ``opensim.Visualizer`` wrapper is gated by the
``requires_opensim`` marker and is skipped when the wheel is missing.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from src.engines.physics_engines.opensim.python.motion_matching.viz import (  # noqa: E402
    plot_error_timecourse,
    plot_fit_quality_card,
    plot_trajectory_overlay,
    render_with_opensim_visualizer,
)

# ---------------------------------------------------------------------------
# Fixtures: deterministic synthetic target and sim-output stand-ins.
# ---------------------------------------------------------------------------


def _synthetic_target(n: int = 64) -> SimpleNamespace:
    """Build a deterministic, valid target-shaped namespace.

    Smooth helical clubhead path so the figures have non-trivial content
    and the finite-difference speed channel is well-defined.
    """
    t = np.linspace(0.0, 0.3, n)
    omega = 2 * np.pi * 3.0
    radius = 0.6
    clubhead = np.column_stack(
        [
            radius * np.cos(omega * t),
            radius * np.sin(omega * t),
            0.5 + 0.1 * t,
        ]
    )
    butt = clubhead - np.array([0.0, 0.0, 1.1])  # rigid offset along z
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    return SimpleNamespace(
        time=t,
        clubhead=clubhead,
        butt=butt,
        club_quat=quat,
        impact_idx=int(0.85 * n),
    )


def _synthetic_sim(
    target: SimpleNamespace,
    *,
    drift_mm: float = 5.0,
    with_torques: bool = True,
) -> SimpleNamespace:
    """Build a sim-output namespace that drifts ``drift_mm`` from the target."""
    n = target.time.size
    drift = (drift_mm / 1000.0) * np.linspace(0.0, 1.0, n)[:, None]
    clubhead = target.clubhead + drift * np.array([1.0, 0.0, 0.0])
    butt = target.butt + drift * np.array([1.0, 0.0, 0.0])
    quat = target.club_quat.copy()
    # Slight rotation drift for the orientation panel.
    angle_rad = np.deg2rad(np.linspace(0.0, 0.5, n))
    quat[:, 0] = np.cos(angle_rad / 2.0)
    quat[:, 3] = np.sin(angle_rad / 2.0)
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)

    torques = None
    if with_torques:
        rng = np.random.default_rng(seed=0)
        torques = rng.normal(scale=2.0, size=(n, 5))

    return SimpleNamespace(
        time=target.time,
        clubhead=clubhead,
        butt=butt,
        club_quat=quat,
        joint_torques=torques,
        impact_idx=target.impact_idx,
    )


# ---------------------------------------------------------------------------
# View 1 — trajectory overlay (matplotlib path).
# ---------------------------------------------------------------------------


def test_plot_trajectory_overlay_matplotlib_returns_figure() -> None:
    """View 1 must produce a Figure with two 3-D axes."""
    target = _synthetic_target()
    sim = _synthetic_sim(target)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # promote warnings to errors
        fig = plot_trajectory_overlay(target, sim)

    assert fig is not None
    # Two 3-D axes (left=measured, right=simulated).
    axes_3d = [ax for ax in fig.axes if getattr(ax, "name", "") == "3d"]
    assert len(axes_3d) == 2, f"expected two 3-D axes, found {len(axes_3d)}"
    plt.close(fig)


def test_plot_trajectory_overlay_handles_missing_butt() -> None:
    """Sim outputs without a butt series should still render the clubhead."""
    target = _synthetic_target()
    sim = _synthetic_sim(target)
    sim.butt = None

    fig = plot_trajectory_overlay(target, sim)
    assert fig is not None
    plt.close(fig)


def test_plot_trajectory_overlay_rejects_bad_shape() -> None:
    """Mismatched clubhead shape must raise ValueError."""
    target = _synthetic_target()
    bad_sim = SimpleNamespace(
        time=target.time,
        clubhead=np.zeros((target.time.size, 2)),  # wrong: 2 cols
    )
    with pytest.raises(ValueError, match="clubhead"):
        plot_trajectory_overlay(target, bad_sim)


# ---------------------------------------------------------------------------
# View 2 — error timecourse.
# ---------------------------------------------------------------------------


def test_plot_error_timecourse_full_panels() -> None:
    """All four panels (position, orientation, speed, torques) render."""
    target = _synthetic_target()
    sim = _synthetic_sim(target, with_torques=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fig = plot_error_timecourse(target, sim)

    # Four panels expected since target+sim both expose quaternions and
    # the sim carries joint torques.
    assert len(fig.axes) == 4
    plt.close(fig)


def test_plot_error_timecourse_drops_optional_panels() -> None:
    """Sim outputs without torques or quats produce a smaller figure."""
    target = _synthetic_target()
    sim = _synthetic_sim(target, with_torques=False)
    sim.club_quat = None

    fig = plot_error_timecourse(target, sim)
    # Only position + speed remain (no orientation, no torques).
    assert len(fig.axes) == 2
    plt.close(fig)


def test_plot_error_timecourse_position_units_are_mm() -> None:
    """A 5 mm constant drift in clubhead must show as 5 mm at t_end."""
    target = _synthetic_target()
    sim = _synthetic_sim(target, drift_mm=5.0, with_torques=False)
    sim.club_quat = None

    fig = plot_error_timecourse(target, sim)
    pos_ax = fig.axes[0]
    # Find the clubhead trace (orange) and assert peak ≈ 5 mm.
    head_lines = [ln for ln in pos_ax.lines if ln.get_label() == "clubhead"]
    assert head_lines, "clubhead error trace not found"
    y = head_lines[0].get_ydata()
    assert float(np.max(y)) == pytest.approx(5.0, abs=0.5)
    plt.close(fig)


# ---------------------------------------------------------------------------
# View 3 — fit quality card.
# ---------------------------------------------------------------------------


def test_plot_fit_quality_card_renders_full_summary() -> None:
    fit_result = SimpleNamespace(
        swing_id="TW_ProV1",
        solver="L-BFGS-B",
        iterations=247,
        wall_clock="4m 12s",
        rmse_clubhead_mm=2.3,
        rmse_butt_mm=1.8,
        mean_orientation_error_deg=0.41,
        clubhead_speed_at_impact_mph=112,
        measured_clubhead_speed_at_impact_mph=111,
        total_work_J=284,
        peak_joint_power_kW=1.2,
        commit="7a3fdeadbeef",
        branch="feat/issue-4130",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fig = plot_fit_quality_card(fit_result)

    assert fig is not None
    # The card body lives in a single axes with all text.
    text_blob = "".join(t.get_text() for ax in fig.axes for t in ax.texts)
    assert "TW_ProV1" in text_blob
    assert "L-BFGS-B" in text_blob
    assert "2.30 mm" in text_blob
    assert "0.41" in text_blob
    plt.close(fig)


def test_plot_fit_quality_card_handles_minimal_input() -> None:
    """Missing fields must not raise — the card degrades gracefully."""
    fig = plot_fit_quality_card({})
    text_blob = "".join(t.get_text() for ax in fig.axes for t in ax.texts)
    assert "no metrics provided" in text_blob
    plt.close(fig)


def test_plot_fit_quality_card_accepts_dict_input() -> None:
    """Pure ``dict`` inputs (loaded from JSON) work identically."""
    fig = plot_fit_quality_card({"swing_id": "demo", "rmse_clubhead_mm": 3.7})
    text_blob = "".join(t.get_text() for ax in fig.axes for t in ax.texts)
    assert "demo" in text_blob
    assert "3.70 mm" in text_blob
    plt.close(fig)


# ---------------------------------------------------------------------------
# Smoke test for the OpenSim Visualizer path (gated).
# ---------------------------------------------------------------------------


@pytest.mark.requires_opensim
def test_render_with_opensim_visualizer_requires_model() -> None:
    """Without a model the helper must raise a clear ``ValueError``."""
    sim = SimpleNamespace(time=np.array([0.0, 0.1]), states=np.zeros((2, 1)))
    with pytest.raises(ValueError, match="opensim.Model"):
        render_with_opensim_visualizer(model=None, sim_out=sim)


def test_render_with_opensim_visualizer_raises_without_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the bindings are missing the helper raises ``RuntimeError``.

    This always-on test fakes the import failure so it runs even on
    machines that have ``opensim`` installed.
    """
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "opensim":
            raise ImportError("simulated missing opensim")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sim = SimpleNamespace(time=np.array([0.0, 0.1]), states=np.zeros((2, 1)))

    with pytest.raises(RuntimeError, match="opensim Python bindings"):
        render_with_opensim_visualizer(model=object(), sim_out=sim)


def test_trajectory_overlay_visualizer_path_is_opt_in() -> None:
    """``use_opensim_visualizer=True`` returns the placeholder figure when
    the bindings are missing — and never auto-launches the Visualizer."""
    target = _synthetic_target()
    sim = _synthetic_sim(target)

    # Default keeps the matplotlib path: two 3-D axes.
    fig = plot_trajectory_overlay(target, sim)
    axes_3d = [ax for ax in fig.axes if getattr(ax, "name", "") == "3d"]
    assert len(axes_3d) == 2
    plt.close(fig)
