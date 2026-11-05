"""Additivity charts for the pinned WSCG run (Issue #14440).

Plots the canonical pointwise counterfactuals of ``wscg_ztcf_additivity``
(#14438) along the same rollout of the model pinned by #14437: for each lead-arm
joint, ZTCF (the drift), ZVCF (the control component), their sum and the full
response; below them, the additivity residual against its declared tolerance
band. Accelerations are generalized hinge accelerations in rad/s^2, positive
along the MJCF joint axis. Event markers are the peak-speed and peak full
response samples of this torque program; it is not a golf swing, so no swing
event is claimed. The identity is pointwise only. A model-scenario result, not
empirical validation.

The figures are written on demand and are not checked in (a rendered SVG is
over a thousand lines).

Usage::

    python3 -m scripts.research.proximal_distal_energy.wscg_ztcf_additivity_charts
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib as mpl
import mujoco
import numpy as np

from scripts.research.proximal_distal_energy.wscg_model_selection import REPO_ROOT
from scripts.research.proximal_distal_energy.wscg_ztcf_additivity import (
    ABS_TOLERANCE,
    RELATIVE_TOLERANCE,
    components,
    load_pinned_model,
    rollout_states,
)

LEAD_ARM_JOINTS = ("left_shoulder_flexion", "left_elbow", "left_wrist_flexion")
FIGURE_STEM = Path(
    "docs/research/proximal_distal_energy_transfer/figures/fig_wscg_ztcf_additivity"
)
_RC = {"svg.hashsalt": "upstreamdrift-wscg-ztcf-additivity-v1", "font.size": 9}
_UNITS = "rad/s²"
_STYLES = {
    "Full": {"color": "#202020", "linewidth": 2.4},
    "ZTCF": {"color": "#1f77b4", "linewidth": 1.2},
    "ZVCF": {"color": "#d62728", "linewidth": 1.2},
    "ZTCF + ZVCF": {"color": "#f2b134", "linewidth": 1.2, "linestyle": "--"},
}
_EVENT_STYLES = ({"color": "#7f7f7f"}, {"color": "#7f7f7f", "linestyle": ":"})


@dataclass(frozen=True)
class AdditivityChartData:
    """Everything the chart draws: one row per sample, one column per joint."""

    time_s: np.ndarray
    joints: tuple[str, ...]
    full: np.ndarray
    ztcf: np.ndarray
    zvcf: np.ndarray
    residual: np.ndarray
    tolerance: np.ndarray
    events: Mapping[str, float]


def _hinge_dofs(model: mujoco.MjModel, joints: Sequence[str]) -> list[int]:
    dofs = []
    for name in joints:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"{name!r} is not a hinge joint of the pinned model")
        dofs.append(int(model.jnt_dofadr[joint]))
    return dofs


def chart_data(joints: Sequence[str] = LEAD_ARM_JOINTS) -> AdditivityChartData:
    """Counterfactual series of the pinned rollout for ``joints``.

    Preconditions: every name is a hinge joint of the pinned model (else
    ``ValueError``). Postconditions: ``residual`` and ``tolerance`` are the
    #14438 additivity residual and declared bound over all degrees of freedom,
    and ``|full - ztcf - zvcf| <= tolerance`` holds per joint.
    """
    model = load_pinned_model()
    dofs = _hinge_dofs(model, joints)
    probe = mujoco.MjData(model)
    times, speed, rows = [], [], []
    for time_s, q, v, u in rollout_states(model):
        times.append(time_s)
        speed.append(np.abs(v).max())
        rows.append(components(model, probe, q, v, u))
    full, ztcf, zvcf = (
        np.array([r[k] for r in rows]) for k in ("full", "ztcf", "zvcf")
    )
    full_norm = np.abs(full).max(axis=1)
    time = np.array(times)
    return AdditivityChartData(
        time_s=time,
        joints=tuple(joints),
        full=full[:, dofs],
        ztcf=ztcf[:, dofs],
        zvcf=zvcf[:, dofs],
        residual=np.abs(full - ztcf - zvcf).max(axis=1),
        tolerance=ABS_TOLERANCE + RELATIVE_TOLERANCE * full_norm,
        events={
            "Peak Speed": float(time[np.argmax(speed)]),
            "Peak |a_full|": float(time[np.argmax(full_norm)]),
        },
    )


def _mark_events(ax, events: Mapping[str, float]) -> None:
    for (name, time_s), style in zip(events.items(), _EVENT_STYLES, strict=True):
        ax.axvline(time_s, label=name, linewidth=0.9, **style)


def draw_chart(data: AdditivityChartData) -> Figure:
    """One panel per joint plus the residual panel, on an Agg canvas."""
    with mpl.rc_context(_RC):
        fig = Figure(figsize=(7.0, 2.0 * (len(data.joints) + 1)), dpi=150)
        FigureCanvasAgg(fig)
        axes = fig.subplots(len(data.joints) + 1, 1, sharex=True)
        for column, (joint, ax) in enumerate(zip(data.joints, axes, strict=False)):
            series = {
                "Full": data.full[:, column],
                "ZTCF": data.ztcf[:, column],
                "ZVCF": data.zvcf[:, column],
                "ZTCF + ZVCF": data.ztcf[:, column] + data.zvcf[:, column],
            }
            for label, values in series.items():
                ax.plot(data.time_s, values, label=label, **_STYLES[label])
            _mark_events(ax, data.events)
            ax.set_title(f"Joint: {joint}")
            ax.set_ylabel(f"Acceleration ({_UNITS})")
        axes[0].legend(loc="upper left", ncols=3, fontsize=7)
        residual_ax = axes[-1]
        residual_ax.fill_between(
            data.time_s, data.tolerance, color="#c7e9c0", label="Declared Tolerance"
        )
        residual_ax.plot(data.time_s, data.residual, color="#202020", label="Residual")
        _mark_events(residual_ax, data.events)
        residual_ax.set_yscale("log")
        residual_ax.set_title(
            "Additivity Residual |a_full - a_ZTCF - a_ZVCF| (All DOFs)"
        )
        residual_ax.set_ylabel(f"Residual ({_UNITS})")
        residual_ax.set_xlabel("Time (s)")
        residual_ax.legend(loc="upper left", fontsize=7)
        fig.suptitle("ZTCF + ZVCF Versus the Full Response (Pinned WSCG Run)")
        fig.tight_layout()
    return fig


def write_charts(data: AdditivityChartData, stem: Path) -> tuple[Path, Path]:
    """Write ``stem.svg`` and ``stem.png``; the same data gives the same bytes."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig = draw_chart(data)
    svg, png = stem.with_suffix(".svg"), stem.with_suffix(".png")
    with mpl.rc_context(_RC):
        fig.savefig(svg, format="svg", metadata={"Date": None})
        fig.savefig(png, format="png", metadata={"Software": None})
    return svg, png


def main() -> None:
    for path in write_charts(chart_data(), REPO_ROOT / FIGURE_STEM):
        print(f"wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
