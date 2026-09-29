#!/usr/bin/env python3
"""Demo: match measured club motion using the NN swing surrogate.

This script demonstrates end-to-end surrogate-based motion matching without
requiring a Simulink license.  It:

1. Loads the production surrogate (``data/training/surrogate_v4/checkpoint_production.pt``).
2. Loads the canonical Wiffle ProV1 target from the Excel file.
3. Extracts the 31-sample impact window (0.30 s at ~100 Hz, centred on impact).
4. Normalises both target and surrogate output into a common relative frame
   (grip centroid → origin, shaft length → unit scale).
5. Runs ``scipy.optimize.minimize`` (L-BFGS-B) to find the polynomial
   coefficient vector ``theta`` (189-dim) that minimises grip + clubhead RMSE.
6. Prints a quality summary and saves a trajectory comparison figure.

Usage::

    cd <repository root>
    python3 scripts/demo_surrogate_match.py

Optional arguments::

    --target   <path to Wiffle xlsx>   default: canonical ProV1 file
    --sheet    <sheet name>            default: TW_ProV1
    --ckpt     <surrogate .pt path>    default: data/training/surrogate_v4/checkpoint_production.pt
    --out      <output directory>      default: output/surrogate_match
    --max-iter <int>                   default: 200
    --seed     <int>                   default: 42
    --no-plot                          skip figure generation
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.optimize as sciopt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.python.motion_matching.surrogate.compact.model import (  # noqa: E402
    COEFF_BOUNDS,
    COEFFS_PER_JOINT,
    N_JOINTS_DEFAULT,
    CoeffNormalizer,
    SwingSurrogate,
)
from src.shared.python.motion_matching.surrogate.compact.predict import (  # noqa: E402
    predict_trajectory,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
_LOG = logging.getLogger("demo_surrogate_match")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonical Wiffle ProV1 file (relative to repo root)
_DEFAULT_TARGET = (
    "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/"
    "src/apps/golf_gui/Motion Capture Plotter/Wiffle_ProV1_club_3D_data.xlsx"
)
_DEFAULT_CKPT = "data/training/surrogate_v4/checkpoint_production.pt"

#: cm → metres conversion factor used by the MATLAB ``load_club_target_excel.m``
CM_TO_M: float = 0.01

#: Number of timesteps in the compact dataset / surrogate output
T_SURROGATE: int = 31

#: Impact sample numbers (1-based, from row-0 of the ProV1 sheet)
_EVENT_IMPACT_SAMPLE: int = 525  # I column in row 0


# ---------------------------------------------------------------------------
# Target loading
# ---------------------------------------------------------------------------


def load_wiffle_target(xlsx_path: Path, sheet_name: str = "TW_ProV1") -> dict:
    """Parse a Wiffle-format Excel file into a canonical target dict.

    Returns a dict with keys:
        ``time``        (N,) seconds, monotonic from 0
        ``grip``        (N, 3) metres (mid-hands position)
        ``clubhead``    (N, 3) metres (club-face centre)
        ``impact_idx``  int, 0-based index of max-CHS sample
        ``chs_mph``     float, clubhead speed at impact
    """
    xl = pd.ExcelFile(xlsx_path)
    if sheet_name not in xl.sheet_names:
        raise ValueError(
            f"Sheet '{sheet_name}' not found in {xlsx_path}; "
            f"available: {xl.sheet_names}"
        )

    # --- parse event markers from row 0 ---
    raw0 = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None, nrows=1)
    row0 = raw0.iloc[0].tolist()
    impact_sample = None
    chs_mph = None
    for idx, val in enumerate(row0):
        if val == "I" and idx + 1 < len(row0):
            impact_sample = int(row0[idx + 1])
        if val == "CHS" and idx + 1 < len(row0):
            chs_mph = float(row0[idx + 1])

    if impact_sample is None:
        impact_sample = _EVENT_IMPACT_SAMPLE
        _LOG.warning(
            "Impact marker 'I' not found in row 0; defaulting to %d", impact_sample
        )

    # --- read data rows (skip rows 0,1 which are headers/labels; row 2 = col names) ---
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=2, skiprows=[3])
    df = df.dropna(subset=["Time"])

    time_s = df["Time"].to_numpy(dtype=np.float64)
    # Shift so first sample is t=0
    time_s = time_s - time_s[0]

    # Mid-hands (grip) — columns X, Y, Z (cols index 2,3,4 = 'X','Y','Z')
    grip = df[["X", "Y", "Z"]].to_numpy(dtype=np.float64) * CM_TO_M

    # Club-face centre (clubhead) — columns 'X.1','Y.1','Z.1' (cols 14,15,16)
    clubhead = df[["X.1", "Y.1", "Z.1"]].to_numpy(dtype=np.float64) * CM_TO_M

    # Convert 1-based sample to 0-based index
    impact_idx = impact_sample - 1

    return {
        "time": time_s,
        "grip": grip,
        "clubhead": clubhead,
        "impact_idx": impact_idx,
        "chs_mph": chs_mph,
    }


def extract_impact_window(target: dict, n_out: int = T_SURROGATE) -> dict:
    """Extract and resample the impact window to ``n_out`` uniformly-spaced samples.

    The window is centred on ``target['impact_idx']`` and spans exactly the
    duration needed to produce ``n_out`` samples at the same cadence as the
    compact dataset (~100 Hz over 0.30 s).

    Returns a copy of ``target`` with arrays resampled to shape ``(n_out, ...)``.
    """
    grip_full = target["grip"]
    club_full = target["clubhead"]
    time_full = target["time"]
    n_full = len(time_full)
    imp = target["impact_idx"]

    # The compact dataset covers ~0.30 s around impact at ~100 Hz.
    # In the 240-Hz Wiffle data, 0.30 s = 72 samples, centred on impact.
    half_window_240 = 36  # ±36 samples at 240 Hz → 0.30 s total
    i_start = max(0, imp - half_window_240)
    i_end = min(n_full, imp + half_window_240 + 1)

    grip_win = grip_full[i_start:i_end]
    club_win = club_full[i_start:i_end]
    time_win = time_full[i_start:i_end] - time_full[i_start]

    # Resample to exactly n_out samples via linear interpolation
    t_uniform = np.linspace(0.0, time_win[-1], n_out)
    grip_resampled = np.stack(
        [np.interp(t_uniform, time_win, grip_win[:, c]) for c in range(3)], axis=-1
    )
    club_resampled = np.stack(
        [np.interp(t_uniform, time_win, club_win[:, c]) for c in range(3)], axis=-1
    )
    new_impact_idx = np.argmin(
        np.abs(t_uniform - (time_full[imp] - time_full[i_start]))
    )

    return {
        "time": t_uniform,
        "grip": grip_resampled,
        "clubhead": club_resampled,
        "impact_idx": int(new_impact_idx),
        "chs_mph": target["chs_mph"],
    }


# ---------------------------------------------------------------------------
# Coordinate normalisation
# ---------------------------------------------------------------------------


def normalise_target(target: dict) -> tuple[dict, dict]:
    """Shift target to a surrogate-compatible relative frame.

    The surrogate was trained in Simscape world-frame coordinates that differ
    from the GEARS/Vicon frame.  Instead of a full 6-DOF ICP alignment (which
    requires MATLAB), we centre the trajectory on the grip mid-point and scale
    by the shaft length.  Both the target and any surrogate prediction can be
    compared in this normalised space.

    Returns ``(normalised_target, normalise_params)`` where
    ``normalise_params`` carries the inverse transform.
    """
    grip = target["grip"]  # (T, 3)
    club = target["clubhead"]  # (T, 3)

    # Origin: mean grip position over the window
    grip_origin = grip.mean(axis=0)

    # Scale: shaft length at impact (should be ~1.07 m for a typical iron)
    imp = target["impact_idx"]
    shaft_vec = club[imp] - grip[imp]
    shaft_len = float(np.linalg.norm(shaft_vec))

    norm_grip = (grip - grip_origin) / shaft_len
    norm_club = (club - grip_origin) / shaft_len

    normalised = dict(target)
    normalised["grip"] = norm_grip
    normalised["clubhead"] = norm_club
    normalised["shaft_len_m"] = shaft_len
    normalised["origin_m"] = grip_origin

    params = {"origin": grip_origin, "shaft_len": shaft_len}
    return normalised, params


def normalise_prediction(pred: dict, params: dict) -> dict:
    """Normalise a ``predict_trajectory`` output dict to the target frame."""
    shaft_len = params["shaft_len"]
    grip_pred = pred["r_grip"][0]  # (T, 3), remove batch dim
    club_pred = pred["r_clubhead"][0]  # (T, 3)

    # Surrogate grip origin
    grip_origin_pred = grip_pred.mean(axis=0)

    norm_grip = (grip_pred - grip_origin_pred) / shaft_len
    norm_club = (club_pred - grip_origin_pred) / shaft_len

    return {"grip": norm_grip, "clubhead": norm_club}


# ---------------------------------------------------------------------------
# Cost function
# ---------------------------------------------------------------------------


def make_cost_fn(
    model: SwingSurrogate,
    target_norm: dict,
    *,
    w_grip: float = 1.0,
    w_club: float = 0.5,
    w_speed: float = 0.05,
    history: list[float] | None = None,
) -> callable:
    """Return a scipy-compatible cost function ``J(theta_flat) -> float``.

    Args:
        model: Loaded surrogate model.
        target_norm: Normalised target dict.
        w_grip: Weight for grip position MSE.
        w_club: Weight for clubhead position MSE.
        w_speed: Weight for relative clubhead-speed error at impact.
        history: Optional list to append per-call costs to.

    Returns:
        Callable ``J(theta_flat) -> float`` where ``theta_flat`` is the raw
        189-dim coefficient vector in physical units.
    """
    grip_tgt = torch.as_tensor(target_norm["grip"], dtype=torch.float32)  # (T, 3)
    club_tgt = torch.as_tensor(target_norm["clubhead"], dtype=torch.float32)  # (T, 3)
    normaliser: CoeffNormalizer = model.coeff_normalizer  # type: ignore[attr-defined]
    params_cache: dict = {}

    def J(theta_flat: np.ndarray) -> float:
        theta_t = torch.as_tensor(theta_flat, dtype=torch.float32).unsqueeze(0)
        theta_norm = normaliser.normalize(theta_t)  # (1, 189) in [-1, 1]

        with torch.no_grad():
            raw = model.forward(theta_norm)  # (1, T, 12)

        # r_grip: channels 6–9, r_clubhead: channels 0–3, clubhead_speed: channel 9
        r_grip_pred = raw[0, :, 6:9]  # (T, 3)
        r_club_pred = raw[0, :, 0:3]  # (T, 3)
        chs_pred = raw[0, :, 9]  # (T,)  mph

        # Normalise prediction to common frame (grip-centred, shaft-scaled)
        grip_origin_pred = r_grip_pred.mean(dim=0)
        shaft_len = params_cache.get("shaft_len_t")
        if shaft_len is None:
            shaft_len = torch.tensor(target_norm["shaft_len_m"], dtype=torch.float32)
            params_cache["shaft_len_t"] = shaft_len

        grip_n = (r_grip_pred - grip_origin_pred) / shaft_len
        club_n = (r_club_pred - grip_origin_pred) / shaft_len

        grip_mse = ((grip_n - grip_tgt) ** 2).mean()
        club_mse = ((club_n - club_tgt) ** 2).mean()

        # Relative CHS error at impact
        imp_idx = target_norm["impact_idx"]
        chs_tgt_val = target_norm.get("chs_mph", 114.5)
        chs_err = ((chs_pred[imp_idx] - chs_tgt_val) / (chs_tgt_val + 1e-6)) ** 2

        cost = float(w_grip * grip_mse + w_club * club_mse + w_speed * chs_err)
        if history is not None:
            history.append(cost)
        return cost

    return J


def cost_and_grad(
    theta_flat: np.ndarray,
    model: SwingSurrogate,
    target_norm: dict,
    *,
    w_grip: float = 1.0,
    w_club: float = 0.5,
    w_speed: float = 0.05,
    history: list[float] | None = None,
) -> tuple[float, np.ndarray]:
    """Return (cost, gradient) for L-BFGS-B with autograd through the surrogate."""
    normaliser: CoeffNormalizer = model.coeff_normalizer  # type: ignore[attr-defined]
    theta_t = (
        torch.as_tensor(theta_flat, dtype=torch.float32)
        .unsqueeze(0)
        .requires_grad_(True)
    )
    theta_norm = normaliser.normalize(theta_t)

    raw = model.forward(theta_norm)  # (1, T, 12)

    r_grip_pred = raw[0, :, 6:9]
    r_club_pred = raw[0, :, 0:3]
    chs_pred = raw[0, :, 9]

    grip_tgt = torch.as_tensor(target_norm["grip"], dtype=torch.float32)
    club_tgt = torch.as_tensor(target_norm["clubhead"], dtype=torch.float32)
    shaft_len = torch.tensor(target_norm["shaft_len_m"], dtype=torch.float32)
    grip_origin_pred = r_grip_pred.mean(dim=0)
    grip_n = (r_grip_pred - grip_origin_pred) / shaft_len
    club_n = (r_club_pred - grip_origin_pred) / shaft_len

    grip_mse = ((grip_n - grip_tgt) ** 2).mean()
    club_mse = ((club_n - club_tgt) ** 2).mean()

    imp_idx = target_norm["impact_idx"]
    chs_tgt_val = float(target_norm.get("chs_mph", 114.5))
    chs_err = ((chs_pred[imp_idx] - chs_tgt_val) / (chs_tgt_val + 1e-6)) ** 2

    loss = w_grip * grip_mse + w_club * club_mse + w_speed * chs_err
    loss.backward()

    cost = float(loss.item())
    grad = theta_t.grad.detach().numpy().flatten().astype(np.float64)

    if history is not None:
        history.append(cost)

    return cost, grad


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


def build_scipy_bounds(n_joints: int = N_JOINTS_DEFAULT) -> sciopt.Bounds:
    """Build scipy bounds for the physical-unit theta vector (189-dim)."""
    lb_list: list[float] = []
    ub_list: list[float] = []
    for _ in range(n_joints):
        for bound in COEFF_BOUNDS:
            lb_list.append(-bound)
            ub_list.append(bound)
    return sciopt.Bounds(lb=np.array(lb_list), ub=np.array(ub_list))


# ---------------------------------------------------------------------------
# Main optimisation loop
# ---------------------------------------------------------------------------


def run_match(
    target_xlsx: Path,
    ckpt_path: Path,
    *,
    sheet_name: str = "TW_ProV1",
    max_iter: int = 200,
    seed: int = 42,
    out_dir: Path | None = None,
    plot: bool = True,
) -> dict:
    """Full surrogate-based motion matching pipeline.

    Returns a results dict with keys:
        ``theta``           (189,) optimised coefficients
        ``cost_final``      scalar final cost
        ``grip_rmse_norm``  grip RMSE in shaft-length units (≤0.02 = good)
        ``club_rmse_norm``  clubhead RMSE in shaft-length units
        ``chs_mph_pred``    predicted clubhead speed at impact (mph)
        ``history``         list of per-evaluation costs
        ``elapsed_s``       wall-clock time
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    # 1. Load surrogate
    _LOG.info("Loading surrogate from %s", ckpt_path)
    model = SwingSurrogate.from_checkpoint(ckpt_path)
    model.eval()
    _LOG.info("Surrogate loaded: %d params", model.parameter_count())

    # 2. Load and preprocess target
    _LOG.info("Loading target from %s [sheet=%s]", target_xlsx, sheet_name)
    raw_target = load_wiffle_target(target_xlsx, sheet_name)
    _LOG.info(
        "Loaded %d samples, impact at idx=%d, CHS=%.1f mph",
        len(raw_target["time"]),
        raw_target["impact_idx"],
        raw_target["chs_mph"] or 0.0,
    )

    win_target = extract_impact_window(raw_target)
    _LOG.info(
        "Impact window: %d samples, t=[%.3f, %.3f] s, impact@idx=%d",
        len(win_target["time"]),
        win_target["time"][0],
        win_target["time"][-1],
        win_target["impact_idx"],
    )

    target_norm, norm_params = normalise_target(win_target)
    _LOG.info(
        "Shaft length at impact: %.4f m  |  grip origin: %s",
        norm_params["shaft_len"],
        np.array2string(norm_params["origin"], precision=4),
    )

    # 3. Build bounds and initial guess
    bounds = build_scipy_bounds()
    # Start at 1% of bound range — small enough to avoid unphysical sims
    theta0 = (
        rng.uniform(-1, 1, size=N_JOINTS_DEFAULT * COEFFS_PER_JOINT)
        * np.array([b for _ in range(N_JOINTS_DEFAULT) for b in COEFF_BOUNDS])
        * 0.01
    )

    # 4. Optimise with L-BFGS-B using autograd through the surrogate
    _LOG.info("Starting L-BFGS-B optimisation (max_iter=%d) ...", max_iter)
    history: list[float] = []

    def fg(theta_flat: np.ndarray) -> tuple[float, np.ndarray]:
        return cost_and_grad(theta_flat, model, target_norm, history=history)

    t_opt0 = time.perf_counter()
    opt_result = sciopt.minimize(
        fg,
        theta0,
        method="L-BFGS-B",
        jac=True,
        bounds=[(lb, ub) for lb, ub in zip(bounds.lb, bounds.ub, strict=False)],
        options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-8, "disp": False},
    )
    t_opt1 = time.perf_counter()

    theta_opt = opt_result.x
    _LOG.info(
        "Optimisation finished: success=%s, %d iters, %d fevals, final cost=%.6f, "
        "wall=%.1f s",
        opt_result.success,
        opt_result.nit,
        opt_result.nfev,
        float(opt_result.fun),
        t_opt1 - t_opt0,
    )

    # 5. Evaluate final prediction
    pred = predict_trajectory(model, theta_opt)
    pred_norm = normalise_prediction(pred, norm_params)

    grip_rmse = float(np.sqrt(np.mean((pred_norm["grip"] - target_norm["grip"]) ** 2)))
    club_rmse = float(
        np.sqrt(np.mean((pred_norm["clubhead"] - target_norm["clubhead"]) ** 2))
    )
    imp = target_norm["impact_idx"]
    chs_pred = float(pred["clubhead_speed"][0, imp])
    chs_tgt = float(target_norm.get("chs_mph", 114.5))

    _LOG.info("Grip RMSE (shaft-lengths): %.5f", grip_rmse)
    _LOG.info("Clubhead RMSE (shaft-lengths): %.5f", club_rmse)
    _LOG.info("CHS predicted: %.1f mph  target: %.1f mph", chs_pred, chs_tgt)

    # 6. Save figure (if requested)
    if plot and out_dir is not None:
        _save_figure(
            target_norm,
            pred_norm,
            history,
            out_dir,
            chs_pred,
            chs_tgt,
            grip_rmse,
            club_rmse,
        )

    elapsed = time.perf_counter() - t0
    _LOG.info("Total elapsed: %.1f s", elapsed)

    return {
        "theta": theta_opt,
        "cost_final": float(opt_result.fun),
        "grip_rmse_norm": grip_rmse,
        "club_rmse_norm": club_rmse,
        "chs_mph_pred": chs_pred,
        "chs_mph_target": chs_tgt,
        "history": history,
        "elapsed_s": elapsed,
        "n_iter": opt_result.nit,
        "n_feval": opt_result.nfev,
        "success": opt_result.success,
        "message": opt_result.message,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def _save_figure(
    target_norm: dict,
    pred_norm: dict,
    history: list[float],
    out_dir: Path,
    chs_pred: float,
    chs_tgt: float,
    grip_rmse: float,
    club_rmse: float,
) -> None:
    """Save a 3-panel comparison figure."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        _LOG.warning("matplotlib not available — skipping figure")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    t = np.arange(T_SURROGATE)
    labels_xyz = ["X", "Y", "Z"]

    # Panel 1: grip trajectory
    ax = axes[0]
    for c, lbl in enumerate(labels_xyz):
        ax.plot(
            t, target_norm["grip"][:, c], "--", label=f"Target {lbl}", linewidth=1.5
        )
        ax.plot(t, pred_norm["grip"][:, c], "-", label=f"Pred {lbl}", linewidth=1.5)
    ax.axvline(
        target_norm["impact_idx"], color="k", linestyle=":", linewidth=1, label="Impact"
    )
    ax.set_title(f"Grip trajectory\nRMSE={grip_rmse:.4f} shaft-lengths")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Position (shaft-length units)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 2: clubhead trajectory
    ax = axes[1]
    for c, lbl in enumerate(labels_xyz):
        ax.plot(
            t, target_norm["clubhead"][:, c], "--", label=f"Target {lbl}", linewidth=1.5
        )
        ax.plot(t, pred_norm["clubhead"][:, c], "-", label=f"Pred {lbl}", linewidth=1.5)
    ax.axvline(
        target_norm["impact_idx"], color="k", linestyle=":", linewidth=1, label="Impact"
    )
    ax.set_title(f"Clubhead trajectory\nRMSE={club_rmse:.4f} shaft-lengths")
    ax.set_xlabel("Timestep")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 3: cost history
    ax = axes[2]
    ax.semilogy(history)
    ax.set_title(
        f"Optimisation cost history\n"
        f"CHS: pred={chs_pred:.1f} mph  target={chs_tgt:.1f} mph"
    )
    ax.set_xlabel("Function evaluation")
    ax.set_ylabel("Cost (log scale)")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Surrogate-based club motion match — Wiffle ProV1", fontsize=13)
    fig.tight_layout()

    fig_path = out_dir / "surrogate_match.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _LOG.info("Figure saved to %s", fig_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--target",
        type=Path,
        default=None,
        help="Path to Wiffle xlsx file (default: canonical ProV1)",
    )
    p.add_argument("--sheet", default="TW_ProV1", help="Excel sheet name")
    p.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Path to surrogate checkpoint (default: production checkpoint)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: output/surrogate_match/<timestamp>)",
    )
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-plot", action="store_true", help="Skip figure generation")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    repo_root = PROJECT_ROOT

    target_path = args.target or (repo_root / _DEFAULT_TARGET)
    ckpt_path = args.ckpt or (repo_root / _DEFAULT_CKPT)
    out_dir = args.out

    if out_dir is None:
        import datetime

        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = repo_root / "output" / "surrogate_match" / stamp

    if not target_path.exists():
        _LOG.error("Target file not found: %s", target_path)
        return 1
    if not ckpt_path.exists():
        _LOG.error("Checkpoint not found: %s", ckpt_path)
        return 1

    results = run_match(
        target_path,
        ckpt_path,
        sheet_name=args.sheet,
        max_iter=args.max_iter,
        seed=args.seed,
        out_dir=out_dir,
        plot=not args.no_plot,
    )

    print("\n" + "=" * 60)
    print("SURROGATE MOTION MATCH RESULTS")
    print("=" * 60)
    print(f"  Grip RMSE       : {results['grip_rmse_norm']:.5f}  shaft-lengths")
    print(f"  Clubhead RMSE   : {results['club_rmse_norm']:.5f}  shaft-lengths")
    print(f"  CHS predicted   : {results['chs_mph_pred']:.1f} mph")
    print(f"  CHS target      : {results['chs_mph_target']:.1f} mph")
    print(f"  Final cost      : {results['cost_final']:.6f}")
    print(f"  Iterations      : {results['n_iter']}")
    print(f"  Fevals          : {results['n_feval']}")
    print(f"  Elapsed         : {results['elapsed_s']:.1f} s")
    print(f"  Output dir      : {out_dir}")
    print("=" * 60)

    # Quality assessment
    grip_threshold = 0.05  # 5% of shaft length = ~5 cm
    chs_error_pct = (
        abs(results["chs_mph_pred"] - results["chs_mph_target"])
        / results["chs_mph_target"]
        * 100
    )
    print("\nQUALITY ASSESSMENT (normalised frame):")
    print(
        f"  Grip RMSE < 5% shaft-length : {'✅ PASS' if results['grip_rmse_norm'] < grip_threshold else '❌ FAIL'}"
    )
    print(
        f"  CHS error < 10%             : {'✅ PASS' if chs_error_pct < 10 else f'❌ FAIL ({chs_error_pct:.1f}%)'}"
    )

    note = (
        "\nNOTE: This demo uses the NN surrogate (no Simulink license required).\n"
        "The surrogate was trained on random-sweep Simscape data (10k trials).\n"
        "Coordinate frames differ — target is in GEARS/Vicon frame,\n"
        "surrogate output is in Simscape frame. Both are normalised to\n"
        "grip-centred shaft-length units for comparison.\n"
        "For production use, run fit_swing_full_pipeline.m with\n"
        "Simscape Multibody (Option 1 fmincon) for physically-grounded fits."
    )
    print(note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
