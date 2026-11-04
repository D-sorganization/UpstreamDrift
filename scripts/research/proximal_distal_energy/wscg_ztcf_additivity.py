"""ZTCF additivity residuals on the pinned WSCG best-match model (Issue #14438).

Runs the model pinned by ``wscg_model_selection.json`` (#14437) from its pinned
initial condition under a deterministic joint-torque program, and at each
sample evaluates the canonical pointwise counterfactuals at the same ``q``:
full ``(q, v, u)``, ZTCF ``(q, v, 0)`` (the drift), ZVCF (the control
component, ``M^-1 (B u + J^T (lambda - lambda_0))``, gravity counted once in
ZTCF) and the zero-velocity, zero-torque acceleration ``(q, 0, 0)``.
Definitions, frames and signs are documented in ``tests/unit/research/
test_wscg_ztcf_additivity.py``; the canonical source is
``src/shared/python/simulation_backends/ztcf_zvcf.py`` (#14494).

Residuals (infinity norms of generalized accelerations):

- ``additivity_residual``: ``a_full - a_ZTCF - a_ZVCF``
- ``velocity_residual``: ``a_ZTCF - a_00 - M^-1 (d(passive - bias) + d(J^T lambda))``

The identity is pointwise only; integrated counterfactual trajectories do not
generally sum. A model-scenario result, not empirical validation.

Usage::

    python3 -m scripts.research.proximal_distal_energy.wscg_ztcf_additivity
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from scripts.research.proximal_distal_energy.wscg_model_selection import (
    CANDIDATES,
    RECORD_PATH,
    REPO_ROOT,
)

_DATA = Path("docs/research/proximal_distal_energy_transfer/data")
SUMMARY_PATH = _DATA / "wscg_ztcf_additivity.json"
SERIES_PATH = _DATA / "wscg_ztcf_additivity_residuals.npz"

#: Declared bound: residual <= ABS_TOLERANCE + RELATIVE_TOLERANCE * |a_full|.
ABS_TOLERANCE = 1e-9
RELATIVE_TOLERANCE = 1e-10

#: Deterministic torque program: ctrl_i(t) = AMPLITUDE sin(2 pi FREQ t + i).
AMPLITUDE = 0.3
FREQUENCY_HZ = 2.0
HORIZON_S = 0.3
SAMPLE_EVERY = 10


class PinnedModelDriftError(ValueError):
    """The model source no longer matches the revision pinned by #14437."""


@dataclass(frozen=True)
class _Counterfactual:
    qacc: np.ndarray
    constraint: np.ndarray
    actuator: np.ndarray
    smooth_passive: np.ndarray


def load_pinned_model(expected_revision: str | None = None) -> mujoco.MjModel:
    """Compile the #14437-selected model, refusing a revision mismatch.

    Raises:
        PinnedModelDriftError: if the model XML hash differs from the pin.
    """
    record = json.loads((REPO_ROOT / RECORD_PATH).read_text(encoding="utf-8"))
    pinned = expected_revision or record["manifest"]["model"]["revision"]
    candidate = next(c for c in CANDIDATES if c.model_id == record["selected"])
    xml = candidate.xml()
    actual = "sha256:" + hashlib.sha256(xml.encode("utf-8")).hexdigest()
    if actual != pinned:
        raise PinnedModelDriftError(
            f"{candidate.model_id} revision {actual} differs from pinned {pinned}"
        )
    return mujoco.MjModel.from_xml_string(xml)


def _evaluate(
    model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray, v: np.ndarray, u: Any
) -> _Counterfactual:
    data.qpos[:], data.qvel[:], data.ctrl[:] = q, v, u
    mujoco.mj_forward(model, data)
    return _Counterfactual(
        data.qacc.copy(),
        data.qfrc_constraint.copy(),
        data.qfrc_actuator.copy(),
        (data.qfrc_passive - data.qfrc_bias).copy(),
    )


def _solve_mass(model: mujoco.MjModel, data: mujoco.MjData, force: Any) -> Any:
    out = np.zeros((1, model.nv))
    mujoco.mj_solveM(model, data, out, np.atleast_2d(force))
    return out[0]


def _sample(model: mujoco.MjModel, data: mujoco.MjData, q: Any, v: Any, u: Any):
    full = _evaluate(model, data, q, v, u)
    ztcf = _evaluate(model, data, q, v, np.zeros_like(u))
    rest = _evaluate(model, data, q, np.zeros_like(v), np.zeros_like(u))
    zvcf = _solve_mass(model, data, full.actuator + full.constraint - ztcf.constraint)
    velocity = _solve_mass(
        model,
        data,
        ztcf.smooth_passive - rest.smooth_passive + ztcf.constraint - rest.constraint,
    )
    return {
        "speed": np.abs(v).max(),
        "full_norm": np.abs(full.qacc).max(),
        "zvcf_norm": np.abs(zvcf).max(),
        "additivity_residual": np.abs(full.qacc - ztcf.qacc - zvcf).max(),
        "velocity_residual": np.abs(ztcf.qacc - rest.qacc - velocity).max(),
    }


def run_additivity() -> dict[str, np.ndarray]:
    """Roll out the pinned run and return per-sample residual series."""
    model = load_pinned_model()
    sim, probe = mujoco.MjData(model), mujoco.MjData(model)
    phases = np.arange(model.nu, dtype=float)
    low, high = model.actuator_ctrlrange.T
    rows: list[dict[str, float]] = []
    times: list[float] = []
    for step in range(round(HORIZON_S / model.opt.timestep)):
        wave = AMPLITUDE * np.sin(2 * np.pi * FREQUENCY_HZ * sim.time + phases)
        sim.ctrl[:] = np.clip(wave, low, high)
        if step % SAMPLE_EVERY == SAMPLE_EVERY - 1:
            times.append(sim.time)
            rows.append(_sample(model, probe, sim.qpos, sim.qvel, sim.ctrl.copy()))
        mujoco.mj_step(model, sim)
    series = {key: np.array([r[key] for r in rows]) for key in rows[0]}
    series["time_s"] = np.array(times)
    return series


def summarize(series: dict[str, np.ndarray]) -> dict[str, Any]:
    """Summary written beside the residual series."""
    share = series["zvcf_norm"] / series["full_norm"]
    return {
        "issue": 14438,
        "pinned_record": RECORD_PATH.as_posix(),
        "identities": {
            "additivity": "a_full = a_ZTCF + a_ZVCF (pointwise only)",
            "zvcf": "a_ZVCF = M^-1 (B u + J^T (lambda - lambda_0))",
            "velocity": "a_ZTCF - a_00 = M^-1 (d(passive - bias) + d(J^T lambda))",
        },
        "torque_program": {
            "amplitude_ctrl": AMPLITUDE,
            "frequency_hz": FREQUENCY_HZ,
            "horizon_s": HORIZON_S,
            "phase_rad": "actuator index",
        },
        "declared_tolerance": {"abs": ABS_TOLERANCE, "rel": RELATIVE_TOLERANCE},
        "samples": int(len(series["time_s"])),
        "max_additivity_residual": float(series["additivity_residual"].max()),
        "max_velocity_residual": float(series["velocity_residual"].max()),
        "median_zvcf_share": float(np.median(share)),
        "scope": "Model-scenario result; not empirical validation.",
        "series_file": SERIES_PATH.name,
    }


def main() -> None:
    series = run_additivity()
    np.savez_compressed(REPO_ROOT / SERIES_PATH, **series)
    summary = json.dumps(summarize(series), indent=2) + "\n"
    (REPO_ROOT / SUMMARY_PATH).write_text(summary, encoding="utf-8")
    print(f"wrote {SUMMARY_PATH} and {SERIES_PATH}")


if __name__ == "__main__":
    main()
