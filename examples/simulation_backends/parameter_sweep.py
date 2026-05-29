"""Monte-Carlo parameter sweep over the golf double-pendulum (CPU-only).

This example sweeps the **clubhead mass** across ~64 randomly sampled values and,
for each, runs a passive "downswing" rollout from a fixed raised (backswing)
pose. It reports a *clubhead-speed proxy* — the magnitude of the joint-velocity
vector ``|omega|`` at the final step — and writes the single fastest rollout to
an HDF5 trace.

Everything here runs on the CPU through the ``ode`` reference backend, so it
needs **no GPU, no CUDA, and no optional ``[warp]`` extra**. It is intentionally
the workload where the GPU would *not* help (see ADR-0023): a handful of short
2-DoF rollouts. For hundreds-to-thousands of rollouts you would instead build
the same per-env factory against the ``mjwarp`` backend and call
``rollout_batch``; the batched CPU reference path used below
(``cpu_batch_rollout``) has the identical contract.

Run it::

    python3 examples/simulation_backends/parameter_sweep.py

Keeps runtime to a few seconds. See ``docs/simulation_backends/README.md``.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

# Allow running this script directly (``python3 examples/.../parameter_sweep.py``):
# put the repository root on sys.path before importing the ``src`` package.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.shared.python.simulation_backends import (  # noqa: E402
    GolfModelParams,
    SimState,
    has_mujoco,
    make_backend,
)
from src.shared.python.simulation_backends.batched import (  # noqa: E402
    cpu_batch_rollout,
)
from src.shared.python.simulation_backends.trace_io import (  # noqa: E402
    write_trace,
)

# --- Sweep configuration ----------------------------------------------------
#: Backend to drive the sweep with. ``ode`` is the always-available CPU
#: reference; ``mujoco`` (guarded by ``has_mujoco``) is an independent CPU
#: engine. The GPU ``mjwarp`` backend is deliberately *not* used: a sweep of a
#: few dozen tiny rollouts is exactly the case where launch/transfer overhead
#: makes the GPU slower (ADR-0023).
BACKEND_NAME = "mujoco" if has_mujoco() else "ode"

NUM_SAMPLES = 64  #: number of clubhead-mass samples to evaluate
HORIZON = 350  #: integration steps per rollout
DT_S = 0.004  #: integration step size [s]
RNG_SEED = 0  #: seed every RNG draw for reproducibility

#: Raised "top of backswing" pose; the passive swing falls from here so the
#: clubhead-speed proxy is non-trivial (from q = 0 the pendulum is at rest).
BACKSWING = SimState(q=np.array([2.2, 0.4]), v=np.zeros(2))

#: Inclusive clubhead-mass sweep bounds [kg] around the ~0.2 kg default.
CLUBHEAD_MASS_MIN_KG = 0.12
CLUBHEAD_MASS_MAX_KG = 0.32


def sample_clubhead_masses(rng: np.random.Generator) -> np.ndarray:
    """Draw ``NUM_SAMPLES`` clubhead masses uniformly within the sweep bounds.

    Args:
        rng: Seeded NumPy generator (the single source of randomness).

    Returns:
        A sorted ``(NUM_SAMPLES,)`` array of masses [kg], ascending so the
        printed table reads monotonically.

    Postconditions:
        Every value lies in ``[CLUBHEAD_MASS_MIN_KG, CLUBHEAD_MASS_MAX_KG]``.
    """
    masses = rng.uniform(CLUBHEAD_MASS_MIN_KG, CLUBHEAD_MASS_MAX_KG, size=NUM_SAMPLES)
    return np.sort(masses)


def build_env_factory(base: GolfModelParams, masses: np.ndarray):
    """Return a per-env backend factory for :func:`cpu_batch_rollout`.

    Each environment gets a backend whose ``clubhead_mass_kg`` is overridden to
    ``masses[i]`` and which is pre-reset to the shared backswing pose, so the
    passive rollout falls from the same start for every sample.

    Args:
        base: The single-source-of-truth model parameters to perturb.
        masses: Per-env clubhead masses [kg]; ``len(masses)`` envs are produced.

    Returns:
        A callable mapping an env index to a fresh, reset
        :class:`~simulation_backends.protocol.SimulationBackend`.
    """

    def factory(index: int):
        lower = base.lower.model_copy(update={"clubhead_mass_kg": float(masses[index])})
        backend = make_backend(BACKEND_NAME, base.model_copy(update={"lower": lower}))
        backend.reset(BACKSWING.copy())
        return backend

    return factory


def clubhead_speed_proxy(final_velocities: np.ndarray) -> np.ndarray:
    """Return the per-env clubhead-speed proxy ``|omega|`` at the final step.

    Args:
        final_velocities: ``(N, nv)`` joint velocities at the last sample.

    Returns:
        A ``(N,)`` array of non-negative proxy speeds [rad/s].
    """
    return np.linalg.norm(final_velocities, axis=1)


def print_summary(masses: np.ndarray, proxy: np.ndarray, *, every: int = 8) -> int:
    """Print a sweep summary table and return the fastest env index.

    Args:
        masses: Swept clubhead masses [kg], shape ``(N,)``.
        proxy: Clubhead-speed proxy per env [rad/s], shape ``(N,)``.
        every: Row stride so the table stays compact for large sweeps.

    Returns:
        Index of the environment with the maximum proxy speed.
    """
    print(f"Parameter sweep: {len(masses)} samples on '{BACKEND_NAME}' backend")
    print(f"  horizon={HORIZON} steps, dt={DT_S} s, seed={RNG_SEED}")
    print(f"{'idx':>4}  {'clubhead_mass [kg]':>18}  {'|omega|_final [rad/s]':>22}")
    for i in range(0, len(masses), every):
        print(f"{i:>4}  {masses[i]:>18.4f}  {proxy[i]:>22.4f}")

    best = int(np.argmax(proxy))
    print(
        f"\nFastest: env {best} "
        f"(clubhead_mass={masses[best]:.4f} kg, |omega|={proxy[best]:.4f} rad/s)"
    )
    print(
        f"Proxy range: [{proxy.min():.4f}, {proxy.max():.4f}] rad/s "
        f"(mean {proxy.mean():.4f})"
    )
    return best


def main() -> None:
    """Run the CPU parameter sweep and persist the fastest trace to HDF5."""
    rng = np.random.default_rng(RNG_SEED)
    base = GolfModelParams.default()
    masses = sample_clubhead_masses(rng)

    factory = build_env_factory(base, masses)
    batch = cpu_batch_rollout(
        factory,
        controls_batch=None,  # passive swing — gravity-driven, zero torque
        horizon=HORIZON,
        dt=DT_S,
        num_envs=NUM_SAMPLES,
    )

    proxy = clubhead_speed_proxy(batch.v[:, -1, :])
    best = print_summary(masses, proxy)

    out_path = Path(tempfile.gettempdir()) / "golf_parameter_sweep_fastest.h5"
    write_trace(batch.env(best), out_path)
    print(f"\nWrote fastest rollout trace to: {out_path}")


if __name__ == "__main__":
    main()
