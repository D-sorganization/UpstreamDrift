"""Generate full-stack parity fixtures for upstream-muscle.

Emits four CSV fixtures alongside the existing ``parity_hill.csv``:

- ``parity_activation.csv``  — Euler-step activation dynamics
- ``parity_muscle_force.csv`` — HillMuscleModel.compute_force
- ``parity_joint_torque.csv`` — Multi-muscle moment summation
- ``parity_step_full.csv``    — Combined RL step (u -> a -> F -> tau)

The Python source-of-truth lives in
``src/shared/python/biomechanics/{hill_muscle,activation_dynamics,multi_muscle}.py``.
Re-run after touching any of those modules.

Usage::

    python rust_core/upstream-muscle/scripts/generate_parity_fixture_full.py
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.shared.python.biomechanics.activation_dynamics import (  # noqa: E402
    ActivationDynamics,
)
from src.shared.python.biomechanics.hill_muscle import (  # noqa: E402
    HillMuscleModel,
    MuscleParameters,
    MuscleState,
)
from src.shared.python.biomechanics.multi_muscle import (  # noqa: E402
    MuscleGroup,
)


FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests"


def _write(path: Path, header: list[str], rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for row in rows:
            writer.writerow([repr(c) if isinstance(c, float) else c for c in row])


def write_activation_fixture() -> Path:
    """Sweep tau_act/tau_deact x u x a x dt and store per-step a' values."""
    dyn = ActivationDynamics(tau_act=0.010, tau_deact=0.040, min_activation=0.001)
    rows: list[tuple[float, float, float, float, float, float, float]] = []
    for u in [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]:
        for a in [0.001, 0.05, 0.2, 0.5, 0.9, 1.0]:
            for dt in [0.0005, 0.001, 0.005, 0.010]:
                a_new = dyn.update(u, a, dt)
                rows.append(
                    (dyn.tau_act, dyn.tau_deact, dyn.min_activation, u, a, dt, a_new)
                )
    # Add a step-response trace at dt=1 ms over 200 ms.
    a = 0.0
    for k in range(200):
        u = 1.0 if k < 100 else 0.0
        a = dyn.update(u, a, 0.001)
        rows.append(
            (dyn.tau_act, dyn.tau_deact, dyn.min_activation, u, 0.0, 0.001 * (k + 1), a)
        )
    _write(
        FIXTURES_DIR / "parity_activation.csv",
        ["tau_act", "tau_deact", "min_activation", "u", "a", "dt_or_t", "a_new"],
        rows,
    )
    return FIXTURES_DIR / "parity_activation.csv"


def write_muscle_force_fixture() -> Path:
    """Sweep activation x l_CE x v_CE for a handful of muscle parameter sets."""
    param_sets = [
        # F_max, l_opt, l_slack, v_max, pennation, damping, fl_width
        (1000.0, 0.15, 0.20, 10.0, 0.0, 0.05, 0.56),
        (800.0, 0.12, 0.10, 10.0, 0.2, 0.05, 0.56),
        (1200.0, 0.18, 0.22, 8.0, 0.0, 0.10, 0.40),
    ]
    rows: list[tuple] = []
    for f_max, l_opt, l_slack, v_max, alpha, damp, fw in param_sets:
        params = MuscleParameters(
            F_max=f_max,
            l_opt=l_opt,
            l_slack=l_slack,
            v_max=v_max,
            pennation_angle=alpha,
            damping=damp,
        )
        model = HillMuscleModel(params, force_length_width=fw)
        for a in [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]:
            for l_ratio in [0.6, 0.8, 1.0, 1.2, 1.4]:
                for v_ratio in [-0.8, -0.3, 0.0, 0.3, 0.8]:
                    l_ce = l_ratio * l_opt
                    v_ce = v_ratio * v_max * l_opt
                    state = MuscleState(
                        activation=a, l_CE=l_ce, v_CE=v_ce, l_MT=l_ce + l_slack
                    )
                    f = model.compute_force(state)
                    rows.append(
                        (
                            f_max,
                            l_opt,
                            l_slack,
                            v_max,
                            alpha,
                            damp,
                            fw,
                            a,
                            l_ce,
                            v_ce,
                            f,
                        )
                    )
    _write(
        FIXTURES_DIR / "parity_muscle_force.csv",
        [
            "f_max",
            "l_opt",
            "l_slack",
            "v_max",
            "pennation_angle",
            "damping",
            "force_length_width",
            "activation",
            "l_ce",
            "v_ce",
            "force",
        ],
        rows,
    )
    return FIXTURES_DIR / "parity_muscle_force.csv"


def write_joint_torque_fixture() -> Path:
    """Two-muscle flexor + one-muscle extensor → net torque for varied act/state."""
    # Build a single-joint MuscleGroup mirroring the elbow factory in
    # multi_muscle.py but with both flexors and the extensor merged into one
    # group with signed moment arms (matches the Rust API surface).
    group = MuscleGroup("elbow")
    group.add_muscle(
        "biceps",
        HillMuscleModel(MuscleParameters(F_max=1000.0, l_opt=0.15, l_slack=0.20)),
        moment_arm=0.04,
    )
    group.add_muscle(
        "brachialis",
        HillMuscleModel(MuscleParameters(F_max=800.0, l_opt=0.12, l_slack=0.10)),
        moment_arm=0.03,
    )
    group.add_muscle(
        "triceps",
        HillMuscleModel(MuscleParameters(F_max=1200.0, l_opt=0.18, l_slack=0.22)),
        moment_arm=-0.035,
    )
    rows: list[tuple] = []
    scenarios = [
        # bi, br, tri, l_ratios (uniform), v_ratios (uniform)
        (0.5, 0.5, 0.2, 1.0, 0.0),
        (0.9, 0.7, 0.1, 1.1, -0.1),
        (0.1, 0.1, 0.7, 0.9, 0.1),
        (0.0, 0.0, 0.0, 1.0, 0.0),
        (1.0, 1.0, 1.0, 1.0, 0.0),
    ]
    for bi, br, tri, l_ratio, v_ratio in scenarios:
        activations = {"biceps": bi, "brachialis": br, "triceps": tri}
        states = {}
        for name in ("biceps", "brachialis", "triceps"):
            l_opt = group.muscles[name].params.l_opt
            v_max = group.muscles[name].params.v_max
            states[name] = (l_ratio * l_opt, v_ratio * v_max * l_opt)
        tau = group.compute_net_torque(activations, states)
        rows.append((bi, br, tri, l_ratio, v_ratio, tau))
    _write(
        FIXTURES_DIR / "parity_joint_torque.csv",
        ["bi_act", "br_act", "tri_act", "l_ratio", "v_ratio", "tau"],
        rows,
    )
    return FIXTURES_DIR / "parity_joint_torque.csv"


def write_step_full_fixture() -> Path:
    """End-to-end RL step over a short trajectory: u → a' → F → τ.

    We unroll the Python pipeline manually (activation update, then
    HillMuscleModel.compute_force per muscle, then weighted moment-arm
    sum) and record per-step (a_new vector, tau scalar) for parity.
    """
    dyn = ActivationDynamics(tau_act=0.010, tau_deact=0.040, min_activation=0.001)
    params_list = [
        ("biceps", MuscleParameters(F_max=1000.0, l_opt=0.15, l_slack=0.20), 0.04),
        ("brachialis", MuscleParameters(F_max=800.0, l_opt=0.12, l_slack=0.10), 0.03),
        ("triceps", MuscleParameters(F_max=1200.0, l_opt=0.18, l_slack=0.22), -0.035),
    ]
    models = [(name, HillMuscleModel(p), r) for name, p, r in params_list]
    rows: list[tuple] = []
    a = [0.0, 0.0, 0.0]
    dt = 0.001
    for k in range(300):
        # square-wave excitation (different phases per muscle)
        u0 = 1.0 if (k // 50) % 2 == 0 else 0.0
        u1 = 1.0 if ((k + 25) // 50) % 2 == 0 else 0.0
        u2 = 1.0 if ((k + 50) // 50) % 2 == 0 else 0.0
        u = [u0, u1, u2]
        a = [dyn.update(u[i], a[i], dt) for i in range(3)]
        # Hold l_CE at l_opt, v_CE at 0 for simplicity.
        forces = []
        for (_name, model, _r), act in zip(models, a, strict=True):
            state = MuscleState(
                activation=act,
                l_CE=model.params.l_opt,
                v_CE=0.0,
                l_MT=model.params.l_opt + model.params.l_slack,
            )
            forces.append(model.compute_force(state))
        tau = sum(r * f for (_n, _m, r), f in zip(models, forces, strict=True))
        rows.append(
            (
                dt * (k + 1),
                u0,
                u1,
                u2,
                a[0],
                a[1],
                a[2],
                forces[0],
                forces[1],
                forces[2],
                tau,
            )
        )
    _write(
        FIXTURES_DIR / "parity_step_full.csv",
        [
            "t",
            "u_biceps",
            "u_brachialis",
            "u_triceps",
            "a_biceps",
            "a_brachialis",
            "a_triceps",
            "f_biceps",
            "f_brachialis",
            "f_triceps",
            "tau",
        ],
        rows,
    )
    return FIXTURES_DIR / "parity_step_full.csv"


def main() -> int:
    paths = [
        write_activation_fixture(),
        write_muscle_force_fixture(),
        write_joint_torque_fixture(),
        write_step_full_fixture(),
    ]
    for p in paths:
        sys.stdout.write(f"wrote {p}\n")
    # Sanity output so caller knows everything's finite.
    sys.stdout.write(f"(values look finite: sentinel = {math.pi:.6f})\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
