"""Seeded independent contact invariants; JSON receipt on stdout."""

import hashlib
from pathlib import Path
import sys
import json
import numpy as np
from dataclasses import replace
from scipy.spatial.transform import Rotation
from src.shared.python.physics.rigid_contact import RigidContactInput
from src.shared.python.physics.incremental_contact import (
    solve_incremental_contact,
    contact_backend_info,
)
from src.shared.python.core.physics_constants import (
    GOLF_BALL_MASS_KG as m,
    GOLF_BALL_MOMENT_OF_INERTIA_KG_M2 as ib,
    GOLF_BALL_RADIUS_M as R,
)

from src.tools.shot_pattern_analysis.matched_export import matched_source_snapshot

execution = matched_source_snapshot()
rng = np.random.default_rng(220926)
mx = dict.fromkeys(
    (
        "linear",
        "angular",
        "work",
        "parity",
        "rotation",
        "energy_excess",
        "dissipation_deficit",
    ),
    0.0,
)
failures = []
refs = []
for i in range(100):
    Q = Rotation.random(random_state=rng).as_matrix()
    n = Q[:, 0]
    rc = Q @ np.array([0.035, *rng.uniform(-0.025, 0.025, 2)])
    inertia = Q @ np.diag([0.00025, 0.0004, 0.00045]) @ Q.T
    vc = Q @ np.array([rng.uniform(20, 50), *rng.uniform(-20, 20, 2)])
    wc = rng.normal(0, 5, 3)
    vb = rng.normal(0, 1, 3)
    wb = rng.normal(0, 150, 3)
    e = [0, 0.4, 0.83, 1][i % 4]
    mu = [0.02, 0.2, 0.4, 1, 5][i % 5]
    s = RigidContactInput(vc, wc, vb, wb, n, rc, inertia, 0.2, e, mu)
    try:
        o = solve_incremental_contact(
            s,
            energetic_restitution=e,
            steps_per_compression=128,
            use_analytic_fastpath=False,
            backend="numba",
        )
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        failures.append((i, str(exc)))
        continue
    p = np.array([0.12, 0.32, -0.17])
    b = p + rc + R * n

    def angular_momentum(v, w, u, z, inertia=inertia, p=p, b=b):
        return inertia @ w + ib * z + np.cross(p, 0.2 * v) + np.cross(b, m * u)

    mx["linear"] = max(
        mx["linear"],
        float(
            np.linalg.norm(
                0.2 * o.club_velocity_mps + m * o.ball_velocity_mps - 0.2 * vc - m * vb
            )
        ),
    )
    mx["angular"] = max(
        mx["angular"],
        float(
            np.linalg.norm(
                angular_momentum(
                    o.club_velocity_mps,
                    o.club_angular_velocity_rad_s,
                    o.ball_velocity_mps,
                    o.ball_angular_velocity_rad_s,
                )
                - angular_momentum(vc, wc, vb, wb)
            )
        ),
    )
    mx["work"] = max(mx["work"], abs(o.rebound_work_j - e * e * o.compression_work_j))
    de = o.post_kinetic_energy_j - o.pre_kinetic_energy_j
    mx["energy_excess"] = max(mx["energy_excess"], de)
    mx["dissipation_deficit"] = max(
        mx["dissipation_deficit"], de + (1 - e * e) * o.compression_work_j
    )
    if i < 12:
        ref = solve_incremental_contact(
            s,
            energetic_restitution=e,
            steps_per_compression=128,
            use_analytic_fastpath=False,
            backend="python_reference",
        )
        mx["parity"] = max(
            mx["parity"],
            float(np.max(np.abs(ref.ball_impulse_n_s - o.ball_impulse_n_s))),
        )
        A = Rotation.random(random_state=rng).as_matrix()
        t = replace(
            s,
            club_velocity_mps=A @ vc,
            club_angular_velocity_rad_s=A @ wc,
            ball_velocity_mps=A @ vb,
            ball_angular_velocity_rad_s=A @ wb,
            normal=A @ n,
            club_lever_m=A @ rc,
            club_inertia_world_kg_m2=A @ inertia @ A.T,
        )
        z = solve_incremental_contact(
            t,
            energetic_restitution=e,
            steps_per_compression=128,
            use_analytic_fastpath=False,
            backend="numba",
        )
        mx["rotation"] = max(
            mx["rotation"],
            float(np.max(np.abs(z.ball_impulse_n_s - A @ o.ball_impulse_n_s))),
        )
        fine = solve_incremental_contact(
            s,
            energetic_restitution=e,
            steps_per_compression=256,
            use_analytic_fastpath=False,
            backend="numba",
        )
        finer = solve_incremental_contact(
            s,
            energetic_restitution=e,
            steps_per_compression=512,
            use_analytic_fastpath=False,
            backend="numba",
        )
        refs.append(
            [
                float(np.linalg.norm(fine.ball_velocity_mps - o.ball_velocity_mps)),
                float(np.linalg.norm(finer.ball_velocity_mps - fine.ball_velocity_mps)),
            ]
        )
sys.stdout.write(
    json.dumps(
        {
            "seed": 220926,
            "cases": 100,
            "execution": execution,
            "source_unchanged": execution == matched_source_snapshot(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "backend": contact_backend_info(),
            "maxima": mx,
            "units": {
                "linear": "kg m/s",
                "angular": "kg m2/s",
                "work": "J",
                "parity": "N s",
                "rotation": "N s",
                "energy_excess": "J",
                "dissipation_deficit": "J",
            },
            "invariants_pass": not failures
            and all(
                mx[key] <= tolerance
                for key, tolerance in {
                    "linear": 1e-10,
                    "angular": 1e-10,
                    "work": 1e-10,
                    "parity": 1e-12,
                    "rotation": 1e-10,
                    "energy_excess": 1e-8,
                    "dissipation_deficit": 1e-8,
                }.items()
            ),
            "failures": failures,
            "refinement_ball_mps_128_256_512": refs,
        },
        indent=2,
    )
)
