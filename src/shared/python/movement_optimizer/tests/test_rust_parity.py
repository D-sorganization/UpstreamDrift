# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Rust <-> NumPy parity for the batch inverse-dynamics hot path (issue #494).

When the optional ``movement_optimizer_core`` maturin extension is installed,
``LagrangianDynamics.inverse_dynamics_batch`` runs in Rust; otherwise a NumPy
fallback runs. Users must get identical optimization results regardless of
whether the extension is present, so this module asserts the two paths agree to
tight tolerance on randomized inputs. If a Rust change ever diverges from the
reference NumPy implementation, this test fails instead of silently shipping
different torques.

The whole module is skipped when the extension is unavailable (e.g. local dev
machines that never ran ``maturin develop``). CI builds the extension and runs
this as a required gate, so the parity contract is enforced where it matters.
"""

from __future__ import annotations

import numpy as np
import pytest

from movement_optimizer.models import BodyModel, make_squat_config

# Skip the entire module unless the compiled Rust accelerator is importable.
# Importing the symbol here (rather than inside each test) lets a single
# collection-time skip stand in for every case.
rust = pytest.importorskip(
    "movement_optimizer_core",
    reason="Rust accelerator (movement_optimizer_core) not built; run `maturin develop --release`.",
)


def _make_dynamics():
    """Build a reference squat ``LagrangianDynamics`` with cached M/a/g terms."""
    body = BodyModel(75.0, 1.75)
    config = make_squat_config(body, 60.0)
    return config[0]


def _random_trajectory(
    rng: np.random.Generator, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return finite, physically-plausible (q, qd, qdd) batches of shape (n, 3).

    Joint speeds are kept below ``CORIOLIS_SLOW_LIMIT_RAD_S`` so the slow-movement
    assumption holds and neither path emits the Coriolis warning, isolating the
    comparison to the shared analytic model rather than divergent guard paths.
    """
    q = rng.uniform(-1.5, 1.5, size=(n, 3))
    qd = rng.uniform(-1.5, 1.5, size=(n, 3))
    qdd = rng.uniform(-3.0, 3.0, size=(n, 3))
    return q, qd, qdd


@pytest.mark.parametrize("n", [1, 8, 256])
def test_rust_matches_numpy_inverse_dynamics_batch(n: int) -> None:
    """Rust ``inverse_dynamics_batch_rs`` matches the NumPy reference path.

    Compares the Rust kernel directly against ``_numpy_inverse_dynamics_batch``
    (the same routine the public method falls back to) so the assertion is
    independent of which path ``inverse_dynamics_batch`` happens to select.
    """
    dyn = _make_dynamics()
    rng = np.random.default_rng(20260612 + n)
    q, qd, qdd = _random_trajectory(rng, n)

    numpy_torques = dyn._numpy_inverse_dynamics_batch(q, qd, qdd)
    rust_torques = rust.inverse_dynamics_batch_rs(
        q,
        qd,
        qdd,
        dyn._M00,
        dyn._M11,
        dyn._M22,
        dyn._a01,
        dyn._a02,
        dyn._a12,
        dyn._g0,
        dyn._g1,
        dyn._g2,
    )

    rust_arr = np.asarray(rust_torques)
    assert rust_arr.shape == numpy_torques.shape == (n, 3)
    assert np.all(np.isfinite(rust_arr))
    np.testing.assert_allclose(rust_arr, numpy_torques, rtol=1e-9, atol=1e-9)


def test_public_method_uses_rust_when_available() -> None:
    """With the extension installed, the public path equals the Rust kernel.

    Guards against a regression where ``inverse_dynamics_batch`` stops dispatching
    to Rust (e.g. an import moved inside a broken try) and silently degrades to
    NumPy on a host that paid to build the accelerator.
    """
    dyn = _make_dynamics()
    rng = np.random.default_rng(42)
    q, qd, qdd = _random_trajectory(rng, 16)

    public = np.asarray(dyn.inverse_dynamics_batch(q, qd, qdd))
    rust_direct = np.asarray(
        rust.inverse_dynamics_batch_rs(
            q,
            qd,
            qdd,
            dyn._M00,
            dyn._M11,
            dyn._M22,
            dyn._a01,
            dyn._a02,
            dyn._a12,
            dyn._g0,
            dyn._g1,
            dyn._g2,
        )
    )
    np.testing.assert_allclose(public, rust_direct, rtol=1e-12, atol=0.0)
