"""Tests for constrained sextic representation and SVD subspace decomposition."""

import numpy as np
import pytest

from src.shared.python.motion_matching.constrained_sextic import (
    build_prefix_svd_basis,
    cubic_to_normalized_sextic,
    evaluate_normalized_sextic,
    normalized_to_simscape_powers,
    simscape_powers_to_normalized,
)

pytestmark = pytest.mark.unit


def test_roundtrip_simscape_and_normalized() -> None:
    # Random Simscape coefficients [A, B, C, D, E, F, G] for 3 joints
    rng = np.random.default_rng(42)
    native = rng.standard_normal((3, 7))
    T_full = 1.814

    normalized = simscape_powers_to_normalized(native, T_full=T_full)
    recovered = normalized_to_simscape_powers(normalized, T_full=T_full)

    np.testing.assert_allclose(recovered, native, rtol=1e-12, atol=1e-12)

    # Evaluate at various times
    t_test = np.linspace(0.0, 1.814, 50)
    for j in range(3):
        # Native polyval evaluates A*t^6 + ... + G
        val_native = np.polyval(native[j], t_test)
        val_norm = evaluate_normalized_sextic(normalized[j], t_test, T_full=T_full)
        np.testing.assert_allclose(val_norm, val_native, rtol=1e-10, atol=1e-10)


def test_cubic_elevation_preserves_exact_values() -> None:
    # Cubic polynomial has A=B=C=0 in native Simscape [0, 0, 0, D, E, F, G]
    native_cubic = np.array([[0.0, 0.0, 0.0, 2.5, -4.0, 10.0, 15.0]])
    T_full = 1.814

    norm_sextic = cubic_to_normalized_sextic(native_cubic, T_full=T_full)
    # Higher order coefficients p_4, p_5, p_6 should be exactly 0
    assert norm_sextic[0, 4] == 0.0
    assert norm_sextic[0, 5] == 0.0
    assert norm_sextic[0, 6] == 0.0

    t_test = np.linspace(0.0, 0.60, 100)
    val_cubic = np.polyval(native_cubic[0], t_test)
    val_sextic = evaluate_normalized_sextic(norm_sextic[0], t_test, T_full=T_full)
    np.testing.assert_allclose(val_sextic, val_cubic, rtol=1e-12, atol=1e-12)


def test_prefix_svd_basis_subspace_property() -> None:
    W, Sigma, V = build_prefix_svd_basis(T_prefix=0.60, T_full=1.814, n_samples=217)

    # W is 7x7 orthogonal
    np.testing.assert_allclose(W.T @ W, np.eye(7), atol=1e-12)

    # Perturbation along w_7 has tiny norm on [0, 0.60]
    early_effect = V @ W[:, -1]
    assert np.max(np.abs(early_effect)) < 2e-5

    # At t=1.15s, effect of w_7 is orders of magnitude larger
    t_later = 1.15
    s_later = t_later / 1.814
    v_later = np.array([s_later**j for j in range(7)])
    later_effect = np.dot(v_later, W[:, -1])
    assert abs(later_effect) > 100 * np.max(np.abs(early_effect))
