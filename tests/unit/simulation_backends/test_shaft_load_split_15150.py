"""Internal shaft loads along the flexible-shaft chain, by part (#15150 Z9)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.simulation_backends.constrained_ztcf_zvcf import (
    constrained_acceleration_at,
)
from src.shared.python.simulation_backends.hand_force_split import PARTS
from src.shared.python.simulation_backends.shaft_load_split import (
    ShaftSectionLoads,
    load_settings,
    section_loads,
    shaft_load_split,
)

pytestmark = pytest.mark.unit

N_LINKS = 4
SPAN_M = 1.1
NV = 6
_GRAVITY = np.array([0.0, 0.0, -9.81])
_LINK_MASS = np.array([0.02, 0.02, 0.02, 0.22])
_K = np.random.default_rng(21).normal(size=(N_LINKS * 6, NV))


class _Provider:
    """Six coordinates, one 6-row weld; constant M, linear bias."""

    is_constrained = True
    nv = NV

    def __init__(self) -> None:
        rng = np.random.default_rng(3)
        a = rng.normal(size=(NV, NV))
        self._mass = a @ a.T + NV * np.eye(NV)
        self._jac = rng.normal(size=(6, NV)) + 3 * np.eye(6)

    def mass_matrix(self, q):
        return self._mass

    def bias_forces(self, q, v):
        return 2.0 * np.asarray(v) + np.array([0, 0, 9.8, 0, 0, 0])

    def closure_jacobian(self, q):
        return self._jac

    def closure_drift(self, q, v):
        return np.zeros(6)


def _stations(_q):
    """Joints along +x from the grip end (x = 0), shaft axis tip-ward."""
    points = np.zeros((N_LINKS, 3))
    points[:, 0] = np.arange(N_LINKS) * SPAN_M / N_LINKS
    return points, np.tile([1.0, 0.0, 0.0], (N_LINKS, 1))


def _link_load(q, v, a, gravity):
    """Fake per-link demand: affine in ``a``, quadratic in ``v``, gravity once."""
    v = np.asarray(v, dtype=float)
    load = (_K @ np.asarray(a, dtype=float)).reshape(N_LINKS, 6)
    load += 0.5 * float(v @ v)
    if gravity:
        load[:, :3] -= _LINK_MASS[:, None] * _GRAVITY
    return load[:, :3], load[:, 3:]


def _cantilever(tip_load: np.ndarray):
    """Massless static links; an external ``tip_load`` acts at the tip (x = L).

    A static link's demand is minus the external load on it.
    """
    points, axes = _stations(None)
    forces, moments = np.zeros((N_LINKS, 3)), np.zeros((N_LINKS, 3))
    forces[-1] = -tip_load
    moments[-1] = np.cross([SPAN_M, 0.0, 0.0], -tip_load)
    return section_loads(points, axes, forces, moments), points


def test_cantilever_tip_load_gives_root_moment_f_l_and_constant_shear() -> None:
    rtol = load_settings().static_check_rtol
    load = 7.5
    loads, points = _cantilever(np.array([0.0, 0.0, -load]))
    shear, bending = loads.shear_magnitude_n, loads.bending_magnitude_nm
    np.testing.assert_allclose(bending[0, 0], load * SPAN_M, rtol=rtol)
    np.testing.assert_allclose(bending[0], load * (SPAN_M - points[:, 0]), rtol=rtol)
    np.testing.assert_allclose(shear[0], load, rtol=rtol)
    np.testing.assert_allclose(loads.axial_n, 0.0, atol=rtol * load)
    np.testing.assert_allclose(loads.torsion_nm, 0.0, atol=rtol * load * SPAN_M)


def test_tip_pull_along_the_axis_is_tension_without_bending() -> None:
    rtol = load_settings().static_check_rtol
    loads, _ = _cantilever(np.array([3.0, 0.0, 0.0]))
    np.testing.assert_allclose(loads.axial_n[0], 3.0, rtol=rtol)
    np.testing.assert_allclose(loads.bending_magnitude_nm, 0.0, atol=rtol)
    np.testing.assert_allclose(loads.shear_magnitude_n, 0.0, atol=rtol)


def test_ztcf_plus_zvcf_is_the_total_per_link() -> None:
    q, v = np.zeros((2, NV)), np.full((2, NV), 0.3)
    tau = np.vstack([np.ones(NV), np.linspace(-2.0, 2.0, NV)])
    split = shaft_load_split(_Provider(), _stations, _link_load, q, v, tau)
    total = split.total()
    summed = split.part("ztcf") + split.part("zvcf")
    for name in ("axial_n", "shear_n", "torsion_nm", "bending_nm"):
        np.testing.assert_allclose(getattr(summed, name), getattr(total, name))
    for k in range(2):
        a, _ = constrained_acceleration_at(_Provider(), q[k], v[k], tau[k])
        ref = section_loads(*_stations(q[k]), *_link_load(q[k], v[k], a, True))
        np.testing.assert_allclose(total.axial_n[k], ref.axial_n[0], atol=1e-9)
        np.testing.assert_allclose(total.bending_nm[k], ref.bending_nm[0], atol=1e-9)
    assert total.axial_n.shape == (2, N_LINKS)
    assert total.shear_n.shape == (2, N_LINKS, 3)
    assert set(split.parts) == set(PARTS)


def test_gravity_sits_in_the_ztcf_only() -> None:
    q, v, tau = np.zeros((1, NV)), np.zeros((1, NV)), np.zeros((1, NV))
    split = shaft_load_split(_Provider(), _stations, _link_load, q, v, tau)
    zvcf = split.part("zvcf")
    np.testing.assert_allclose(zvcf.bending_nm, 0.0, atol=1e-12)
    np.testing.assert_allclose(zvcf.shear_n, 0.0, atol=1e-12)
    assert split.part("ztcf").bending_magnitude_nm[0, 0] > 0.0


def test_mismatched_shapes_are_refused() -> None:
    points, axes = _stations(None)
    three = np.zeros((N_LINKS, 3))
    with pytest.raises(ValueError, match="shape"):
        section_loads(points, axes[:-1], three, three)
    with pytest.raises(ValueError, match="shape"):
        section_loads(points, axes, np.zeros((N_LINKS, 2)), three)
    with pytest.raises(ValueError, match="axis"):
        section_loads(points, np.zeros_like(axes), three, three)
    with pytest.raises(ValueError, match="rows"):
        shaft_load_split(
            _Provider(),
            _stations,
            _link_load,
            np.zeros((2, NV)),
            np.zeros((3, NV)),
            np.zeros((2, NV)),
        )
    short = lambda q, v, a, g: tuple(x[:-1] for x in _link_load(q, v, a, g))  # noqa: E731
    with pytest.raises(ValueError, match="shape"):
        shaft_load_split(
            _Provider(), _stations, short, *(np.zeros((1, NV)) for _ in range(3))
        )
    one, _ = _cantilever(np.array([0.0, 0.0, -1.0]))
    two = ShaftSectionLoads(
        *(np.zeros((1, 2, 3)) if i % 2 else np.zeros((1, 2)) for i in range(4))
    )
    with pytest.raises(ValueError, match="shape"):
        _ = one + two


def test_unknown_part_is_refused() -> None:
    split = shaft_load_split(
        _Provider(), _stations, _link_load, *(np.zeros((1, NV)) for _ in range(3))
    )
    with pytest.raises(ValueError, match="part"):
        split.part("gravity")
