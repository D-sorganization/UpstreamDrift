"""C6 (#15150): the chain shaft against beam theory, flex targets and RK4.

Independent references on the SAME station profile (``station_profile``):
Euler-Bernoulli / Saint-Venant tip statics from the vendored Tools
``solve_cantilever_tip_response``, a trapezoid Euler-Bernoulli integral for
gravity, a Rayleigh quotient, and the ``SHAFT_FLEX_DATA`` clamp frequencies.
Every band and its justification live in
``tests/fixtures/shaft_chain_validation/settings_15150.json``; rows that miss
their band are ``xfail(strict=True)`` with the measured value.
"""

from __future__ import annotations

import json
from functools import lru_cache
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.motion_matching.club_models import CLUBS, ClubSpec
from src.shared.python.motion_matching.club_shaft_chain import (
    CPM_CLAMP_M,
    DEFAULT_LINKS,
    ShaftChain,
    build_shaft_chain,
    clamped_head_frequencies,
    clamped_shaft_document,
    flex_frequency_hz,
    span_properties,
    station_profile,
)

mujoco = pytest.importorskip("mujoco")

pytestmark = pytest.mark.unit
SETTINGS_PATH = (
    Path(__file__).resolve().parents[2]
    / "fixtures/shaft_chain_validation/settings_15150.json"
)
SETTINGS: dict[str, Any] = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
BANDS: dict[str, Any] = SETTINGS["bands"]
CLUB: ClubSpec = CLUBS[SETTINGS["club"]]
SUBSTEP_S = SETTINGS["same_input_step_s"] / SETTINGS["same_input_rk4_substeps"]


# --- models -------------------------------------------------------------------


def cpm_clamp_club(club: ClubSpec) -> ClubSpec:
    """``club`` whose chain starts at the 5 in frequency-analyser clamp.

    The shaft mass grows with the span so the linear density, and therefore
    the calibrated modulus and the whole EI(x) profile, are unchanged.
    """
    span, clamp_span = club.length_m - club.grip_length_m, club.length_m - CPM_CLAMP_M
    return replace(
        club,
        grip_length_m=CPM_CLAMP_M,
        shaft_mass_kg=club.shaft_mass_kg * clamp_span / span,
    )


def compile_chain(chain: ShaftChain, gravity_m_s2: float = 0.0) -> tuple[Any, Any]:
    """Compiled grip-clamped chain (gravity along club ``-z``) at rest."""
    from src.engines.physics_engines.mujoco.python.shaft_chain_mjcf import (
        export_clamped_shaft_mjcf,
    )

    document = clamped_shaft_document(chain, gravity=(0.0, 0.0, -gravity_m_s2))
    model = mujoco.MjModel.from_xml_string(export_clamped_shaft_mjcf(document))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def tip_jacobians(model: Any, data: Any) -> tuple[np.ndarray, np.ndarray]:
    from src.engines.physics_engines.mujoco.python.shaft_chain_mjcf import TIP_SITE

    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, model.site(TIP_SITE).id)
    return jacp, jacr


def chain_static(chain: ShaftChain, gravity: float, force: float) -> np.ndarray:
    """Linear tip ``(deflection m, slope rad)`` under a club-``z`` load."""
    model, data = compile_chain(chain, gravity)
    jacp, jacr = tip_jacobians(model, data)
    generalized = jacp.T @ np.array([0.0, 0.0, force]) - data.qfrc_bias
    q = generalized / model.jnt_stiffness[model.dof_jntid]
    return np.abs([jacp[2] @ q, jacr[0] @ q])


def linear_system(chain: ShaftChain, damped: bool = True, axis: str = "") -> Any:
    """State matrix ``[[0, I], [-M^-1 K, -M^-1 C]]`` of the clamped chain.

    ``axis`` ("x" or "z") keeps one bending family (bending about that club
    axis); the families decouple at zero deflection.
    """
    from src.shared.python.engine_core.mujoco_compat import full_mass_matrix

    model, data = compile_chain(chain)
    names = [model.joint(j).name for j in model.dof_jntid]
    dofs = [d for d, name in enumerate(names) if f"_bend_{axis}" in name]
    mass = full_mass_matrix(mujoco, model, data)[np.ix_(dofs, dofs)]
    mass_inv = np.linalg.inv(mass)
    stiffness = np.diag(model.jnt_stiffness[model.dof_jntid[dofs]])
    damping = np.diag(model.dof_damping[dofs]) * float(damped)
    n = len(dofs)
    return np.block(
        [
            [np.zeros((n, n)), np.eye(n)],
            [-mass_inv @ stiffness, -mass_inv @ damping],
        ]
    )


def first_frequency_hz(chain: ShaftChain, axis: str = "") -> float:
    """Lowest undamped frequency (Hz) of one bending family or of all modes."""
    eigenvalues = np.linalg.eigvals(linear_system(chain, False, axis))
    return float(np.abs(eigenvalues.imag).min() / (2.0 * np.pi))


def rk4_amplification(chain: ShaftChain, step_s: float) -> float:
    """Largest |R(lambda h)| of classical RK4 over the damped chain modes."""
    z = np.linalg.eigvals(linear_system(chain)) * step_s
    return float(np.abs(1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24).max())


# --- beam theory on the same profile -------------------------------------------


@lru_cache(maxsize=16)
def profile_arrays(club: ClubSpec, flex: str) -> tuple[Any, np.ndarray, np.ndarray]:
    profile = station_profile_of(club, flex)
    s = np.linspace(0.0, profile.flexible_length_m, SETTINGS["quadrature_points"])
    ei = np.array([profile.station_at(x).ei_about_x_n_m2 for x in s])
    return profile, s, ei


def beam_tip_load(club: ClubSpec, flex: str) -> tuple[np.ndarray, float]:
    """Euler-Bernoulli tip ``(deflection, slope)`` and Saint-Venant twist."""
    from shared.python.golf_club.shaft_statics import (
        ShaftTipLoad,
        solve_cantilever_tip_response,
    )

    load = ShaftTipLoad(
        force_y_n=SETTINGS["tip_force_n"],
        torque_about_shaft_nm=SETTINGS["tip_torque_n_m"],
    )
    response = solve_cantilever_tip_response(station_profile_of(club, flex), load)
    bending = np.abs([response.deflection_y_m, response.rotation_about_x_rad])
    return bending, abs(response.twist_about_shaft_rad)


@lru_cache(maxsize=16)
def station_profile_of(club: ClubSpec, flex: str) -> Any:
    return station_profile(span_properties(club, flex), club)


def beam_gravity(club: ClubSpec, flex: str) -> np.ndarray:
    """Euler-Bernoulli tip ``(deflection, slope)`` under gravity with the head."""
    profile, s, ei = profile_arrays(club, flex)
    lever = profile.flexible_length_m - s
    mu = profile.stations[0].linear_density_kg_m  # constant by construction
    moment = SETTINGS["gravity_m_s2"] * (club.head_mass_kg * lever + mu * lever**2 / 2)
    return np.array(
        [np.trapezoid(moment * lever / ei, s), np.trapezoid(moment / ei, s)]
    )


def rayleigh_frequency_hz(club: ClubSpec, flex: str) -> float:
    """Rayleigh quotient with the static tip-load shape (head mass + inertia)."""
    from scipy.integrate import cumulative_trapezoid

    profile, s, ei = profile_arrays(club, flex)
    slope = cumulative_trapezoid((profile.flexible_length_m - s) / ei, s, initial=0)
    w = cumulative_trapezoid(slope, s, initial=0.0)
    mu = profile.stations[0].linear_density_kg_m
    # The softer family (largest head inertia about x or z) has the lowest mode.
    gyration = max(club.head_gyration_m[0], club.head_gyration_m[2])
    head_inertia = club.head_mass_kg * gyration**2
    kinetic = (
        np.trapezoid(mu * w**2, s)
        + club.head_mass_kg * w[-1] ** 2
        + head_inertia * slope[-1] ** 2
    )
    return float(np.sqrt(w[-1] / kinetic) / (2.0 * np.pi))


def rel(value: float, reference: float) -> float:
    return abs(value / reference - 1.0)


# --- 1. static cantilever ----------------------------------------------------


def static_errors(n_links: int) -> np.ndarray:
    """Relative errors (tip deflection, tip slope, gravity deflection, slope)."""
    chain = build_shaft_chain(CLUB, n_links)
    tip, _ = beam_tip_load(CLUB, "Stiff")
    chain_tip = chain_static(chain, 0.0, SETTINGS["tip_force_n"])
    chain_gravity = chain_static(chain, SETTINGS["gravity_m_s2"], 0.0)
    return np.r_[chain_tip / tip, chain_gravity / beam_gravity(CLUB, "Stiff")] - 1.0


SLOPE_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="4-link midpoint lumping overstates the tip slope by 1.20 % (tip load) "
    "and 1.04 % (gravity) against Euler-Bernoulli; band 1 % (measured #15150)",
)


@pytest.mark.parametrize(
    "row",
    [0, pytest.param(1, marks=SLOPE_XFAIL), 2, pytest.param(3, marks=SLOPE_XFAIL)],
    ids=["tip_deflection", "tip_slope", "gravity_deflection", "gravity_slope"],
)
def test_default_chain_static_matches_euler_bernoulli(row: int) -> None:
    error = static_errors(DEFAULT_LINKS)[row]
    assert abs(error) < BANDS["static_rel_error_default_links"]


def test_static_error_converges_with_links() -> None:
    errors = np.abs([static_errors(n) for n in SETTINGS["convergence_links"]])
    assert np.all(np.diff(errors, axis=0) < 0.0)  # every quantity improves with N
    assert errors[-1].max() < BANDS["static_rel_error_finest_links"]


# --- 2. clamp frequency per flex ----------------------------------------------


@pytest.mark.parametrize("flex", SETTINGS["flexes"])
def test_cpm_clamp_frequency_meets_the_flex_target(flex: str) -> None:
    club = cpm_clamp_club(CLUB)
    f_chain = first_frequency_hz(build_shaft_chain(club, DEFAULT_LINKS, flex=flex))
    # Continuous Euler-Bernoulli FEM with the head's rotary inertia (softer axis).
    gyration = max(club.head_gyration_m[0], club.head_gyration_m[2])
    properties = span_properties(club, flex)
    f_beam = clamped_head_frequencies(properties, club, club.head_mass_kg * gyration**2)
    excess = rayleigh_frequency_hz(club, flex) / float(f_beam[0]) - 1.0
    assert 0.0 <= excess < BANDS["rayleigh_rel_excess"]  # Rayleigh bounds from above
    assert rel(f_chain, float(f_beam[0])) < BANDS["chain_vs_beam_frequency_rel_error"]
    assert rel(f_chain, flex_frequency_hz(flex)) < BANDS["flex_frequency_rel_error"]


def test_flex_frequencies_are_monotonic() -> None:
    club = cpm_clamp_club(CLUB)
    f = [
        first_frequency_hz(build_shaft_chain(club, DEFAULT_LINKS, flex=flex))
        for flex in SETTINGS["flexes"]
    ]
    assert all(np.diff(f) > 0.0)


# --- 3. convergence and the same-input step ----------------------------------


def test_frequency_converges_with_links() -> None:
    club = cpm_clamp_club(CLUB)
    f = {
        n: first_frequency_hz(build_shaft_chain(club, n))
        for n in SETTINGS["convergence_links"]
    }
    finest = f[max(f)]
    assert rel(f[DEFAULT_LINKS], finest) < BANDS["chain_vs_beam_frequency_rel_error"]


@pytest.mark.parametrize("flex", SETTINGS["flexes"])
def test_default_chain_is_rk4_stable_at_the_same_input_substep(flex: str) -> None:
    chain = build_shaft_chain(CLUB, DEFAULT_LINKS, flex=flex)
    assert rk4_amplification(chain, SUBSTEP_S) <= BANDS["rk4_amplification_max"]


RK4_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="stiffness-proportional damping puts the top damped mode at lambda = "
    "-2.0e5 1/s (N=8) and -4.1e6 1/s (N=16): |R| = 1.5e4 and 2.9e9 at 0.125 ms; "
    "largest stable steps 0.014 and 0.0007 ms (Stiff, measured #15150)",
)


@pytest.mark.parametrize(
    "n_links",
    [2, 4, pytest.param(8, marks=RK4_XFAIL), pytest.param(16, marks=RK4_XFAIL)],
)
def test_refined_chain_rk4_stability(n_links: int) -> None:
    chain = build_shaft_chain(CLUB, n_links)
    assert rk4_amplification(chain, SUBSTEP_S) <= BANDS["rk4_amplification_max"]


# --- 4. undamped free vibration: frequency and energy -------------------------


def free_vibration(chain: ShaftChain) -> tuple[float, float]:
    """RK4 free vibration (damping off): ``(frequency Hz, energy drift)``."""
    model, data = compile_chain(chain)
    model.dof_damping[:] = 0.0
    model.opt.timestep = SUBSTEP_S
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    jacp, _ = tip_jacobians(model, data)
    shape = jacp[2] / model.jnt_stiffness[model.dof_jntid]
    data.qpos[:] = shape * SETTINGS["free_vibration_amplitude_m"] / (jacp[2] @ shape)
    mujoco.mj_forward(model, data)
    steps = round(SETTINGS["free_vibration_duration_s"] / SUBSTEP_S)
    tip, energy = np.empty(steps), np.empty(steps)
    for k in range(steps):
        mujoco.mj_step(model, data)
        tip[k], energy[k] = jacp[2] @ data.qpos, data.energy.sum()
    up = np.flatnonzero((tip[:-1] < 0.0) & (tip[1:] >= 0.0))
    crossings = up + tip[up] / (tip[up] - tip[up + 1])
    period_s = np.diff(crossings).mean() * SUBSTEP_S
    return 1.0 / period_s, float(np.abs(energy / energy[0] - 1.0).max())


def test_free_vibration_matches_eigenvalue_and_conserves_energy() -> None:
    chain = build_shaft_chain(CLUB, DEFAULT_LINKS)
    frequency, drift = free_vibration(chain)
    eigen = first_frequency_hz(chain, "x")  # club-z tip motion: bending about x
    assert rel(frequency, eigen) < BANDS["time_domain_vs_eigen_rel_error"]
    assert drift < BANDS["energy_drift_fraction"]


# --- 5. torsion ---------------------------------------------------------------


def test_torsion_matches_saint_venant() -> None:
    chain = build_shaft_chain(CLUB, DEFAULT_LINKS, torsion=True)
    twist = SETTINGS["tip_torque_n_m"] * sum(1.0 / k for k in chain.twist_stiffness)
    _, theory = beam_tip_load(CLUB, "Stiff")
    assert rel(twist, theory) < BANDS["torsion_rel_error_default_links"]


@pytest.mark.xfail(
    strict=True,
    reason="isotropic GJ = E I / (1 + nu) gives 2.29 deg under 1 ft lbf over the "
    "5 in clamp span (Stiff); band 2.5-7 deg (measured #15150)",
)
def test_torque_rating_is_in_the_published_driver_range() -> None:
    _, twist_rad = beam_tip_load(cpm_clamp_club(CLUB), "Stiff")
    low, high = BANDS["torque_rating_deg"]
    assert low <= np.degrees(twist_rad) <= high
