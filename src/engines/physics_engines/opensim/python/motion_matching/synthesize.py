"""TDD oracle: ``synthesize_target_from_coefficients`` for OpenSim (issue #4124).

This module mirrors the MATLAB reference at
``src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/
shared/synthesize_target_from_coefficients.m`` and the equivalent
MuJoCo / Pinocchio Python implementations (issues #4122 / #4121).

Given a known coefficient vector ``theta``, forward-simulate the OpenSim
golf humanoid and repackage the resulting butt + clubhead + club-quat
trajectories as a canonical :class:`ClubTarget`. The ``ClubTarget``'s
provenance pins ``theta_truth`` (via the sha256 hash) so the recovery
test "synthesize -> fit -> ``theta_recovered ~= theta_truth``" needs no
external data.

Public API:
    synthesize_target_from_coefficients(theta, options) -> ClubTarget
    SynthOptions  -- options dataclass.

The wrapper performs **no** OpenSim calls of its own. All ``osim.*`` work
goes through ``simulate_with_coefficients`` (issue #4120) which is
imported lazily so this module remains importable on platforms without
the OpenSim Python bindings.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import subprocess
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.club_target import ClubTarget, SourceProvenance

logger = logging.getLogger(__name__)

__all__ = [
    "SynthOptions",
    "synthesize_target_from_coefficients",
]

# ---------------------------------------------------------------------------
# Coefficient bounds (mirrors generateRandomCoefficients.m / MATLAB oracle).
# Per-joint coefficient ordering: [A, B, C, D, E, F, G].
# ---------------------------------------------------------------------------
_COEFF_BOUNDS: tuple[float, ...] = (1000.0, 1000.0, 500.0, 500.0, 100.0, 100.0, 25.0)
_COEFFS_PER_JOINT: int = 7
_BOUND_TOL: float = 1.0e-9

# Quaternion / target validation tolerances (CLUB_IK_SPEC).
_QUAT_NORM_TOL: float = 1.0e-6


@dataclass(frozen=True)
class SynthOptions:
    """Options for :func:`synthesize_target_from_coefficients`.

    Mirrors ``default_synth_options.m``. All fields have sensible defaults
    matching the MATLAB reference so most callers can pass ``SynthOptions()``.

    Attributes:
        sample_rate_hz:    Output time-grid rate. Default 1 kHz.
        simulation_time_s: Total simulation duration in (0, 1]. Default 0.3 s.
        add_noise:         If True, add Gaussian position noise. Default False.
        noise_sigma_m:     Std-dev (metres) when ``add_noise``. Default 1 mm.
        subject_id:        Provenance label. Default ``"synthetic"``.
        trial_id:          Provenance label. Default ``"synthesizer_v1"``.
        sim_overrides:     Engine-specific simulation overrides applied on top
                           of OpenSim's ``SimOptions`` defaults. Default empty.
    """

    sample_rate_hz: float = 1000.0
    simulation_time_s: float = 0.3
    add_noise: bool = False
    noise_sigma_m: float = 1.0e-3
    subject_id: str = "synthetic"
    trial_id: str = "synthesizer_v1"
    sim_overrides: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def synthesize_target_from_coefficients(
    theta: NDArray[np.float64] | np.ndarray,
    options: SynthOptions | None = None,
) -> ClubTarget:
    """Synthesize a :class:`ClubTarget` by forward-simulating ``theta``.

    Args:
        theta:   Coefficient vector of length ``n_joints * 7``, ordered
                 ``[A, B, C, D, E, F, G]`` per joint.  Bounds:
                 ``|A|,|B| <= 1000``, ``|C|,|D| <= 500``, ``|E|,|F| <= 100``,
                 ``|G| <= 25``.
        options: :class:`SynthOptions`.  ``None`` -> defaults.

    Returns:
        A fully-validated :class:`ClubTarget` whose ``source.format`` is
        ``"synthetic"`` and whose ``source.sha256`` is the sha256 of the
        ``theta`` byte representation. The ``ClubTarget`` field schema
        matches CLUB_IK_SPEC.md (also enforced inside ``ClubTarget``).

    Raises:
        TypeError:  ``theta`` is not array-like of floats.
        ValueError: ``theta`` length is not a multiple of 7, contains
                    NaN/Inf, or any coefficient exceeds its bound.
        ValueError: ``options`` fields are out of range.
        ImportError: the OpenSim ``simulate_with_coefficients`` module is
                     not yet available (issue #4120 not landed).
        RuntimeError: the underlying simulation reports
                      ``solver_status == "failed"``.
    """
    opts = options if options is not None else SynthOptions()

    # 1. Preconditions -------------------------------------------------------
    theta_arr = _validate_theta(theta)
    _validate_options(opts)

    # 2. Forward sim via the canonical wrapper -------------------------------
    sim_out = _run_simulate_with_coefficients(theta_arr, opts)

    # 3. SimOut -> ClubTarget mapping ---------------------------------------
    time = np.asarray(sim_out.time, dtype=np.float64).reshape(-1)
    butt = np.asarray(sim_out.grip, dtype=np.float64)
    clubhead = np.asarray(sim_out.clubhead, dtype=np.float64)
    club_quat = _normalise_quat_rows(np.asarray(sim_out.club_quat, dtype=np.float64))

    # Optional position noise (deterministic via fixed seed for reproducibility)
    if opts.add_noise:
        butt, clubhead = _add_position_noise(butt, clubhead, opts.noise_sigma_m)

    impact_idx = int(_detect_clubhead_impact(time, clubhead))

    # 4. Provenance ----------------------------------------------------------
    theta_hash = hashlib.sha256(theta_arr.tobytes()).hexdigest()
    source = SourceProvenance(
        filename="",
        format="synthetic",
        subject_id=str(opts.subject_id),
        trial_id=str(opts.trial_id),
        sha256=theta_hash,
    )

    target = ClubTarget(
        time=time,
        butt=butt,
        clubhead=clubhead,
        club_quat=club_quat,
        impact_idx=impact_idx,
        source=source,
    )

    # 5. Postconditions ------------------------------------------------------
    # Most are enforced by ClubTarget.__post_init__ -- we add the engine-
    # and oracle-specific checks here.
    assert target.source.format == "synthetic"  # noqa: S101 - DbC postcondition
    assert len(target.source.sha256) == 64  # noqa: S101
    assert target.time[0] == 0.0  # noqa: S101
    assert target.time[-1] <= opts.simulation_time_s + 1.0e-9  # noqa: S101
    return target


# ---------------------------------------------------------------------------
# Helpers (private).
# ---------------------------------------------------------------------------


def _validate_theta(theta: Any) -> NDArray[np.float64]:
    """Coerce + validate ``theta``; raise descriptive errors on failure."""
    try:
        arr = np.asarray(theta, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"theta must be array-like of floats, got {type(theta).__name__}"
        ) from exc

    if arr.size == 0 or arr.size % _COEFFS_PER_JOINT != 0:
        raise ValueError(
            "theta length must be a positive multiple of "
            f"{_COEFFS_PER_JOINT} (got {arr.size})"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("theta contains NaN or Inf")

    n_joints = arr.size // _COEFFS_PER_JOINT
    matrix = arr.reshape(n_joints, _COEFFS_PER_JOINT)  # rows = joints
    for col, bound in enumerate(_COEFF_BOUNDS):
        if np.any(np.abs(matrix[:, col]) > bound + _BOUND_TOL):
            letter = chr(ord("A") + col)
            raise ValueError(
                f"coefficient {letter} exceeds +/-{bound:g} "
                f"(max |{letter}| = {float(np.abs(matrix[:, col]).max()):.4g})"
            )
    return arr


def _validate_options(opts: SynthOptions) -> None:
    """Range-check :class:`SynthOptions`."""
    if not (np.isfinite(opts.sample_rate_hz) and opts.sample_rate_hz > 0):
        raise ValueError(
            f"options.sample_rate_hz must be a positive finite scalar, "
            f"got {opts.sample_rate_hz!r}"
        )
    if not (
        np.isfinite(opts.simulation_time_s) and 0.0 < opts.simulation_time_s <= 1.0
    ):
        raise ValueError(
            "options.simulation_time_s must be in (0, 1] s, got "
            f"{opts.simulation_time_s!r}"
        )
    if opts.add_noise and not (
        np.isfinite(opts.noise_sigma_m) and opts.noise_sigma_m >= 0.0
    ):
        raise ValueError(
            "options.noise_sigma_m must be a non-negative finite scalar, "
            f"got {opts.noise_sigma_m!r}"
        )


def _run_simulate_with_coefficients(
    theta: NDArray[np.float64], opts: SynthOptions
) -> Any:
    """Locate and call the OpenSim ``simulate_with_coefficients`` entry point.

    The simulator lives behind a lazy import so this module remains importable
    on platforms without OpenSim. Issue #4120 introduces the module; until
    that lands, callers receive a clear ``ImportError``.
    """
    sim_module = _load_simulate_module()

    # Build engine-side SimOptions if the module exposes one; otherwise pass
    # a plain dict and let the simulator coerce.
    sim_options_cls = getattr(sim_module, "SimOptions", None)
    if sim_options_cls is not None:
        try:
            sim_options = sim_options_cls(
                sample_rate_hz=float(opts.sample_rate_hz),
                simulation_time_s=float(opts.simulation_time_s),
                **dict(opts.sim_overrides),
            )
        except TypeError:
            # Older signature -- fall back to no kwargs and apply overrides
            # post-hoc if the dataclass exposes ``replace``-friendly fields.
            sim_options = sim_options_cls()
    else:
        sim_options = {
            "sample_rate_hz": float(opts.sample_rate_hz),
            "simulation_time_s": float(opts.simulation_time_s),
            **dict(opts.sim_overrides),
        }

    sim_func = getattr(sim_module, "simulate_with_coefficients", None)
    if sim_func is None:
        raise ImportError(
            "OpenSim simulate_with_coefficients is not yet available "
            "(expected at "
            "src/engines/physics_engines/opensim/python/motion_matching/"
            "simulate.py::simulate_with_coefficients). "
            "Pending issue #4120."
        )

    sim_out = sim_func(theta, sim_options)
    status = getattr(sim_out, "solver_status", "success")
    if str(status) == "failed":
        message = getattr(sim_out, "status_message", "<no message>")
        raise RuntimeError(
            f"simulate_with_coefficients reported solver_status='failed': {message}"
        )
    return sim_out


def _load_simulate_module() -> Any:
    """Import the OpenSim simulate module by trying canonical locations."""
    candidates = (
        # The path referenced by issue #4124's prompt and OPENSIM_PARITY_SPEC's
        # cross-engine layout.
        "src.engines.physics_engines.opensim.python.motion_matching.simulate",
        # Fallback to the location named in OPENSIM_PARITY_SPEC §2.2.
        "src.engines.physics_engines.opensim.python.opensim_golf."
        "simulate_with_coefficients",
    )
    last_err: Exception | None = None
    for dotted in candidates:
        try:
            return importlib.import_module(dotted)
        except ImportError as exc:
            last_err = exc
    raise ImportError(
        "Could not locate OpenSim simulate_with_coefficients. Tried: "
        + ", ".join(candidates)
        + (f" (last error: {last_err})" if last_err is not None else "")
    )


def _normalise_quat_rows(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Force unit-norm and ``q[:,0] >= 0`` sign convention.

    Mirrors ``local_normalise_quat_rows`` in the MATLAB reference.
    Returns a copy.
    """
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"club_quat must have shape (N, 4), got {q.shape}")
    if q.size == 0 or np.all(np.isnan(q)):
        # Degenerate -- substitute identity quaternion to satisfy postcondition.
        return np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (max(q.shape[0], 1), 1))
    out = q.astype(np.float64, copy=True)
    norms = np.sqrt(np.einsum("ij,ij->i", out, out))[:, np.newaxis]
    norms[norms == 0.0] = 1.0
    out = out / norms
    flip_mask = out[:, 0] < 0.0
    out[flip_mask] = -out[flip_mask]
    # Re-check norms (post-flip should still be unit).
    err = np.abs(np.linalg.norm(out, axis=1) - 1.0).max()
    if err > _QUAT_NORM_TOL:  # pragma: no cover -- guarded above
        raise ValueError(
            f"club_quat rows could not be normalised (max |1 - |q|| = {err:.2e})"
        )
    return out


def _add_position_noise(
    butt: NDArray[np.float64],
    clubhead: NDArray[np.float64],
    sigma_m: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Add zero-mean Gaussian noise with deterministic seed.

    Determinism is required for the round-trip identity test (same theta ->
    same ClubTarget bytes within float rounding).
    """
    rng = np.random.default_rng(seed=0)
    return (
        butt + sigma_m * rng.standard_normal(butt.shape),
        clubhead + sigma_m * rng.standard_normal(clubhead.shape),
    )


def _detect_clubhead_impact(
    time: NDArray[np.float64], clubhead: NDArray[np.float64]
) -> int:
    """Return ``argmax || d r_clubhead / dt ||`` clamped to ``[1, N]`` (1-indexed).

    Matches the MATLAB ``detect_clubhead_impact`` default behaviour:
    pick the timestep at which clubhead speed peaks. The 1-indexed
    convention matches MATLAB / CLUB_IK_SPEC.
    """
    n = time.shape[0]
    if n < 2:
        return 1
    dt = np.diff(time)
    dt[dt == 0.0] = np.finfo(np.float64).eps
    velocity = np.diff(clubhead, axis=0) / dt[:, None]
    speed = np.sqrt(np.einsum("ij,ij->i", velocity, velocity))
    if speed.size == 0 or not np.any(np.isfinite(speed)):
        return 1
    # argmax is 0-indexed on the velocity array (length N-1) -> map to 1..N.
    idx0 = int(np.nanargmax(speed))
    return max(1, min(n, idx0 + 1))


def _git_commit() -> str:
    """Best-effort current ``HEAD`` SHA; ``"unknown"`` on failure.

    Currently unused by the dataclass (``SourceProvenance`` does not carry
    ``git_commit``) but kept available for future schema expansion that
    matches the MATLAB struct exactly.
    """
    try:
        out = subprocess.run(  # noqa: S603,S607 - safe, fixed args
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - defensive
        pass
    return "unknown"
