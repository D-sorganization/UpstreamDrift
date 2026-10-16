"""Declared disturbance processes for variability experiments (NSX-82, #12711).

Every variability experiment constructs its disturbance through one
:class:`DisturbanceSpec` (epic #12593 rule V6: disturbances are declared, not
assumed).  The specification this implements is
``docs/research/loop_nullspace/loop_nullspace.tex``:

* ``sec:var-cov``: noise amplitudes are per-step variances and must be
  timestep-consistent; a continuous white spectral density ``S_w`` becomes
  ``Sigma_w = S_w / dt``.
* ``sec:var-mechanisms`` (M1): signal-dependent noise
  ``sigma_i^2 = sigma_{0,i}^2 + k_i^2 u_i^2``.

Conventions (declared choices where the specification fixes none):

* ``covariance`` is the *two-sided* spectral density level ``S0`` in
  ``units^2 * s`` for a per-step disturbance, and the plain covariance in
  ``units^2`` for a persistent one (a per-trial constant random state, e.g.
  MOSAIC parameter uncertainty, epic rule R14).
* White: per-step covariance ``S0 / dt``.  Ornstein-Uhlenbeck: PSD
  ``S0 / (1 + (2 pi f tau)^2)`` with ``tau = 1 / (2 pi bandwidth_hz)``,
  stationary variance ``S0 / (2 tau)``, sampled with the exact AR(1)
  discretisation ``phi = exp(-dt / tau)`` so the variance is independent of
  ``dt``.  Pink: PSD ``S0 * (1 Hz) / |f|`` inside ``band_hz`` and zero outside,
  synthesised by spectral summation on the bins ``k / (n_steps dt)``.
* Cross-channel structure is ``covariance``; the signal-dependent law adds
  ``diag(k_i^2 u_i^2)`` (independent per channel), so a diagonal ``covariance``
  recovers the specification's ``sigma_{0,i}^2 + k_i^2 u_i^2`` exactly.  The
  law is supported for white and persistent disturbances only.
* ``signal_bounds`` are the admissible range of the nominal signal ``u`` that
  drives the law; a signal outside them is rejected.

Non-claims: amplitudes are declared hypotheses tied to
``H-NOISE-ADDITIVE`` / ``H-NOISE-SIGNAL-DEPENDENT``, not measured golfer noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["DisturbanceSpec", "InjectionStage", "Spectrum"]

_SYMMETRY_RTOL = 1e-9
_PSD_RTOL = 1e-9


class InjectionStage(str, Enum):
    """Where in the model the disturbance enters."""

    TORQUE = "torque"
    ACTIVATION = "activation"
    INITIAL_STATE = "initial_state"
    TIMING = "timing"
    GRIP_SPLIT = "grip_split"
    EXTERNAL_HAND_WRENCH = "external_hand_wrench"


class Spectrum(str, Enum):
    """Spectral shape of a per-step disturbance."""

    WHITE = "white"
    PINK = "pink"
    ORNSTEIN_UHLENBECK = "ornstein_uhlenbeck"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _readonly(arr: np.ndarray) -> np.ndarray:
    out = np.array(arr, dtype=float, copy=True)
    out.flags.writeable = False
    return out


def _validate_covariance(cov: np.ndarray, n: int) -> None:
    _require(
        cov.shape == (n, n),
        f"covariance shape {cov.shape} does not match {n} channels (expected {(n, n)})",
    )
    _require(bool(np.isfinite(cov).all()), "covariance must be finite")
    scale = max(1.0, float(np.abs(cov).max()))
    _require(
        bool(np.abs(cov - cov.T).max() <= _SYMMETRY_RTOL * scale),
        "covariance must be symmetric",
    )
    eigmin = float(np.linalg.eigvalsh(0.5 * (cov + cov.T)).min())
    _require(
        eigmin >= -_PSD_RTOL * scale,
        f"covariance must be positive semi-definite (min eigenvalue {eigmin:.3e})",
    )


@dataclass(frozen=True, eq=False)
class DisturbanceSpec:
    """One declared disturbance model (frozen value object).

    Design by Contract
    ------------------
    Pre: ``channels`` unique, ``units`` and ``name`` non-empty.
    Pre: ``covariance`` finite, symmetric, positive semi-definite, ``n x n``.
    Pre: ``persistent`` iff ``spectrum is None``.
    Pre: OU needs ``bandwidth_hz > 0``; pink needs ``0 < band_hz[0] < band_hz[1]``;
         white takes neither.
    Pre: ``signal_gain >= 0`` per channel with ``signal_bounds`` (lo < hi) and
         a white or persistent disturbance.
    Post: the instance is immutable and ``covariance`` is a read-only copy.
    """

    name: str
    stage: InjectionStage
    channels: tuple[str, ...]
    units: str
    covariance: np.ndarray
    spectrum: Spectrum | None = None
    persistent: bool = False
    bandwidth_hz: float | None = None
    band_hz: tuple[float, float] | None = None
    signal_gain: tuple[float, ...] | None = None
    signal_bounds: tuple[tuple[float, float], ...] | None = None
    _chol: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.stage, InjectionStage):
            raise TypeError(f"stage must be an InjectionStage, got {self.stage!r}")
        if self.spectrum is not None and not isinstance(self.spectrum, Spectrum):
            raise TypeError(
                f"spectrum must be a Spectrum or None, got {self.spectrum!r}"
            )
        _require(bool(self.name), "name must be a non-empty string")
        _require(bool(self.units), "units must be a non-empty string")
        n = len(self.channels)
        _require(n >= 1, "channels must name at least one channel")
        _require(len(set(self.channels)) == n, "channel names must be unique")
        cov = np.asarray(self.covariance, dtype=float)
        _validate_covariance(cov, n)
        object.__setattr__(self, "covariance", _readonly(cov))
        self._validate_spectrum()
        self._validate_signal_law(n)
        vals, vecs = np.linalg.eigh(0.5 * (cov + cov.T))
        factor = vecs * np.sqrt(np.clip(vals, 0.0, None))
        object.__setattr__(self, "_chol", _readonly(factor))

    # ----------------------------------------------------------- validation
    def _validate_spectrum(self) -> None:
        if self.persistent:
            _require(
                self.spectrum is None,
                "persistent disturbances must declare spectrum=None",
            )
        else:
            _require(
                self.spectrum is not None,
                "spectrum None requires persistent=True (a per-trial constant)",
            )
        ou = self.spectrum is Spectrum.ORNSTEIN_UHLENBECK
        if ou:
            bw = self.bandwidth_hz
            _require(bw is not None, "bandwidth_hz required for Ornstein-Uhlenbeck")
            _require(
                bw is not None and math.isfinite(bw) and bw > 0,
                "bandwidth_hz must be positive and finite",
            )
        else:
            _require(
                self.bandwidth_hz is None,
                "bandwidth_hz is only valid for Ornstein-Uhlenbeck",
            )
        if self.spectrum is Spectrum.PINK:
            band = self.band_hz
            _require(band is not None, "band_hz required for pink spectrum")
            _require(
                band is not None
                and all(math.isfinite(f) for f in band)
                and 0 < band[0] < band[1],
                "band_hz must satisfy 0 < low < high (finite, Hz)",
            )
        else:
            _require(self.band_hz is None, "band_hz is only valid for pink spectrum")

    def _validate_signal_law(self, n: int) -> None:
        if self.signal_gain is None:
            _require(
                self.signal_bounds is None,
                "signal_bounds given without signal_gain",
            )
            return
        _require(
            len(self.signal_gain) == n,
            f"signal_gain needs one entry per channel ({n})",
        )
        _require(
            all(math.isfinite(k) and k >= 0 for k in self.signal_gain),
            "signal_gain must be finite and non-negative",
        )
        _require(
            self.signal_bounds is not None, "signal_bounds required with signal_gain"
        )
        bounds = self.signal_bounds or ()
        _require(
            len(bounds) == n, f"signal_bounds needs one (lo, hi) per channel ({n})"
        )
        _require(
            all(
                math.isfinite(lo) and math.isfinite(hi) and lo < hi for lo, hi in bounds
            ),
            "signal_bounds must be finite with lo < hi per channel",
        )
        _require(
            self.persistent or self.spectrum is Spectrum.WHITE,
            "signal-dependent law is supported for white or persistent "
            "disturbances only",
        )

    # --------------------------------------------------------- derived data
    @property
    def n_channels(self) -> int:
        """Number of disturbance channels."""
        return len(self.channels)

    @property
    def has_signal_law(self) -> bool:
        """Whether the M1 signal-dependent law is declared."""
        return self.signal_gain is not None

    @property
    def covariance_units(self) -> str:
        """Unit string of ``covariance`` (spectral density or plain variance)."""
        base = f"({self.units})^2"
        return base if self.persistent else f"{base}*s"

    @property
    def ou_time_constant(self) -> float:
        """OU correlation time ``tau = 1 / (2 pi bandwidth_hz)`` in seconds."""
        _require(
            self.spectrum is Spectrum.ORNSTEIN_UHLENBECK,
            "ou_time_constant needs an Ornstein-Uhlenbeck spectrum",
        )
        assert self.bandwidth_hz is not None
        return 1.0 / (2.0 * math.pi * self.bandwidth_hz)

    def _pink_bins(self, dt: float, n_steps: int) -> tuple[np.ndarray, float]:
        assert self.band_hz is not None
        lo, hi = self.band_hz
        _require(
            hi < 0.5 / dt,
            f"pink band_hz upper edge {hi} Hz must be below the Nyquist "
            f"frequency {0.5 / dt} Hz",
        )
        df = 1.0 / (n_steps * dt)
        freqs = np.arange(1, n_steps // 2) * df
        sel = freqs[(freqs >= lo) & (freqs <= hi)]
        _require(sel.size > 0, "no frequency bin falls inside pink band_hz")
        return sel, df

    def pink_band_variance_factor(self, dt: float, n_steps: int) -> float:
        """Ensemble variance of a unit-``S0`` pink channel for this grid.

        ``sum_k 2 * (1 Hz / f_k) * df`` over the bins inside ``band_hz``;
        tends to ``2 ln(high / low)`` as the grid refines.
        """
        _require(self.spectrum is Spectrum.PINK, "needs a pink spectrum")
        _check_grid(1, n_steps, dt)
        freqs, df = self._pink_bins(dt, n_steps)
        return float(np.sum(2.0 * df / freqs))

    # ----------------------------------------------------------- covariance
    def _check_signal(self, signal: ArrayLike | None) -> np.ndarray | None:
        if not self.has_signal_law:
            _require(
                signal is None, "signal given but no signal-dependent law declared"
            )
            return None
        _require(
            signal is not None, "signal required: a signal-dependent law is declared"
        )
        u = np.asarray(signal, dtype=float)
        _require(bool(np.isfinite(u).all()), "signal must be finite")
        _require(
            u.shape[-1] == self.n_channels and u.ndim <= 2,
            f"signal must end in {self.n_channels} channels",
        )
        bounds = np.asarray(self.signal_bounds, dtype=float)
        inside = (u >= bounds[:, 0]) & (u <= bounds[:, 1])
        _require(bool(inside.all()), "signal outside declared signal_bounds")
        return u

    def covariance_at(self, signal: ArrayLike | None = None) -> np.ndarray:
        """Covariance at one signal value (spectral-density or persistent units).

        Pre: ``signal`` has shape ``(n_channels,)`` iff a law is declared and
        lies inside ``signal_bounds``.  Post: ``covariance + diag(k^2 u^2)``
        (M1, ``sec:var-mechanisms``), symmetric positive semi-definite.
        """
        u = self._check_signal(signal)
        if u is None:
            return np.array(self.covariance)
        _require(u.ndim == 1, "covariance_at takes one signal vector")
        gain = np.asarray(self.signal_gain, dtype=float)
        return self.covariance + np.diag((gain * u) ** 2)

    def discrete_covariance(
        self, dt: float, signal: ArrayLike | None = None
    ) -> np.ndarray:
        """Covariance of one white per-step draw, or of the persistent state.

        ``S_w / dt`` for white (``sec:var-cov``), unchanged for persistent.
        Pre: ``dt > 0`` and finite; spectrum is white or persistent.
        """
        _require(math.isfinite(dt) and dt > 0, f"dt must be positive, got {dt}")
        _require(
            self.persistent or self.spectrum is Spectrum.WHITE,
            "discrete_covariance is defined for white or persistent only",
        )
        cov = self.covariance_at(signal)
        return cov if self.persistent else cov / dt

    # -------------------------------------------------------------- sampling
    def sample(
        self,
        n_trials: int,
        n_steps: int,
        dt: float,
        seed: int | None = None,
        signal: ArrayLike | None = None,
    ) -> np.ndarray:
        """Draw disturbance realisations.

        Returns shape ``(n_trials, n_channels)`` for a persistent disturbance
        (one constant random state per trial, no time axis) and
        ``(n_trials, n_steps, n_channels)`` otherwise, in ``units``.

        Pre: ``n_trials, n_steps >= 1``, ``dt > 0``; ``signal`` iff a law is
        declared (shape ``(n_channels,)`` or ``(n_steps, n_channels)`` for
        white; ``(n_channels,)`` for persistent).  Post: finite output;
        identical for an identical ``seed``; per-step variance is
        independent of ``dt`` once integrated (``sec:var-cov``).
        """
        _check_grid(n_trials, n_steps, dt)
        u = self._check_signal(signal)
        rng = np.random.default_rng(seed)
        if self.persistent:
            out = self._sample_gaussian(rng, (n_trials,), u, 1.0)
        elif self.spectrum is Spectrum.WHITE:
            out = self._sample_gaussian(rng, (n_trials, n_steps), u, dt)
        elif self.spectrum is Spectrum.ORNSTEIN_UHLENBECK:
            out = self._sample_ou(rng, n_trials, n_steps, dt)
        else:
            out = self._sample_pink(rng, n_trials, n_steps, dt)
        if not np.isfinite(out).all():
            raise ValueError("sampled disturbance is not finite")
        return out

    def _sample_gaussian(
        self,
        rng: np.random.Generator,
        lead: tuple[int, ...],
        u: np.ndarray | None,
        dt: float,
    ) -> np.ndarray:
        shape = (*lead, self.n_channels)
        out = rng.standard_normal(shape) @ self._chol.T
        if u is not None:
            gain = np.asarray(self.signal_gain, dtype=float)
            if u.ndim == 2:
                _require(
                    len(lead) == 2 and u.shape[0] == lead[1],
                    "per-step signal needs shape (n_steps, n_channels)",
                )
            out = out + gain * u * rng.standard_normal(shape)
        return out / math.sqrt(dt)

    def _sample_ou(
        self, rng: np.random.Generator, n_trials: int, n_steps: int, dt: float
    ) -> np.ndarray:
        tau = self.ou_time_constant
        phi = math.exp(-dt / tau)
        var = 1.0 / (2.0 * tau)  # stationary variance of a unit-S0 channel
        innov = math.sqrt(var * (1.0 - phi * phi))
        shape = (n_trials, self.n_channels)
        y = np.empty((n_trials, n_steps, self.n_channels))
        state = math.sqrt(var) * rng.standard_normal(shape)
        for k in range(n_steps):
            y[:, k, :] = state
            state = phi * state + innov * rng.standard_normal(shape)
        return y @ self._chol.T

    def _sample_pink(
        self, rng: np.random.Generator, n_trials: int, n_steps: int, dt: float
    ) -> np.ndarray:
        freqs, df = self._pink_bins(dt, n_steps)
        amp = np.sqrt(2.0 * df / freqs)  # sqrt(2 S_k df), S_k = 1 Hz / f_k
        shape = (n_trials, self.n_channels, freqs.size)
        xi = rng.standard_normal(shape)
        eta = rng.standard_normal(shape)
        spec = np.zeros((n_trials, self.n_channels, n_steps // 2 + 1), dtype=complex)
        first = int(round(freqs[0] / df))
        spec[..., first : first + freqs.size] = 0.5 * n_steps * amp * (xi - 1j * eta)
        y = np.fft.irfft(spec, n=n_steps, axis=-1)  # (trials, channels, steps)
        return np.swapaxes(y, 1, 2) @ self._chol.T


def _check_grid(n_trials: int, n_steps: int, dt: float) -> None:
    _require(n_trials >= 1, f"n_trials must be >= 1, got {n_trials}")
    _require(n_steps >= 1, f"n_steps must be >= 1, got {n_steps}")
    _require(math.isfinite(dt) and dt > 0, f"dt must be positive and finite, got {dt}")
