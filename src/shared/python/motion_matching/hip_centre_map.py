"""Maximum-a-posteriori hip joint centre estimation with a covariance (HRG-C2).

``hip_calibration.functional_hip_calibration`` fits a *free* sphere to each
knee marker track in a pelvis frame. Nothing in that fit says how well the
centre is determined: on the best capture it returned an inter-HJC distance of
220 mm against a stature prior of 178 mm, with a sphere standard deviation of
only 2 mm. This module keeps the same physical model but makes the
uncertainty explicit and combines it with a declared prior.

Model (per side s, pelvis-frame points p_i, i = 1..n_s):

    |p_i - c_s| = r_s + e_i,   e_i ~ N(0, sigma_s^2)

so the knee marker of a rigid femur moves on a sphere of radius r_s about the
hip joint centre c_s. A femur-fixed marker held off the knee axis (the lateral
epicondyle marker is about 60 mm out) is still a point rigidly attached to the
femur, so it stays on a sphere about the true centre: the lateral offset
changes the radius, not the centre (see the synthetic test
``test_rigid_femur_lateral_marker_does_not_bias_the_centre``). No
medialisation is applied.

The estimator is a Gauss-Newton / Levenberg-Marquardt solve of the stacked
least-squares problem

    min  sum_s sum_i ((|p_i - c_s| - r_s)/sigma_s)^2 + (x - mu)^T L (x - mu)

with x = (c_R, c_L) and a Gaussian prior N(mu, L^-1) from
:class:`HipCentrePrior`. The posterior covariance is the inverse of the final
Gauss-Newton information matrix, with the radii as nuisance parameters. The
prior is a declared assumption, never data: ``prior_weight`` reports the share
of the posterior information that comes from it.

Status thresholds (all documented constants, see below): ``degenerate`` when a
knee arc subtends less than ``MIN_ARC_SPAN_DEG`` or the data information of a
side is ill-conditioned; ``prior_dominated`` when more than
``PRIOR_DOMINATED_WEIGHT`` of the posterior information is the prior;
otherwise ``identified``.

The estimator takes pelvis-frame knee tracks, so it is engine- and
capture-neutral. ``hip_calibration.pelvis_frame_tracks`` produces those from a
capture with a qualified waist marker set (:func:`estimate_hip_centres_from_capture`).

Limitations: frames of a swing are autocorrelated, so the independent-noise
covariance is optimistic; ``frames_per_independent_sample`` rescales it. The
Harrington (2007) pelvic-width proxy is not implemented (it needs ASIS
markers or a documented waist-marker regression); the stature prior is the
only prior. Software correctness only; scientific qualification is separate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.anthropometric_geometry import (
    HIP_HALF_WIDTH_FRACTION_OF_STATURE,
)
from src.shared.python.motion_matching.hip_calibration import (
    fit_sphere,
    pelvis_frame_tracks,
)

Array: TypeAlias = NDArray[np.float64]
Vec3: TypeAlias = tuple[float, float, float]

#: Full cone angle of a knee arc below which the sphere centre is not identified.
MIN_ARC_SPAN_DEG = 20.0
#: Numerical-singularity guard: smallest allowed ratio of the weakest to strongest data-information
#: eigenvalue (centre and radius, one side) before a side is degenerate.
MIN_INFO_EIGEN_RATIO = 1.0e-8
#: Share of posterior information from the prior above which the result is
#: reported as ``prior_dominated``.
PRIOR_DOMINATED_WEIGHT = 0.5
#: Floor on the per-side noise standard deviation (m); avoids an infinitely
#: confident data term for noiseless synthetic data.
MIN_NOISE_SD_M = 1.0e-6
MIN_FRAMES = 8
STATUSES = ("identified", "prior_dominated", "degenerate")
_MAX_ITERATIONS = 100


@dataclass(frozen=True)
class HipCentrePrior:
    """Declared Gaussian prior on the two hip centres in the builder Hip frame.

    Frame: x forward, y left, z up (``pelvis_alignment_for``). The right
    centre sits at ``-half_width`` on y, the left at ``+half_width``;
    ``half_width = HIP_HALF_WIDTH_FRACTION_OF_STATURE * stature`` (inter-HJC
    0.104 x stature, 178 mm at 1.71 m). The builder places both centres at
    x = z = 0, which is the default AP/SI mean of the midpoint.

    Uncertainty is stated on the midpoint ``m = (c_R + c_L)/2`` and the
    half-vector ``h = (c_R - c_L)/2``, independent, per coordinate (x, y, z):

    * ``half_width_sd_m`` (y of ``h``): 3 mm per side, so 6 mm on the
      inter-HJC distance.
    * ``half_ap_si_sd_m`` (x and z of ``h``): 15 mm, about +-5 degrees of
      hip-line yaw/roll at 178 mm; keeps a level, frontal hip line only as a
      weak preference.
    * ``mid_sd_m``: (50, 10, 50) mm, because the waist-frame origin is only
      loosely tied to the joint (the best fit has the midpoint 79 mm below
      it); the lateral midpoint is tied tighter because the waist frame is
      built symmetric.

    These widths are declared assumptions, recorded in ``provenance``.
    """

    stature_m: float
    half_width_fraction: float = HIP_HALF_WIDTH_FRACTION_OF_STATURE
    half_width_sd_m: float = 0.003
    half_ap_si_sd_m: float = 0.015
    mid_mean_m: Vec3 = (0.0, 0.0, 0.0)
    mid_sd_m: Vec3 = (0.05, 0.01, 0.05)
    provenance: str = (
        "stature prior: inter-HJC distance 0.104 x stature "
        "(anthropometric_geometry.HIP_HALF_WIDTH_FRACTION_OF_STATURE), "
        "declared SDs (3 mm half-width, 15 mm AP/SI half-vector, 50/10/50 mm "
        "midpoint); no pelvic-width proxy from the waist markers"
    )

    @classmethod
    def from_stature(cls, stature_m: float) -> HipCentrePrior:
        """Stature prior. Raises ``ValueError`` unless ``stature_m`` is finite > 0."""
        if not math.isfinite(stature_m) or stature_m <= 0.0:
            raise ValueError(f"stature must be finite and positive, got {stature_m!r}")
        return cls(stature_m=float(stature_m))

    def __post_init__(self) -> None:
        if not math.isfinite(self.stature_m) or self.stature_m <= 0.0:
            raise ValueError(f"stature must be finite and positive: {self.stature_m!r}")
        sds = (self.half_width_sd_m, self.half_ap_si_sd_m, *self.mid_sd_m)
        if not all(math.isfinite(s) and s > 0.0 for s in sds):
            raise ValueError("prior standard deviations must be finite and positive")

    @property
    def half_width_m(self) -> float:
        return self.half_width_fraction * self.stature_m

    @property
    def mean_r(self) -> Array:
        return np.asarray(self.mid_mean_m, float) + [0.0, -self.half_width_m, 0.0]

    @property
    def mean_l(self) -> Array:
        return np.asarray(self.mid_mean_m, float) + [0.0, self.half_width_m, 0.0]

    def covariance(self) -> Array:
        """6 x 6 covariance of ``(c_R, c_L)``."""
        mid = np.diag(np.square(self.mid_sd_m))
        half = np.diag(
            np.square(
                [self.half_ap_si_sd_m, self.half_width_sd_m, self.half_ap_si_sd_m]
            )
        )
        return np.block([[mid + half, mid - half], [mid - half, mid + half]])


@dataclass(frozen=True)
class Interval:
    """A scalar estimate with its standard deviation and 95% interval."""

    estimate: float
    sd: float
    lo: float
    hi: float


@dataclass(frozen=True)
class SymmetryTest:
    """Test of left/right (AP and SI) agreement of the centres, from the data
    only. ``chi2`` has 2 degrees of freedom; ``p_value = exp(-chi2/2)``. NaN
    when the data do not identify both centres. Symmetry is never imposed."""

    chi2: float
    p_value: float


@dataclass(frozen=True)
class HipCentreEstimate:
    """MAP hip centres (frame of the input tracks) with uncertainty and status.

    ``height_difference_m`` is (right - left) along the superior axis and
    ``roll_deg`` the hip-line roll, positive when the right centre is higher.
    ``prior_weight`` is the fraction of posterior information from the prior
    (0 = all data, 1 = all prior).
    """

    centre_r: Vec3
    centre_l: Vec3
    cov_r: Array
    cov_l: Array
    cov_joint: Array
    inter_hjc_m: Interval
    height_difference_m: Interval
    roll_deg: Interval
    prior_weight: float
    status: str
    status_reason: str
    symmetry: SymmetryTest
    radius_r_m: float
    radius_l_m: float
    noise_sd_r_m: float
    noise_sd_l_m: float
    arc_span_deg_r: float
    arc_span_deg_l: float
    frames_r: int
    frames_l: int
    prior_provenance: str
    waist_residual_rms_m: float
    waist_residual_max_m: float
    used_prior: bool
    frames_per_independent_sample: float


def _as_track(points: Array, name: str) -> Array:
    x = np.asarray(points, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"{name} must be (frames, 3), got shape {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    if x.shape[0] < MIN_FRAMES:
        raise ValueError(f"{name} needs at least {MIN_FRAMES} frames, got {x.shape[0]}")
    return x


def _jacobian_rows(points: Array, centre: Array) -> tuple[Array, Array]:
    """Residuals ``|p - c|`` and rows of d(residual)/d(c, r) (n x 4)."""
    diff = points - centre
    dist = np.linalg.norm(diff, axis=1)
    safe = np.where(dist > 0.0, dist, 1.0)
    jac = np.column_stack([-diff / safe[:, None], -np.ones(len(dist))])
    return dist, jac


def _arc_span_deg(points: Array, centre: Array) -> float:
    """Full cone angle (deg) of the directions from ``centre`` to the points."""
    u = points - centre
    u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
    mean = u.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm < 1e-12:
        return 360.0
    cosines = np.clip(u @ (mean / norm), -1.0, 1.0)
    return float(2.0 * np.degrees(np.arccos(cosines.min())))


def _solve(
    tracks: tuple[Array, Array],
    start: Array,
    sigma: tuple[float, float],
    prior: tuple[Array, Array] | None,
) -> tuple[Array, Array]:
    """LM solve over theta = (c_R, c_L, r_R, r_L); returns (theta, information)."""
    theta = start.copy()
    lam = 1.0e-3

    def system(th: Array) -> tuple[Array, Array, float]:
        grad = np.zeros(8)
        info = np.zeros((8, 8))
        cost = 0.0
        for s in (0, 1):
            cols = [3 * s, 3 * s + 1, 3 * s + 2, 6 + s]
            dist, jac = _jacobian_rows(tracks[s], th[3 * s : 3 * s + 3])
            res = (dist - th[6 + s]) / sigma[s]
            jac = jac / sigma[s]
            grad[cols] += jac.T @ res
            info[np.ix_(cols, cols)] += jac.T @ jac
            cost += float(res @ res)
        if prior is not None:
            diff = th[:6] - prior[0]
            grad[:6] += prior[1] @ diff
            info[:6, :6] += prior[1]
            cost += float(diff @ prior[1] @ diff)
        return grad, info, cost

    grad, info, cost = system(theta)
    for _ in range(_MAX_ITERATIONS):
        step = -np.linalg.solve(info + lam * np.diag(np.diag(info) + 1e-12), grad)
        trial = theta + step
        g2, i2, c2 = system(trial)
        if c2 <= cost:
            done = float(np.max(np.abs(step))) < 1.0e-11
            theta, grad, info, cost = trial, g2, i2, c2
            lam = max(lam * 0.3, 1e-12)
            if done:
                break
        else:
            lam *= 10.0
            if lam > 1e12:
                break
    return theta, info


#: Posterior draws and seed for the derived-quantity intervals. The distance
#: ``|c_R - c_L|`` and the roll are nonlinear in the centres (with a 30 mm
#: AP/SI prior SD against a 6 mm lateral one the curvature of the norm is worth
#: several mm), so a delta-method interval undercovers; draws do not.
DERIVED_DRAWS = 4000
DERIVED_SEED = 20261017


def _derived(x: Array, cov: Array, up: Array) -> tuple[Interval, Interval, Interval]:
    """Inter-HJC distance, height difference and roll with 95% intervals.

    Intervals are the 2.5/97.5 percentiles of a fixed-seed Gaussian posterior
    sample; ``sd`` is the sample standard deviation; ``estimate`` is the value
    at the MAP centres.
    """

    def quantities(v: Array) -> Array:
        d = v[..., :3] - v[..., 3:]
        z = d @ up
        horizontal = np.linalg.norm(d - z[..., None] * up, axis=-1)
        return np.stack(
            [np.linalg.norm(d, axis=-1), z, np.degrees(np.arctan2(z, horizontal))]
        )

    rng = np.random.default_rng(DERIVED_SEED)
    draws = quantities(rng.multivariate_normal(x, cov, size=DERIVED_DRAWS))
    point = quantities(x)
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=1)
    sd = draws.std(axis=1, ddof=1)
    out = tuple(
        Interval(float(point[k]), float(sd[k]), float(lo[k]), float(hi[k]))
        for k in range(3)
    )
    return out[0], out[1], out[2]


def _classify(
    spans: tuple[float, float], ratios: tuple[float, float], weight: float
) -> tuple[str, str]:
    for side, span, ratio in zip("RL", spans, ratios, strict=True):
        if span < MIN_ARC_SPAN_DEG:
            return (
                "degenerate",
                f"{side} knee arc spans {span:.1f} deg < {MIN_ARC_SPAN_DEG}",
            )
        if ratio < MIN_INFO_EIGEN_RATIO:
            return "degenerate", f"{side} information eigenvalue ratio {ratio:.2e}"
    if weight > PRIOR_DOMINATED_WEIGHT:
        return "prior_dominated", f"prior carries {weight:.2f} of posterior information"
    return "identified", f"prior carries {weight:.2f} of posterior information"


def integrated_autocorrelation_time(residuals: Array) -> float:
    """Frames per independent sample, ``1 + 2 * sum(rho_k)`` of a residual series.

    The sum stops at the first non-positive autocorrelation (initial positive
    sequence) and at a quarter of the series; the result is at least 1.
    Precondition: >= 8 finite residuals (``ValueError`` otherwise).
    """
    x = np.asarray(residuals, dtype=float)
    if x.ndim != 1 or x.size < MIN_FRAMES or not np.isfinite(x).all():
        raise ValueError("residuals must be >= 8 finite values in a 1-D series")
    x = x - x.mean()
    var = float(x @ x)
    if var <= 0.0:
        return 1.0
    tau = 1.0
    for lag in range(1, x.size // 4):
        rho = float(x[:-lag] @ x[lag:]) / var
        if rho <= 0.0:
            break
        tau += 2.0 * rho
    return max(tau, 1.0)


def _mean_residual_tau(tracks: tuple[Array, Array], theta: Array) -> float:
    """Mean integrated autocorrelation time of the two sides' sphere residuals."""
    taus = []
    for s in (0, 1):
        dist = np.linalg.norm(tracks[s] - theta[3 * s : 3 * s + 3], axis=1)
        taus.append(integrated_autocorrelation_time(dist - theta[6 + s]))
    return float(np.mean(taus))


def _invert(info: Array) -> Array:
    """Symmetric inverse of an information matrix with a tiny ridge."""
    cov = np.linalg.inv(info + 1e-14 * np.eye(info.shape[0]))
    return 0.5 * (cov + cov.T)


def _noise_sds(
    tracks: tuple[Array, Array],
    theta: Array,
    override: float | None,
    inflate: float,
) -> tuple[float, float]:
    """Per-side residual SD (m) of the data-only fit, floored and inflated."""
    out = []
    for s in (0, 1):
        dist = np.linalg.norm(tracks[s] - theta[3 * s : 3 * s + 3], axis=1)
        sd = float(np.sqrt(np.sum((dist - theta[6 + s]) ** 2) / (len(dist) - 4)))
        out.append(
            max(override if override is not None else sd, MIN_NOISE_SD_M) * inflate
        )
    return out[0], out[1]


def _data_information(
    tracks: tuple[Array, Array], theta: Array, sigma: tuple[float, float]
) -> Array:
    """Block-diagonal 8 x 8 data information J^T J / sigma^2 at ``theta``."""
    info = np.zeros((8, 8))
    for s in (0, 1):
        _, jac = _jacobian_rows(tracks[s], theta[3 * s : 3 * s + 3])
        cols = [3 * s, 3 * s + 1, 3 * s + 2, 6 + s]
        info[np.ix_(cols, cols)] = jac.T @ jac / sigma[s] ** 2
    return info


def _symmetry_test(theta_data: Array, cov_data: Array) -> SymmetryTest:
    """Chi-square (2 dof) test of equal AP and SI position of the data-only centres."""
    a = np.zeros((2, 6))
    a[0, 0], a[0, 3] = 1.0, -1.0  # x_R - x_L (AP)
    a[1, 2], a[1, 5] = 1.0, -1.0  # z_R - z_L (SI)
    diff = a @ theta_data[:6]
    chi2 = float(diff @ np.linalg.solve(a @ cov_data[:6, :6] @ a.T, diff))
    return SymmetryTest(chi2, math.exp(-0.5 * chi2))


def estimate_hip_centres(
    knee_r: Array,
    knee_l: Array,
    prior: HipCentrePrior,
    *,
    superior_axis: Sequence[float] = (0.0, 0.0, 1.0),
    noise_sd_m: float | None = None,
    frames_per_independent_sample: float | None = 1.0,
    use_prior: bool = True,
    waist_residual_rms_m: float = float("nan"),
    waist_residual_max_m: float = float("nan"),
) -> HipCentreEstimate:
    """MAP hip centres from pelvis-frame knee marker tracks.

    ``knee_r``/``knee_l`` are (frames, 3) knee marker positions in the pelvis
    frame. ``superior_axis`` is that frame's up direction (used for the height
    difference and roll). ``noise_sd_m`` overrides the per-side residual SD
    estimated from the sphere fit (floored at ``MIN_NOISE_SD_M``);
    ``frames_per_independent_sample`` (>= 1; ``None`` estimates it from the
    residual autocorrelation, see :func:`integrated_autocorrelation_time`) inflates the covariance for
    autocorrelated frames. ``use_prior=False`` gives the likelihood-only fit.
    ``waist_residual_*`` are passed through for reporting.

    Preconditions: finite tracks of >= ``MIN_FRAMES`` frames, finite nonzero
    ``superior_axis``, positive ``noise_sd_m`` and ``frames_per_independent_sample``.
    Raises ``ValueError`` otherwise. Postconditions: the result never forces
    left/right symmetry; ``status`` is one of ``STATUSES``; covariances are
    symmetric positive definite; a degenerate geometry returns a result (mostly
    prior) with ``status == "degenerate"`` instead of raising.
    """
    r_pts, l_pts = _as_track(knee_r, "knee_r"), _as_track(knee_l, "knee_l")
    up = np.asarray(superior_axis, dtype=float)
    if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) < 1e-9:
        raise ValueError("superior_axis must be a finite nonzero 3-vector")
    up = up / np.linalg.norm(up)
    if noise_sd_m is not None and not (math.isfinite(noise_sd_m) and noise_sd_m > 0):
        raise ValueError("noise_sd_m must be finite and positive")
    if frames_per_independent_sample is not None and not (
        math.isfinite(frames_per_independent_sample)
        and frames_per_independent_sample >= 1.0
    ):
        raise ValueError("frames_per_independent_sample must be finite and >= 1")

    tracks = (r_pts, l_pts)
    fits = [fit_sphere(t) for t in tracks]
    start = np.array(
        [*fits[0].centre, *fits[1].centre, fits[0].radius_m, fits[1].radius_m]
    )
    theta0, _ = _solve(tracks, start, (1.0, 1.0), None)  # data-only geometric LS
    tau = (
        _mean_residual_tau(tracks, theta0)
        if frames_per_independent_sample is None
        else float(frames_per_independent_sample)
    )
    sigma = _noise_sds(tracks, theta0, noise_sd_m, math.sqrt(tau))
    data_info = _data_information(tracks, theta0, sigma)

    prior_info: Array = np.asarray(np.linalg.inv(prior.covariance()), dtype=float)
    prior_mean: Array = np.asarray(
        np.concatenate([prior.mean_r, prior.mean_l]), dtype=float
    )
    if use_prior:
        # Start at the prior mean: weak data can send the free fit far away.
        radii = [
            float(np.linalg.norm(t - m, axis=1).mean())
            for t, m in zip(tracks, (prior.mean_r, prior.mean_l), strict=True)
        ]
        theta, info = _solve(
            tracks, np.array([*prior_mean, *radii]), sigma, (prior_mean, prior_info)
        )
    else:
        theta, info = _solve(tracks, theta0, sigma, None)
    cov8 = _invert(info)
    cov6 = cov8[:6, :6]
    weight = float(np.trace(cov6 @ prior_info) / 6.0) if use_prior else 0.0
    weight = min(max(weight, 0.0), 1.0)
    spans, ratios = [], []
    for s in (0, 1):
        centre = theta[3 * s : 3 * s + 3]
        spans.append(_arc_span_deg(tracks[s], centre))
        _, jac = _jacobian_rows(tracks[s], centre)
        eig = np.linalg.eigvalsh(jac.T @ jac)
        ratios.append(float(eig[0] / eig[-1]) if eig[-1] > 0 else 0.0)
    status, reason = _classify((spans[0], spans[1]), (ratios[0], ratios[1]), weight)
    inter, height, roll = _derived(theta[:6], cov6, up)
    sym = (
        _symmetry_test(theta0, _invert(data_info))
        if status != "degenerate"
        else SymmetryTest(float("nan"), float("nan"))
    )

    return HipCentreEstimate(
        centre_r=(float(theta[0]), float(theta[1]), float(theta[2])),
        centre_l=(float(theta[3]), float(theta[4]), float(theta[5])),
        cov_r=cov6[:3, :3].copy(),
        cov_l=cov6[3:, 3:].copy(),
        cov_joint=cov6.copy(),
        inter_hjc_m=inter,
        height_difference_m=height,
        roll_deg=roll,
        prior_weight=weight,
        status=status,
        status_reason=reason,
        symmetry=sym,
        radius_r_m=float(theta[6]),
        radius_l_m=float(theta[7]),
        noise_sd_r_m=sigma[0],
        noise_sd_l_m=sigma[1],
        arc_span_deg_r=spans[0],
        arc_span_deg_l=spans[1],
        frames_r=len(r_pts),
        frames_l=len(l_pts),
        prior_provenance=prior.provenance if use_prior else "none (likelihood only)",
        waist_residual_rms_m=waist_residual_rms_m,
        waist_residual_max_m=waist_residual_max_m,
        used_prior=use_prior,
        frames_per_independent_sample=tau,
    )


def estimate_hip_centres_from_capture(
    points: Array,
    valid: NDArray[np.bool_],
    labels: Sequence[str],
    waist_offsets: Mapping[str, Sequence[float]],
    prior: HipCentrePrior,
    *,
    knee_labels: tuple[str, str] = ("RKneeOut", "LKneeOut"),
    superior_axis: Sequence[float] = (0.0, 0.0, 1.0),
    noise_sd_m: float | None = None,
    frames_per_independent_sample: float | None = None,
    use_prior: bool = True,
) -> HipCentreEstimate:
    """:func:`estimate_hip_centres` for a marker capture with a waist marker set.

    The pelvis pose per frame is the rigid fit of ``waist_offsets`` (see
    ``hip_calibration.pelvis_frame_tracks``); the waist fit residual is
    reported as RMS and maximum absolute component on the result.
    """
    tracks = pelvis_frame_tracks(points, valid, labels, waist_offsets, knee_labels)
    return estimate_hip_centres(
        tracks.tracks[0],
        tracks.tracks[1],
        prior,
        superior_axis=superior_axis,
        noise_sd_m=noise_sd_m,
        frames_per_independent_sample=frames_per_independent_sample,
        use_prior=use_prior,
        waist_residual_rms_m=tracks.rms_m,
        waist_residual_max_m=tracks.max_abs_m,
    )
