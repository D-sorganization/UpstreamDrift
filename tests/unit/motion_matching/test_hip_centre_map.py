"""MAP hip-joint-centre estimator (HRG-C2, #13168): recovery, degeneracy, coverage."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.motion_matching import hip_centre_map as module
from src.shared.python.motion_matching.hip_centre_map import (
    HipCentrePrior,
    estimate_hip_centres,
)

pytestmark = pytest.mark.unit

STATURE_M = 1.71
KNEE_OUT_M = 0.06
THIGH_M = 0.40
TRUTH = {
    "r": np.array([-0.012, -0.095, -0.060]),
    "l": np.array([-0.012, 0.093, -0.058]),
}


def _femur_marker_tracks(
    centre: np.ndarray,
    side: str,
    *,
    frames: int,
    flexion_deg: float,
    abduction_deg: float,
    axial_deg: float,
    rng: np.random.Generator,
    noise_sd_m: float = 0.0,
) -> np.ndarray:
    """Femur-fixed lateral knee marker for a rigid femur rotating about ``centre``.

    The marker sits ``KNEE_OUT_M`` off the knee axis; the femur swings through a
    small cone with axial rotation, like a golf swing.
    """
    sign = -1.0 if side == "r" else 1.0
    local = np.array([0.0, sign * KNEE_OUT_M, -THIGH_M])
    phase = np.linspace(0.0, 2.0 * np.pi, frames, endpoint=False)
    out = np.empty((frames, 3))
    for i, ph in enumerate(phase):
        rot = Rotation.from_euler(
            "yxz",
            [
                flexion_deg * math.sin(ph),
                abduction_deg * math.sin(2.0 * ph + 0.7),
                axial_deg * math.sin(ph + 1.1),
            ],
            degrees=True,
        )
        out[i] = centre + rot.apply(local)
    return out + rng.normal(0.0, noise_sd_m, out.shape)


def _tracks(
    truth: dict[str, np.ndarray],
    rng: np.random.Generator,
    *,
    frames: int = 120,
    noise_sd_m: float = 0.0,
    flexion_deg: float = 25.0,
    abduction_deg: float = 12.0,
    axial_deg: float = 18.0,
) -> tuple[np.ndarray, np.ndarray]:
    kw = {
        "frames": frames,
        "flexion_deg": flexion_deg,
        "abduction_deg": abduction_deg,
        "axial_deg": axial_deg,
        "rng": rng,
        "noise_sd_m": noise_sd_m,
    }
    return (
        _femur_marker_tracks(truth["r"], "r", **kw),
        _femur_marker_tracks(truth["l"], "l", **kw),
    )


def test_noiseless_identifiable_synthetic_recovers_centres_within_1e6() -> None:
    rng = np.random.default_rng(1)
    r, lk = _tracks(TRUTH, rng, frames=200)
    est = estimate_hip_centres(r, lk, HipCentrePrior.from_stature(STATURE_M))
    assert est.status == "identified"
    np.testing.assert_allclose(est.centre_r, TRUTH["r"], atol=1e-6)
    np.testing.assert_allclose(est.centre_l, TRUTH["l"], atol=1e-6)


def test_rigid_femur_lateral_marker_does_not_bias_the_centre() -> None:
    """Astra/Argon dispute: a femur-fixed marker 60 mm off the knee axis is on a
    sphere about the true centre, so the centre is unbiased (no medialisation)."""
    rng = np.random.default_rng(2)
    biases = []
    for _ in range(40):
        r, lk = _tracks(TRUTH, rng, frames=150, noise_sd_m=0.002)
        est = estimate_hip_centres(
            r, lk, HipCentrePrior.from_stature(STATURE_M), use_prior=False
        )
        biases.append(
            [np.array(est.centre_r) - TRUTH["r"], np.array(est.centre_l) - TRUTH["l"]]
        )
    mean_bias = np.mean(biases, axis=0)
    assert np.abs(mean_bias).max() < 1.0e-3  # far below the 21 mm claimed


def test_short_arc_trajectory_is_flagged_degenerate() -> None:
    rng = np.random.default_rng(3)
    r, lk = _tracks(
        TRUTH, rng, noise_sd_m=0.001, flexion_deg=2.0, abduction_deg=1.0, axial_deg=1.0
    )
    est = estimate_hip_centres(r, lk, HipCentrePrior.from_stature(STATURE_M))
    assert est.status == "degenerate"
    assert est.arc_span_deg_r < module.MIN_ARC_SPAN_DEG


def test_weak_data_is_prior_dominated() -> None:
    rng = np.random.default_rng(4)
    r, lk = _tracks(
        TRUTH, rng, noise_sd_m=0.02, flexion_deg=9.0, abduction_deg=5.0, axial_deg=5.0
    )
    est = estimate_hip_centres(r, lk, HipCentrePrior.from_stature(STATURE_M))
    assert est.status in ("prior_dominated", "degenerate")
    assert est.prior_weight > 0.5


def test_symmetry_is_reported_not_forced() -> None:
    rng = np.random.default_rng(5)
    skew = {"r": TRUTH["r"] + [0.0, 0.0, 0.04], "l": TRUTH["l"]}
    r, lk = _tracks(skew, rng, noise_sd_m=0.001, frames=300)
    est = estimate_hip_centres(r, lk, HipCentrePrior.from_stature(STATURE_M))
    assert est.height_difference_m.estimate > 0.02  # right higher: not symmetrised
    assert est.symmetry.p_value < 0.05
    sym = {"r": TRUTH["r"], "l": TRUTH["l"] * [1, 1, 1] + [0, 0, -0.002]}
    r, lk = _tracks(sym, rng, noise_sd_m=0.001, frames=300)
    assert (
        estimate_hip_centres(
            r, lk, HipCentrePrior.from_stature(STATURE_M)
        ).symmetry.p_value
        > 0.001
    )


def test_interval_semantics_and_waist_residual_fields() -> None:
    rng = np.random.default_rng(6)
    r, lk = _tracks(TRUTH, rng, noise_sd_m=0.002)
    est = estimate_hip_centres(
        r,
        lk,
        HipCentrePrior.from_stature(STATURE_M),
        waist_residual_rms_m=0.01,
        waist_residual_max_m=0.0278,
    )
    d = est.inter_hjc_m
    assert d.lo < d.estimate < d.hi
    assert d.estimate == pytest.approx(
        np.linalg.norm(TRUTH["r"] - TRUTH["l"]), abs=0.01
    )
    assert est.cov_r.shape == (3, 3) and np.all(np.linalg.eigvalsh(est.cov_r) > 0)
    assert est.waist_residual_rms_m == 0.01 and est.waist_residual_max_m == 0.0278
    assert est.prior_provenance


def test_result_is_frozen() -> None:
    rng = np.random.default_rng(7)
    r, lk = _tracks(TRUTH, rng, noise_sd_m=0.002)
    est = estimate_hip_centres(r, lk, HipCentrePrior.from_stature(STATURE_M))
    with pytest.raises(AttributeError):
        est.status = "x"  # type: ignore[misc]


def test_prior_mean_is_the_stature_prior() -> None:
    prior = HipCentrePrior.from_stature(STATURE_M)
    r, lk = prior.mean_r, prior.mean_l
    assert lk[1] - r[1] == pytest.approx(0.104 * STATURE_M)
    assert prior.provenance
    assert prior.covariance().shape == (6, 6)


@pytest.mark.parametrize("bad", ["nan", "few", "shape"])
def test_dbc_rejections(bad: str) -> None:
    rng = np.random.default_rng(8)
    r, lk = _tracks(TRUTH, rng, noise_sd_m=0.001)
    prior = HipCentrePrior.from_stature(STATURE_M)
    if bad == "nan":
        r = r.copy()
        r[3, 1] = np.nan
    elif bad == "few":
        r = r[:3]
    else:
        lk = lk[:, :2]
    with pytest.raises(ValueError):
        estimate_hip_centres(r, lk, prior)


@pytest.mark.parametrize("stature", [0.0, -1.7, float("nan"), float("inf")])
def test_prior_rejects_non_positive_stature(stature: float) -> None:
    with pytest.raises(ValueError, match="stature"):
        HipCentrePrior.from_stature(stature)


def test_rejects_bad_superior_axis_and_noise() -> None:
    rng = np.random.default_rng(9)
    r, lk = _tracks(TRUTH, rng, noise_sd_m=0.001)
    prior = HipCentrePrior.from_stature(STATURE_M)
    with pytest.raises(ValueError, match="superior"):
        estimate_hip_centres(r, lk, prior, superior_axis=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="noise"):
        estimate_hip_centres(r, lk, prior, noise_sd_m=-1.0)


def _coverage(trials: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    prior = HipCentrePrior.from_stature(STATURE_M)
    cov = prior.covariance()
    mean = np.concatenate([prior.mean_r, prior.mean_l])
    hits = {"inter": 0, "height": 0, "roll": 0}
    for _ in range(trials):
        x = rng.multivariate_normal(mean, cov)  # truth drawn from the declared prior
        truth = {"r": x[:3], "l": x[3:]}
        r, lk = _tracks(
            truth,
            rng,
            frames=60,
            noise_sd_m=0.004,
            flexion_deg=14.0,
            abduction_deg=7.0,
            axial_deg=9.0,
        )
        est = estimate_hip_centres(r, lk, prior)
        d = truth["r"] - truth["l"]
        up_h = d[2]
        hits["inter"] += est.inter_hjc_m.lo <= np.linalg.norm(d) <= est.inter_hjc_m.hi
        hits["height"] += (
            est.height_difference_m.lo <= up_h <= est.height_difference_m.hi
        )
        roll = math.degrees(math.atan2(up_h, math.hypot(d[0], d[1])))
        hits["roll"] += est.roll_deg.lo <= roll <= est.roll_deg.hi
    return {k: v / trials for k, v in hits.items()}


def test_interval_coverage_small_fast() -> None:
    cov = _coverage(40, seed=11)
    assert 0.80 <= cov["inter"] <= 1.0
    assert 0.80 <= cov["height"] <= 1.0
    assert 0.80 <= cov["roll"] <= 1.0


@pytest.mark.slow
def test_interval_coverage_over_200_seeded_trials() -> None:
    cov = _coverage(300, seed=12)
    for key in ("inter", "height", "roll"):
        assert 0.90 <= cov[key] <= 0.99, (key, cov)
