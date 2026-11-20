"""Multi-frame marker-offset bundle adjustment on a synthetic chain (#15296)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.shared.python.motion_matching.pipeline.bundle_adjustment import (
    BundleProblem,
    BundleSettings,
    solve_bundle,
)

pytestmark = [pytest.mark.unit]
LINK_M = 0.4
NOISE_M = 0.001
PRIOR_SIGMA_M = 0.02
#: Three markers per link; ``SPARSE`` leaves a single marker on the second.
BODIES = np.array([0, 0, 0, 1, 1, 1])
SEEDS = np.random.default_rng(7).uniform(0.05, 0.3, (6, 3)) * [1.0, 0.3, 0.2]
SPARSE = [0, 1, 2, 3]


def chain(pose: np.ndarray) -> np.ndarray:
    """Two hinges about z: link one at the origin, link two at its tip."""
    frames = np.tile(np.eye(4), (2, 1, 1))
    for body, angle in enumerate(np.cumsum(pose)):
        c, s = np.cos(angle), np.sin(angle)
        frames[body, :2, :2] = [[c, -s], [s, c]]
    frames[1, :3, 3] = frames[0, :3, :3] @ [LINK_M, 0.0, 0.0]
    return frames


def observe(poses: np.ndarray, offsets: np.ndarray, bodies: np.ndarray) -> np.ndarray:
    frames = np.array([chain(pose)[bodies] for pose in poses])
    return np.einsum("kmij,mj->kmi", frames[..., :3, :3], offsets) + frames[..., :3, 3]


def problem(n_frames: int = 8, keep: list[int] | None = None, shift=None, **changes):
    """A problem whose true offsets are the seeds plus ``shift`` (default: random)."""
    rng = np.random.default_rng(15296)
    keep = list(range(len(BODIES))) if keep is None else keep
    bodies, seeds = BODIES[keep], SEEDS[keep]
    truth = seeds + (rng.normal(0.0, 0.01, seeds.shape) if shift is None else shift)
    poses = rng.uniform(-1.2, 1.2, (n_frames, 2))
    observed = observe(poses, truth, bodies) + rng.normal(
        0.0, NOISE_M, (n_frames, len(keep), 3)
    )
    fields = {
        "fk": chain,
        "marker_bodies": bodies,
        "observations": observed,
        "valid": np.ones(observed.shape[:2], bool),
        "offset_seeds": seeds,
        "offset_sigmas": np.full(seeds.shape, PRIOR_SIGMA_M),
        "pose_start": poses + rng.normal(0.0, 0.05, poses.shape),
        "pose_bounds": (np.full(2, -np.pi), np.full(2, np.pi)),
    }
    return BundleProblem(**{**fields, **changes}), truth, poses


def test_recovers_known_offsets_and_poses_from_noisy_frames() -> None:
    """The leaf link's cluster can turn about its own hinge with the joint
    angle, so that one direction is reported weak; everything else is found."""
    prob, truth, poses = problem()
    result = solve_bundle(prob, BundleSettings(marker_sigma_m=NOISE_M))
    assert np.abs(result.offsets[:3] - truth[:3]).max() < 0.002
    radius = np.linalg.norm(result.offsets[3:, :2], axis=1)
    assert np.abs(radius - np.linalg.norm(truth[3:, :2], axis=1)).max() < 0.002
    assert np.abs(result.offsets[3:, 2] - truth[3:, 2]).max() < 0.002
    assert np.abs(result.poses[:, 0] - poses[:, 0]).max() < 0.01
    assert result.rms_m < 2.0 * NOISE_M * np.sqrt(3.0)
    assert (len(result.per_marker_rms_m), len(result.per_frame_rms_m)) == (6, 8)
    assert len(result.weak_information_ratios) == 1 and result.on_bound == ()
    assert set(result.weakly_identified) <= {3, 4, 5} and result.converged
    assert result.data_chi2 > result.offset_prior_chi2 > 0.0
    assert 0.0 < result.prior_share < 0.5


def test_one_frame_is_refused() -> None:
    prob, _, _ = problem(n_frames=1)
    with pytest.raises(ValueError, match="at least 2 frames"):
        solve_bundle(prob, BundleSettings())
    with pytest.raises(ValueError, match="marker_sigma_m"):
        BundleSettings(marker_sigma_m=0.0)
    with pytest.raises(TypeError, match="BundleSettings"):
        solve_bundle(prob, {"marker_sigma_m": 0.01})  # type: ignore[arg-type]


def test_prior_holds_an_unobserved_marker_and_reports_it() -> None:
    prob, _, _ = problem()
    valid = prob.valid.copy()
    valid[:, 1] = False
    observed = prob.observations.copy()
    observed[:, 1] = np.nan  # ignored where not valid
    prob = dataclasses.replace(prob, valid=valid, observations=observed)
    result = solve_bundle(prob, BundleSettings(marker_sigma_m=NOISE_M))
    assert np.allclose(result.offsets[1], SEEDS[1], atol=1e-9)
    assert 1 in result.weakly_identified
    assert not {0, 2} & set(result.weakly_identified)
    # its three axes plus the leaf link gauge
    assert sum(ratio < 1e-9 for ratio in result.weak_information_ratios) == 4
    assert np.isnan(result.per_marker_rms_m[1])


def test_offset_pinned_at_a_bound_is_reported() -> None:
    shift = np.zeros((6, 3))
    shift[1, 0] = 5.0 * PRIOR_SIGMA_M  # truth lies beyond the 3-sigma bound
    prob, _, _ = problem(shift=shift)
    result = solve_bundle(prob, BundleSettings(marker_sigma_m=NOISE_M))
    assert (1, 0) in result.on_bound
    assert result.offsets[1, 0] == pytest.approx(SEEDS[1, 0] + 3.0 * PRIOR_SIGMA_M)


def test_gauge_direction_is_reported_weak_and_left_at_the_prior() -> None:
    """One marker on a hinged link: its offset can rotate about the hinge with
    the joint angle, so the tangential direction is a gauge the data never fix."""
    prob, truth, _ = problem(keep=SPARSE)
    result = solve_bundle(prob, BundleSettings(marker_sigma_m=NOISE_M))
    assert result.weakly_identified == (3,)
    assert result.weak_information_ratios[0] < 1e-6
    radius = np.linalg.norm(result.offsets[3, :2])
    assert radius == pytest.approx(np.linalg.norm(truth[3, :2]), abs=0.002)
    tangent = np.cross([0.0, 0.0, 1.0], SEEDS[3]) / np.linalg.norm(SEEDS[3, :2])
    assert abs((result.offsets[3] - SEEDS[3]) @ tangent) < 0.002
    # a pose prior fixes the joint, but the data-only report still names the gauge
    _, _, poses = problem(keep=SPARSE)
    sigmas = np.stack([np.full(len(poses), np.inf), np.full(len(poses), 1e-4)], 1)
    prob = dataclasses.replace(prob, pose_prior=(poses, sigmas))
    held = solve_bundle(prob, BundleSettings(marker_sigma_m=NOISE_M))
    assert np.abs(held.poses[:, 1] - poses[:, 1]).max() < 1e-3
    assert held.pose_prior_chi2 > 0.0 and held.weakly_identified == (3,)


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"offset_sigmas": np.zeros((6, 3))}, ValueError, "offset_sigmas"),
        ({"offset_seeds": np.full((6, 3), np.nan)}, ValueError, "offset_seeds"),
        ({"marker_bodies": np.zeros(5, int)}, ValueError, "marker_bodies"),
        ({"valid": np.ones((8, 5), bool)}, ValueError, "valid"),
        ({"pose_start": np.zeros((7, 2))}, ValueError, "pose_start"),
        ({"pose_bounds": (np.ones(2), np.zeros(2))}, ValueError, "pose_bounds"),
        ({"fk": None}, TypeError, "fk"),
        ({"observations": np.full((8, 6, 3), np.inf)}, ValueError, "observations"),
    ],
)
def test_contract_violations_raise(changes, error, message) -> None:
    prob, _, _ = problem(**changes)
    with pytest.raises(error, match=message):
        solve_bundle(prob, BundleSettings())
