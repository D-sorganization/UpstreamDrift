"""Multi-frame marker-offset bundle adjustment on a synthetic chain (#15296)."""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import pytest

from src.shared.python.motion_matching.pipeline import bundle_adjustment as ba
from src.shared.python.motion_matching.pipeline.bundle_adjustment import (
    BundleProblem,
    BundleSettings,
    solve_bundle,
)

pytestmark = [pytest.mark.unit]
LINK_M = 0.4
NOISE_M = 0.001
PRIOR_SIGMA_M = 0.02
WELD_SIGMA_M = 0.0005
#: Scale probe: hinges, frames and markers of the driver fit's order of size.
PROBE = (30, 35, 30)
#: One fit of the capture takes about twenty minutes; the adjustment may add a
#: tenth of that. Chosen from that share, not from a measured run of this test.
PROBE_BUDGET_S = 120.0
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
    """Predicted marker positions of ``bodies`` for the given poses and offsets."""
    frames = np.array([chain(pose)[bodies] for pose in poses])
    return np.einsum("kmij,mj->kmi", frames[..., :3, :3], offsets) + frames[..., :3, 3]


def chain_twists(pose: np.ndarray) -> np.ndarray:
    """Origin velocity (rows 0-2) and angular velocity (3-5) of both links."""
    out = np.zeros((2, 6, 2))
    out[0, 5, 0] = out[1, 5, 0] = out[1, 5, 1] = 1.0
    out[1, :3, 0] = np.cross([0.0, 0.0, 1.0], chain(pose)[1, :3, 3])
    return out


def tip_height(frame: int, pose: np.ndarray) -> np.ndarray:
    """A closed loop: the tip of link two (as long as link one) slides on y = 0."""
    return np.array([np.sin(pose[0]) + np.sin(pose.sum())]) * LINK_M / WELD_SIGMA_M


def problem(n_frames=8, keep=None, shift=None, poses=None, **changes):
    """A problem whose true offsets are the seeds plus ``shift`` (default: random)."""
    rng = np.random.default_rng(15296)
    keep = list(range(len(BODIES))) if keep is None else keep
    bodies, seeds = BODIES[keep], SEEDS[keep]
    truth = seeds + (rng.normal(0.0, 0.01, seeds.shape) if shift is None else shift)
    poses = rng.uniform(-1.2, 1.2, (n_frames, 2)) if poses is None else poses
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
    """A single frame cannot separate offset from pose and is refused."""
    prob, _, _ = problem(n_frames=1)
    with pytest.raises(ValueError, match="at least 2 frames"):
        solve_bundle(prob, BundleSettings())
    with pytest.raises(ValueError, match="marker_sigma_m"):
        BundleSettings(marker_sigma_m=0.0)
    with pytest.raises(TypeError, match="BundleSettings"):
        solve_bundle(prob, {"marker_sigma_m": 0.01})  # type: ignore[arg-type]


def test_prior_holds_an_unobserved_marker_and_reports_it() -> None:
    """A zero-sigma prior holds an unobserved marker at its seed."""
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
    """Offsets parked on a bound are reported in on_bound."""
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
        ({"offset_sigmas": np.full((6, 3), -1.0)}, ValueError, "offset_sigmas"),
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
    """Ill-formed problems and settings fail closed at check."""
    prob, _, _ = problem(**changes)
    with pytest.raises(error, match=message):
        solve_bundle(prob, BundleSettings())


def test_caller_pose_jacobian_gives_the_same_fit() -> None:
    """A supplied body_jacobian matches differenced-pose results."""
    prob, _, _ = problem()
    settings = BundleSettings(marker_sigma_m=NOISE_M)
    exact = solve_bundle(
        dataclasses.replace(prob, body_jacobian=chain_twists), settings
    )
    differenced = solve_bundle(prob, settings)
    assert exact.converged and differenced.converged
    assert np.abs(exact.offsets - differenced.offsets).max() < 1e-5


def test_constraint_keeps_the_loop_closed_and_fixes_the_gauge() -> None:
    """With the tip held on a line the second joint follows from the first, so
    the lone marker on link two is no longer free to turn about its hinge."""
    first = np.linspace(-1.0, 1.0, 8)
    poses = np.column_stack([first, -2.0 * first])  # sin(a) + sin(-a) = 0
    prob, truth, _ = problem(keep=SPARSE, poses=poses, constraint=tip_height)
    result = solve_bundle(prob, BundleSettings(marker_sigma_m=NOISE_M))
    tip = np.array([tip_height(0, pose)[0] * WELD_SIGMA_M for pose in result.poses])
    assert np.abs(tip).max() < 2.0 * WELD_SIGMA_M
    assert result.converged and result.weakly_identified == ()
    assert np.abs(result.offsets[3] - truth[3]).max() < 0.003
    assert 0.0 < result.constraint_chi2 < result.data_chi2
    with pytest.raises(TypeError, match="constraint"):
        solve_bundle(dataclasses.replace(prob, constraint=1.0), BundleSettings())


def test_zero_sigma_holds_a_marker_and_equal_bounds_hold_a_coordinate() -> None:
    """Zero sigma pins a marker; equal pose bounds pin a coordinate."""
    prob, _, _ = problem()
    sigmas = prob.offset_sigmas.copy()
    sigmas[4] = 0.0
    start = prob.pose_start.copy()
    start[:, 1] = 0.3
    bounds = (np.array([-np.pi, 0.3]), np.array([np.pi, 0.3]))
    held = dataclasses.replace(
        prob, offset_sigmas=sigmas, pose_start=start, pose_bounds=bounds
    )
    result = solve_bundle(held, BundleSettings(marker_sigma_m=NOISE_M))
    assert np.array_equal(result.offsets[4], SEEDS[4])
    assert np.all(result.poses[:, 1] == 0.3)
    assert 4 not in result.weakly_identified
    assert all(marker != 4 for marker, _ in result.on_bound)
    assert np.isfinite(result.offset_prior_chi2)


@pytest.mark.slow
def test_converges_at_the_size_of_the_driver_fit_within_budget() -> None:
    """The driver-fit-sized probe converges inside the time budget."""
    hinges, n_frames, markers = PROBE
    rng = np.random.default_rng(0)
    axes = rng.integers(0, 3, hinges)

    def serial(pose: np.ndarray) -> np.ndarray:
        """Serial hinge-chain fk for the probe's joints."""
        frames, world = np.empty((hinges, 4, 4)), np.eye(4)
        for i, (axis, angle) in enumerate(zip(axes, pose, strict=True)):
            a, b = (axis + 1) % 3, (axis + 2) % 3
            joint = np.eye(4)
            joint[[a, b], [a, b]] = np.cos(angle)
            joint[a, b], joint[b, a] = -np.sin(angle), np.sin(angle)
            joint[0, 3] = 0.1 * (i > 0)
            world = frames[i] = world @ joint
        return frames

    bodies = np.arange(markers) % hinges
    seeds = rng.uniform(-0.05, 0.05, (markers, 3))
    truth = seeds + rng.normal(0.0, 0.01, seeds.shape)
    poses = rng.uniform(-0.5, 0.5, (n_frames, hinges))
    frames = np.array([serial(pose)[bodies] for pose in poses])
    observed = (
        np.einsum("kmij,mj->kmi", frames[..., :3, :3], truth) + frames[..., :3, 3]
    )
    prob = BundleProblem(
        fk=serial,
        marker_bodies=bodies,
        observations=observed + rng.normal(0.0, 0.002, observed.shape),
        valid=np.ones(observed.shape[:2], bool),
        offset_seeds=seeds,
        offset_sigmas=np.full(seeds.shape, PRIOR_SIGMA_M),
        pose_start=poses + rng.normal(0.0, 0.03, poses.shape),
        pose_bounds=(np.full(hinges, -2.0), np.full(hinges, 2.0)),
    )
    started = time.process_time()  # processor time: the host's load is not ours
    result = solve_bundle(prob, BundleSettings(marker_sigma_m=0.002))
    assert time.process_time() - started < PROBE_BUDGET_S
    assert result.converged
    assert result.rms_m < 0.002 * np.sqrt(3.0)
    assert np.abs(result.offsets - truth).max() < PRIOR_SIGMA_M


def test_offset_reach_caps_the_bound_and_progress_is_logged(caplog) -> None:
    """offset_reach caps the bound and progress reaches the log."""
    shift = np.zeros_like(SEEDS)
    shift[0, 0] = 0.030
    prob, _, _ = problem(shift=shift, offset_reach=np.full(SEEDS.shape, 0.004))
    with caplog.at_level("INFO"):
        result = solve_bundle(prob, BundleSettings(marker_sigma_m=NOISE_M))
    assert np.abs(result.offsets - SEEDS).max() <= 0.004 + 1e-12
    assert (0, 0) in result.on_bound
    assert "bundle iteration" in caplog.text
    with pytest.raises(ValueError, match="offset_reach"):
        problem(offset_reach=np.full(SEEDS.shape, -1.0))[0].check(BundleSettings())


def test_a_marker_weight_scales_its_residual_and_shields_the_fit() -> None:
    """Marker weights scale residuals and shield the fit."""
    prob, truth, poses = problem()
    wild = np.array(prob.observations)
    wild[:, 0] += np.random.default_rng(1).normal(0.0, 0.02, wild[:, 0].shape)
    trusted = dataclasses.replace(prob, observations=wild)
    weights = np.array([1e-6, 1.0, 1.0, 1.0, 1.0, 1.0])
    shielded = dataclasses.replace(trusted, marker_weights=weights)
    settings = BundleSettings(marker_sigma_m=NOISE_M)
    plain, fit = solve_bundle(trusted, settings), solve_bundle(shielded, settings)
    assert np.abs(fit.offsets[1:3] - truth[1:3]).max() < 0.002
    assert np.abs(fit.poses[:, 0] - poses[:, 0]).max() < 0.01
    assert fit.data_chi2 < 0.1 * plain.data_chi2
    assert fit.per_marker_rms_m[0] > 0.01  # the reported error stays unweighted
    with pytest.raises(ValueError, match="marker_weights"):
        dataclasses.replace(prob, marker_weights=-weights).check(settings)


class TestFc011BundleGuards15345:
    """FC-011 edge cases for bundle_adjustment (#15345)."""

    def test_no_weak_directions_returns_empty_without_error(self) -> None:
        """No weak direction means empty output, not an error."""
        info = np.diag([2.0, 3.0, 4.0])
        owners = np.array([0, 0, 0])
        settings = BundleSettings(weak_information_ratio=1.0)
        weak, ratios = ba._weak_directions(info, owners, settings)
        assert weak == () and ratios == ()

    def test_check_requires_at_least_one_valid_observation(self) -> None:
        """check rejects a problem with no valid observations."""
        prob, _, _ = problem()
        invalid = dataclasses.replace(
            prob, valid=np.zeros(prob.valid.shape, dtype=bool)
        )
        with pytest.raises(ValueError, match="at least one valid observation"):
            invalid.check(BundleSettings())

    def test_check_requires_a_free_offset_component(self) -> None:
        """check rejects a problem with no free offset component."""
        prob, _, _ = problem()
        held = dataclasses.replace(
            prob, offset_sigmas=np.zeros_like(prob.offset_sigmas)
        )
        with pytest.raises(ValueError, match="free offset"):
            held.check(BundleSettings())

    def test_check_requires_active_pose_prior_when_provided(self) -> None:
        """An active pose prior needs a finite positive sigma somewhere."""
        prob, _, poses = problem()
        inactive = dataclasses.replace(
            prob,
            pose_prior=(poses, np.full(poses.shape, np.inf)),
        )
        with pytest.raises(ValueError, match="active pose_prior"):
            inactive.check(BundleSettings())

    def test_solve_refuses_problem_with_no_valid_observations(self) -> None:
        """solve_bundle refuses an all-invalid problem."""
        prob, _, _ = problem()
        invalid = dataclasses.replace(
            prob, valid=np.zeros(prob.valid.shape, dtype=bool)
        )
        with pytest.raises(ValueError, match="at least one valid observation"):
            solve_bundle(invalid, BundleSettings())

    def test_inward_fd_steps_toward_the_box_interior(self) -> None:
        """Differencing steps inward at the box edge."""
        lower, upper = np.array([0.0, -1.0]), np.array([1.0, 0.0])
        pose = upper.copy()
        touched: list[np.ndarray] = []

        def sampler(q: np.ndarray) -> float:
            """Pose-sum probe that records every point sampled."""
            touched.append(q.copy())
            return float(q.sum())

        step = 1e-4
        _, slope = ba._difference(sampler, pose, step, (lower, upper))
        assert slope.shape == (1, 2)
        assert touched[1][0] == pytest.approx(upper[0] - step)
        assert touched[2][1] == pytest.approx(upper[1] - step)

    def test_result_is_never_converged_with_non_finite_rms(self) -> None:
        """A non-finite rms never counts as converged."""
        prob, _, _ = problem()
        prob.check(BundleSettings())
        result = solve_bundle(prob, BundleSettings(marker_sigma_m=NOISE_M))
        assert np.isfinite(result.rms_m)
        if result.converged:
            assert result.rms_m < np.inf


def translating_problem(n_q: int = 1) -> BundleProblem:
    """Two frames with one translating body and otherwise inert coordinates."""

    def fk(pose: np.ndarray) -> np.ndarray:
        """Body frame of the single translating body at this pose."""
        frames = np.eye(4)[None].copy()
        frames[0, 0, 3] = pose[0]
        return frames

    return BundleProblem(
        fk=fk,
        marker_bodies=np.array([0]),
        observations=np.zeros((2, 1, 3)),
        valid=np.ones((2, 1), bool),
        offset_seeds=np.zeros((1, 3)),
        offset_sigmas=np.full((1, 3), 0.01),
        pose_start=np.zeros((2, n_q)),
        pose_bounds=(np.zeros(n_q), np.ones(n_q)),
    )


@pytest.mark.parametrize("fixed", [False, True])
def test_all_bound_poses_solve_and_report_offset_information(fixed: bool) -> None:
    """Fully bound poses solve; offset information reflects the pinning."""
    prob = translating_problem()
    if fixed:
        prob = dataclasses.replace(prob, pose_bounds=(np.zeros(1), np.zeros(1)))
    result = solve_bundle(prob, BundleSettings())
    assert result.rms_m == pytest.approx(0.0)
    assert np.array_equal(result.poses, prob.pose_start)
    assert np.array_equal(result.offsets, prob.offset_seeds)
    assert result.weakly_identified == (() if fixed else (0,))
    assert len(result.weak_information_ratios) == (0 if fixed else 1)
    if not fixed:
        assert result.weak_information_ratios[0] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("position", [0.0, 1e-5, 1.0])
def test_inward_pose_tangent_removes_translation_information(position: float) -> None:
    """An inward pose tangent drops translation information."""
    prob = translating_problem(n_q=2)
    poses = np.tile([position, 0.5], (2, 1))
    information = ba._offset_information(
        prob, BundleSettings(), poses, prob.offset_seeds
    )
    np.testing.assert_allclose(information, np.diag([0.0, 2.0, 2.0]), atol=1e-12)


@pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0])
def test_fd_probes_respect_fixed_and_narrow_boxes(fraction: float) -> None:
    """Forward-difference probes stay inside the pose box."""
    lower = np.zeros(2)
    upper = np.array([0.0, 1e-8])
    pose = upper * fraction
    prob = dataclasses.replace(translating_problem(n_q=2), pose_bounds=(lower, upper))
    calls = {"fk": 0, "constraint": 0}

    def checked_fk(q: np.ndarray) -> np.ndarray:
        """fk asserting the probed pose lies within the box."""
        assert np.all((lower <= q) & (q <= upper))
        calls["fk"] += 1
        frames = np.eye(4)[None].copy()
        frames[0, 0, 3] = q.sum()
        return frames

    def checked_constraint(k: int, q: np.ndarray) -> np.ndarray:
        """Constraint asserting the probed pose lies within the box."""
        assert np.all((lower <= q) & (q <= upper))
        calls["constraint"] += 1
        return np.array([q.sum()])

    prob = dataclasses.replace(prob, fk=checked_fk, constraint=checked_constraint)
    _, by_pose, _ = ba._linearise(prob, BundleSettings(), 0, (pose, prob.offset_seeds))
    np.testing.assert_allclose(by_pose[:, 0], 0.0)
    np.testing.assert_allclose(by_pose[:, 1], [100.0, 0.0, 0.0, 1.0])
    assert calls["fk"] > 1 and calls["constraint"] > 1


@pytest.mark.parametrize("sigma", [-np.inf, np.inf])
def test_mixed_pose_prior_requires_positive_sigmas(sigma: float) -> None:
    """Mixed pose-prior sigmas must stay positive, inf allowed."""
    prob = translating_problem(n_q=2)
    sigmas = np.tile([1.0, sigma], (2, 1))
    prob = dataclasses.replace(prob, pose_prior=(prob.pose_start, sigmas))
    if sigma < 0:
        with pytest.raises(ValueError, match="pose_prior sigmas must be positive"):
            prob.check(BundleSettings())
    else:
        prob.check(BundleSettings())


@pytest.mark.unit
class TestInputGuards15862:
    """FC-011 follow-up (#15862): fail closed on NaN prior means and bad body indices."""

    def test_nan_pose_prior_mean_is_refused(self) -> None:
        """A NaN prior mean must raise, not return the seeds as converged."""
        prob = translating_problem(n_q=2)
        mean = prob.pose_start.copy()
        mean[1, 0] = np.nan
        prob = dataclasses.replace(prob, pose_prior=(mean, np.full(mean.shape, 1.0)))
        with pytest.raises(ValueError, match="pose_prior mean"):
            solve_bundle(prob, BundleSettings())

    @pytest.mark.parametrize(
        "bodies",
        [
            pytest.param(np.array([0, -1]), id="negative-wraps"),
            pytest.param(np.array([0, 1]), id="beyond-body-count"),
            pytest.param(np.array([0.0, 0.5]), id="non-integer"),
        ],
    )
    def test_invalid_marker_bodies_are_refused(self, bodies: np.ndarray) -> None:
        """Indices must be integers naming real bodies; ``-1`` silently wrapped."""
        observations = np.zeros((2, len(bodies), 3))
        prob = dataclasses.replace(
            translating_problem(n_q=1),
            marker_bodies=bodies,
            observations=observations,
            valid=np.ones((2, len(bodies)), bool),
            offset_seeds=np.zeros((len(bodies), 3)),
            offset_sigmas=np.full((len(bodies), 3), 0.01),
        )
        match = (
            "marker_bodies must be an integer"
            if bodies.dtype.kind == "f"
            else "marker_bodies must index"
        )
        with pytest.raises(ValueError, match=match):
            solve_bundle(prob, BundleSettings())

    def test_converged_result_has_finite_pose_prior_chi2(self) -> None:
        """The postcondition the guards exist to protect: no NaN chi2 on convergence."""
        result = solve_bundle(translating_problem(), BundleSettings())
        assert not result.converged or np.isfinite(result.pose_prior_chi2)
