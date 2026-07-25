"""Benchmark comparing trajectory-funnel RL policies across solver configurations."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_CONVERGENCE_EPSILON = 1.0e-12
_PHASE_REWARD_SCALE = 0.5
_TRANSVERSE_COST_SCALE = 10.0
_PLANT_DT = 0.25
_ACTION_LIMIT = 5.0


class TrajectoryFunnelBenchmark:
    """Compare a trajectory-funnel reward against a classical setpoint reward.

    The comparison is run by actually training a policy: :meth:`train_agent`
    optimises a linear feedback policy with Augmented Random Search (ARS) on a
    controlled first-order plant, using this instance's reward as the training
    signal. The two modes therefore produce *different* state sequences, which
    is the minimum requirement for the comparison to mean anything.

    Fair comparison (issue #7983): the two reward functions are not on a common
    scale, so their returns and return variances are not comparable. Use the
    mode-neutral metrics that :meth:`train_agent` also reports -
    ``mean_transverse_error`` and ``terminal_setpoint_error`` - when ranking the
    two objectives against each other.

    Attributes:
        mode: ``"transverse"`` (funnel reward) or ``"setpoint"``.
        phase_window: Fraction of the reference length the funnel projection is
            allowed to search around the commanded phase. ``None`` (the
            default) means the projection is global, i.e. unrestricted phase
            slippage, and ``current_phase`` is unused.
        learning_curve: Per-iteration evaluation return from the most recent
            :meth:`train_agent` call.
    """

    def __init__(
        self, mode: str = "transverse", phase_window: float | None = None
    ) -> None:
        assert mode in [
            "transverse",
            "setpoint",
        ], "Mode must be 'transverse' or 'setpoint'"
        assert phase_window is None or 0.0 < phase_window <= 1.0, (
            "phase_window must be None or in (0, 1]"
        )
        self.mode = mode
        self.phase_window = phase_window
        self.learning_curve: list[float] = []

    def setpoint_reward(
        self, current_state: np.ndarray, target_state: np.ndarray
    ) -> float:
        """
        Classical control approach: Drive Euclidean distance to the destination to zero.
        Ignores path geometry, heavily penalizes phase asynchrony.
        """
        assert current_state is not None, "current_state must be provided"
        error = np.asarray(current_state, dtype=np.float64) - np.asarray(
            target_state, dtype=np.float64
        )
        # ⚡ Bolt: np.vdot is ~3x faster than np.sum(error**2) and avoids temporary array allocations
        return float(-np.vdot(error, error))

    def trajectory_funnel_reward(
        self,
        current_state: np.ndarray,
        reference_trajectory: np.ndarray,
        current_phase: float,
    ) -> float:
        """Reward confinement to the trajectory tube (orbital stability).

        Args:
            current_state: State to score.
            reference_trajectory: Reference manifold, shape ``(T, d)``.
            current_phase: Commanded phase in ``[0, 1]``. This is used **only**
                when :attr:`phase_window` is set, in which case the projection
                is restricted to reference indices within that fraction of the
                commanded phase (a local funnel). With ``phase_window=None``
                the projection is global and this argument is ignored, which is
                the "unrestricted phase slippage" behaviour (issue #7983).

        Returns:
            Scalar reward.
        """
        # Find the geometrically closest point on the reference trajectory manifold
        assert current_state is not None, "current_state must be provided"
        rewards = self._trajectory_funnel_rewards(
            np.asarray(current_state, dtype=np.float64)[np.newaxis, :],
            reference_trajectory,
            phases=np.array([current_phase], dtype=np.float64),
        )
        return float(rewards[0])

    def _trajectory_funnel_rewards(
        self,
        current_states: np.ndarray,
        reference_trajectory: np.ndarray,
        phases: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return trajectory-funnel rewards for a batch of states.

        Args:
            current_states: States to score, shape ``(N, d)``.
            reference_trajectory: Reference manifold, shape ``(T, d)``.
            phases: Commanded phase per state in ``[0, 1]``. Only consulted
                when :attr:`phase_window` is set.

        Returns:
            Rewards, shape ``(N,)``.
        """
        states = np.asarray(current_states, dtype=np.float64)
        reference = np.asarray(reference_trajectory, dtype=np.float64)
        assert states.ndim == 2, "current_states must be a 2D array"
        assert reference.ndim == 2, "reference_trajectory must be a 2D array"
        assert len(reference) > 0, "reference_trajectory must not be empty"
        assert states.shape[1] == reference.shape[1], (
            "current_states and reference_trajectory state dimensions must match"
        )
        assert np.all(np.isfinite(states)), "current_states must be finite"
        assert np.all(np.isfinite(reference)), "reference_trajectory must be finite"

        state_norms = np.einsum("ij,ij->i", states, states, optimize=True)
        reference_norms = np.einsum("ij,ij->i", reference, reference, optimize=True)
        squared_distances = (
            state_norms[:, np.newaxis]
            + reference_norms[np.newaxis, :]
            - 2.0 * (states @ reference.T)
        )
        np.maximum(squared_distances, 0.0, out=squared_distances)

        if self.phase_window is not None and phases is not None:
            # Local funnel: only reference indices within phase_window of the
            # commanded phase are admissible projections (issue #7983).
            n_ref = reference.shape[0]
            commanded = np.asarray(phases, dtype=np.float64) * (n_ref - 1)
            half_width = self.phase_window * n_ref
            index_grid = np.arange(n_ref, dtype=np.float64)[np.newaxis, :]
            outside = np.abs(index_grid - commanded[:, np.newaxis]) > half_width
            squared_distances = np.where(outside, np.inf, squared_distances)

        projected_phase_idx = np.argmin(squared_distances, axis=1)
        min_squared_distances = squared_distances[
            np.arange(states.shape[0]), projected_phase_idx
        ]

        transverse_cost = -_TRANSVERSE_COST_SCALE * min_squared_distances
        phase_velocity_reward = _PHASE_REWARD_SCALE * (
            projected_phase_idx / len(reference)
        )
        return transverse_cost + phase_velocity_reward

    @staticmethod
    def _rolling_means(values: np.ndarray, window_size: int) -> np.ndarray:
        """Return trailing rolling means in O(n) time."""
        assert window_size > 0, "window_size must be positive"
        prefix = np.concatenate(
            (np.array([0.0], dtype=np.float64), np.cumsum(values, dtype=np.float64))
        )
        ends = np.arange(1, len(values) + 1)
        starts = np.maximum(0, ends - window_size)
        window_sums = prefix[ends] - prefix[starts]
        window_counts = ends - starts
        return window_sums / window_counts

    def _estimate_convergence(
        self,
        reward_trajectory: list[float],
        window_size: int = 10,
        threshold: float = 0.01,
    ) -> tuple[int, float]:
        """Estimate convergence epoch and terminal variance from reward history.

        Args:
            reward_trajectory: List of rewards per episode.
            window_size: Rolling window for variance computation.
            threshold: Relative improvement threshold for convergence detection.

        Returns:
            Tuple of (convergence_epoch, terminal_variance).
        """
        if not reward_trajectory:
            return 0, float("inf")

        assert window_size > 0, "window_size must be positive"
        assert threshold >= 0.0 and np.isfinite(threshold), (
            "threshold must be finite and non-negative"
        )
        rewards = np.asarray(reward_trajectory, dtype=np.float64)
        assert np.all(np.isfinite(rewards)), "reward_trajectory must be finite"

        # Convergence: first epoch where rolling mean improvement < threshold
        rolling_means = self._rolling_means(rewards, window_size)

        convergence_epoch = len(rewards)
        for i in range(window_size, len(rolling_means)):
            prev = rolling_means[i - window_size]
            curr = rolling_means[i]
            denominator = max(abs(prev), _CONVERGENCE_EPSILON)
            if abs(curr - prev) / denominator < threshold:
                convergence_epoch = i
                break

        # Terminal variance: std of final window
        final_window = rewards[-window_size:]
        terminal_variance = (
            float(np.std(final_window, dtype=np.float64))
            if len(final_window) > 1
            else 0.0
        )

        return convergence_epoch, terminal_variance

    @staticmethod
    def build_reference(n_steps: int, state_dim: int) -> np.ndarray:
        """Build the sinusoidal reference manifold used by the benchmark."""
        assert n_steps > 1, "n_steps must be greater than 1"
        assert state_dim > 0, "state_dim must be positive"
        t = np.linspace(0, 2 * np.pi, n_steps)
        return np.stack([np.sin(t + i * 0.3) for i in range(state_dim)], axis=-1)

    def _policy_action(
        self, theta: np.ndarray, state: np.ndarray, phase: float
    ) -> np.ndarray:
        """Linear policy: action = theta @ [state, phase, 1]."""
        features = np.concatenate([state, [phase, 1.0]])
        return theta @ features

    def rollout(
        self,
        theta: np.ndarray,
        reference: np.ndarray,
        rng: np.random.Generator,
        process_noise: float = 0.0,
    ) -> tuple[float, np.ndarray]:
        """Execute one episode under the policy and return its mode reward.

        The state is produced by the policy's actions - it is not the reference
        plus a hand-written noise schedule (issue #7983), so the two reward
        modes visit genuinely different states.

        Args:
            theta: Policy parameters, shape ``(d, d + 2)``.
            reference: Reference manifold, shape ``(T, d)``.
            rng: Source of process noise.
            process_noise: Std-dev of additive plant noise.

        Returns:
            ``(total_reward, states)`` where ``states`` has shape ``(T, d)``.
        """
        n_steps, state_dim = reference.shape
        target_state = reference[-1]
        state = reference[0].copy()
        states = np.empty((n_steps, state_dim), dtype=np.float64)
        total = 0.0

        for step in range(n_steps):
            states[step] = state
            phase = step / max(1, n_steps - 1)
            if self.mode == "setpoint":
                total += self.setpoint_reward(state, target_state)
            action = self._policy_action(theta, state, phase)
            action = np.clip(action, -_ACTION_LIMIT, _ACTION_LIMIT)
            state = state + _PLANT_DT * action
            if process_noise > 0.0:
                state = state + rng.normal(0.0, process_noise, size=state_dim)

        if self.mode == "transverse":
            phases = np.arange(n_steps, dtype=np.float64) / max(1, n_steps - 1)
            total = float(
                np.sum(self._trajectory_funnel_rewards(states, reference, phases))
            )

        return float(total), states

    def evaluate_policy(
        self, theta: np.ndarray, reference: np.ndarray
    ) -> dict[str, float]:
        """Score a policy with mode-neutral metrics.

        Both metrics are independent of which reward trained the policy, so
        they are the only defensible basis for comparing the two modes.

        Args:
            theta: Policy parameters.
            reference: Reference manifold.

        Returns:
            ``mean_transverse_error`` (mean distance to the reference manifold)
            and ``terminal_setpoint_error`` (distance from the final state to
            the reference endpoint).
        """
        rng = np.random.default_rng(seed=0)
        _reward, states = self.rollout(theta, reference, rng)
        deltas = states[:, np.newaxis, :] - reference[np.newaxis, :, :]
        distances = np.sqrt(np.einsum("ijk,ijk->ij", deltas, deltas))
        return {
            "mean_transverse_error": float(np.mean(np.min(distances, axis=1))),
            "terminal_setpoint_error": float(
                np.linalg.norm(states[-1] - reference[-1])
            ),
        }

    def train_agent(
        self,
        n_iterations: int = 100,
        n_steps: int = 50,
        state_dim: int = 4,
        n_directions: int = 8,
        exploration_std: float = 0.05,
        learning_rate: float = 0.02,
        seed: int = 42,
    ) -> dict[str, float | str]:
        """Train a linear policy with Augmented Random Search under this mode.

        This replaces the former ``simulate_agent_training``, which contained
        no agent, no action and no learning update: it set
        ``state = reference[step] + noise(episode)`` and its apparent
        convergence was that hand-written annealing schedule (issue #7983).

        Args:
            n_iterations: ARS iterations (one policy update each).
            n_steps: Steps per episode.
            state_dim: Dimensionality of the state vector.
            n_directions: Perturbation directions sampled per iteration.
            exploration_std: Std-dev of the parameter perturbations.
            learning_rate: ARS step size.
            seed: RNG seed; runs are deterministic for a given seed.

        Returns:
            Dict with the mode's own learning metrics
            (``convergence_iteration``, ``terminal_return_std``,
            ``initial_return``, ``final_return``) plus the mode-neutral
            ``mean_transverse_error`` / ``terminal_setpoint_error``.

        Raises:
            AssertionError: If any size argument is non-positive.
        """
        assert n_iterations > 0, "n_iterations must be positive"
        assert n_directions > 0, "n_directions must be positive"
        assert exploration_std > 0.0, "exploration_std must be positive"

        reference = self.build_reference(n_steps, state_dim)
        rng = np.random.default_rng(seed=seed)
        theta = np.zeros((state_dim, state_dim + 2), dtype=np.float64)

        learning_curve: list[float] = []
        for _iteration in range(n_iterations):
            deltas = rng.normal(size=(n_directions, *theta.shape))
            plus = np.empty(n_directions)
            minus = np.empty(n_directions)
            for k in range(n_directions):
                plus[k], _ = self.rollout(
                    theta + exploration_std * deltas[k], reference, rng
                )
                minus[k], _ = self.rollout(
                    theta - exploration_std * deltas[k], reference, rng
                )

            advantage = plus - minus
            scale = float(np.std(np.concatenate([plus, minus])))
            if scale > _CONVERGENCE_EPSILON:
                theta = theta + (learning_rate / (n_directions * scale)) * np.tensordot(
                    advantage, deltas, axes=(0, 0)
                )

            evaluation, _states = self.rollout(theta, reference, rng)
            learning_curve.append(evaluation)

        self.learning_curve = learning_curve
        convergence_iteration, terminal_return_std = self._estimate_convergence(
            learning_curve
        )
        neutral = self.evaluate_policy(theta, reference)

        logger.info(
            "Training complete: mode=%s iterations=%d converged_at=%d "
            "return %.4f -> %.4f",
            self.mode,
            n_iterations,
            convergence_iteration,
            learning_curve[0],
            learning_curve[-1],
        )

        return {
            "mode": self.mode,
            "convergence_iteration": convergence_iteration,
            "terminal_return_std": terminal_return_std,
            "initial_return": learning_curve[0],
            "final_return": learning_curve[-1],
            **neutral,
        }


def _main() -> None:
    """Run both modes and report a conclusion derived from the numbers."""
    logging.basicConfig(level=logging.INFO)
    logger.info("--- Funnel vs Setpoint reward benchmark (ARS, linear policy) ---")

    res_sp = TrajectoryFunnelBenchmark("setpoint").train_agent()
    logger.info("Setpoint Results: %s", res_sp)

    res_fn = TrajectoryFunnelBenchmark("transverse").train_agent()
    logger.info("Transverse Results: %s", res_fn)

    # The two returns are on different scales and must not be compared; the
    # neutral tracking metrics are what the conclusion is drawn from (#7983).
    sp_err = float(res_sp["mean_transverse_error"])
    fn_err = float(res_fn["mean_transverse_error"])
    if fn_err < sp_err:
        verdict = "lower"
    elif fn_err > sp_err:
        verdict = "higher"
    else:
        verdict = "equal"
    logger.info(
        "Result: funnel-trained policy has %s mean transverse error than the "
        "setpoint-trained policy (%.6f vs %.6f); convergence iteration %d vs %d.",
        verdict,
        fn_err,
        sp_err,
        res_fn["convergence_iteration"],
        res_sp["convergence_iteration"],
    )


if __name__ == "__main__":
    _main()
