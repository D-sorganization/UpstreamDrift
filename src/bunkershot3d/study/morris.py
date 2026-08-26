"""Morris elementary-effects screening.

Morris screening is the cheap step that comes *first* in a real workflow: at
``r (D + 1)`` model evaluations it tells you which factors are worth spending a
Sobol' analysis on. For a 10-factor space, ``r = 10`` costs 110 runs against
the ``N (D + 2) = 12,288`` a modest Sobol' analysis needs.

Each trajectory is a walk through the unit hypercube that changes exactly one
factor per step, so a step's output difference *is* that factor's elementary
effect. Aggregating over trajectories gives three statistics:

- ``mu`` -- mean effect; near zero either for an inert factor **or** for one
  whose effect changes sign, which is why it is never used alone;
- ``mu_star`` -- mean absolute effect, the factor-importance measure
  (Campolongo et al. 2007). An inert factor scores exactly zero;
- ``sigma`` -- spread of the effects, large when the factor interacts or acts
  non-linearly.

Effects are reported per *unit-cube* step, so factors with different physical
units remain comparable.

References:
    Morris, M. D. (1991). Factorial sampling plans for preliminary
    computational experiments. *Technometrics*, 33(2), 161-174.

    Campolongo, F., Cariboni, J. and Saltelli, A. (2007). An effective
    screening design for sensitivity analysis of large models. *Environmental
    Modelling & Software*, 22(10), 1509-1518.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.shared.python.core.contracts import require

from .design_space import DesignSpace
from .manifest import StudyManifest
from .rng import SeedRecord, new_seed_record

__all__ = [
    "MorrisDesign",
    "MorrisResult",
    "morris_design",
    "morris_screening",
    "morris_statistics",
]

ModelFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True, eq=False)
class MorrisDesign:
    """A set of Morris trajectories awaiting evaluation.

    Attributes:
        space: The design space sampled.
        trajectories: ``(r, D + 1, D)`` unit-cube trajectories.
        delta: The grid step, ``p / (2 (p - 1))``.
        n_levels: Number of grid levels ``p``.
        manifest: Provenance, including the entropy that produced it.
    """

    space: DesignSpace
    trajectories: np.ndarray
    delta: float
    n_levels: int
    manifest: StudyManifest

    @property
    def n_trajectories(self) -> int:
        """Number of trajectories ``r``.

        Returns:
            The trajectory count.
        """
        return int(self.trajectories.shape[0])

    @property
    def n_evaluations(self) -> int:
        """Total model evaluations required, ``r (D + 1)``.

        Returns:
            The evaluation count.
        """
        return int(self.trajectories.shape[0] * self.trajectories.shape[1])

    def design_matrix(self) -> np.ndarray:
        """Flatten the trajectories into one physical design matrix.

        Returns:
            An ``(r (D + 1), D)`` array in physical units, trajectory-major.
        """
        flat = self.trajectories.reshape(-1, self.space.dimension)
        return self.space.to_physical(flat)


@dataclass(frozen=True, eq=False)
class MorrisResult:
    """Morris screening statistics.

    Attributes:
        names: Parameter names, in column order.
        mu: ``(D,)`` mean elementary effect.
        mu_star: ``(D,)`` mean absolute elementary effect.
        sigma: ``(D,)`` standard deviation of the elementary effects.
        mu_star_conf: ``(D,)`` half-width of the 95 % normal interval on
            ``mu_star`` (``1.96 * sigma_|EE| / sqrt(r)``).
        elementary_effects: ``(r, D)`` raw effects, one row per trajectory.
        n_trajectories: Number of trajectories ``r``.
        n_evaluations: Model evaluations consumed, ``r (D + 1)``.
        manifest: Provenance for the sample that produced the statistics.
    """

    names: tuple[str, ...]
    mu: np.ndarray
    mu_star: np.ndarray
    sigma: np.ndarray
    mu_star_conf: np.ndarray
    elementary_effects: np.ndarray
    n_trajectories: int
    n_evaluations: int
    manifest: StudyManifest | None = None

    def ranked(self) -> tuple[str, ...]:
        """Order parameter names from most to least influential by ``mu_star``.

        Returns:
            Parameter names, most influential first.
        """
        order = np.argsort(-self.mu_star, kind="stable")
        return tuple(self.names[i] for i in order)

    def inert(self, threshold: float = 1e-12) -> tuple[str, ...]:
        """Identify factors whose mean absolute effect is negligible.

        Args:
            threshold: ``mu_star`` at or below this counts as inert. The
                default only catches exactly-inert factors; scale it to the
                output magnitude for a practical screen.

        Returns:
            The names of the inert factors, in column order.
        """
        return tuple(
            name
            for name, value in zip(self.names, self.mu_star, strict=True)
            if value <= threshold
        )


def _one_trajectory(
    dimension: int,
    delta: float,
    n_levels: int,
    generator: np.random.Generator,
) -> np.ndarray:
    """Build a single Morris trajectory in the unit cube.

    Implements ``B* = (J x* + (delta / 2) [(2 B - J) D* + J]) P*`` from
    Morris (1991): ``B`` strictly lower triangular, ``D*`` a random sign
    matrix and ``P*`` a random permutation.

    Args:
        dimension: Number of factors ``D``.
        delta: Grid step.
        n_levels: Number of grid levels ``p``.
        generator: Seeded generator.

    Returns:
        A ``(D + 1, D)`` array whose consecutive rows differ in exactly one
        column, by ``+delta`` or ``-delta``.
    """
    # Base point on the grid, restricted so that a +delta step stays in [0, 1].
    max_level = n_levels - 1 - int(round(delta * (n_levels - 1)))
    base = generator.integers(0, max_level + 1, size=dimension) / (n_levels - 1)

    lower = np.tril(np.ones((dimension + 1, dimension)), -1)
    ones = np.ones((dimension + 1, dimension))
    signs = np.diag(generator.choice([-1.0, 1.0], size=dimension))
    permutation = np.eye(dimension)[generator.permutation(dimension)]

    stepped = (2.0 * lower - ones) @ signs + ones
    trajectory = (ones * base + 0.5 * delta * stepped) @ permutation
    return np.clip(trajectory, 0.0, 1.0)


def morris_design(
    space: DesignSpace,
    n_trajectories: int,
    n_levels: int = 4,
    seed: int | SeedRecord | None = None,
    oversample: int = 1,
) -> MorrisDesign:
    """Build a Morris screening design.

    Args:
        space: The design space to screen.
        n_trajectories: Number of trajectories ``r``. Ten is the usual
            starting point; the cost is ``r (D + 1)``.
        n_levels: Number of grid levels ``p``; must be even so that
            ``delta = p / (2 (p - 1))`` keeps the design balanced.
        seed: Explicit entropy or seed record; ``None`` draws fresh entropy.
        oversample: Generate ``oversample * r`` candidate trajectories and
            greedily keep the ``r`` that are furthest apart (Campolongo's
            spread criterion, greedy rather than exhaustive). ``1`` disables
            the selection.

    Returns:
        The screening design.

    Raises:
        ValueError: If any argument is out of range.
    """
    if n_trajectories <= 0:
        raise ValueError(f"n_trajectories must be positive, got {n_trajectories}")
    if n_levels < 2 or n_levels % 2 != 0:
        raise ValueError(f"n_levels must be an even integer >= 2, got {n_levels}")
    if oversample < 1:
        raise ValueError(f"oversample must be >= 1, got {oversample}")

    record = new_seed_record(seed)
    generator = record.generator()
    delta = n_levels / (2.0 * (n_levels - 1))
    dimension = space.dimension

    n_candidates = n_trajectories * oversample
    candidates = np.stack(
        [
            _one_trajectory(dimension, delta, n_levels, generator)
            for _ in range(n_candidates)
        ]
    )
    trajectories = (
        candidates
        if oversample == 1
        else candidates[_greedy_spread_selection(candidates, n_trajectories)]
    )

    require(
        trajectories.shape == (n_trajectories, dimension + 1, dimension),
        "Morris design has the wrong shape",
        value=trajectories.shape,
    )
    manifest = StudyManifest(
        seed=record,
        method="morris",
        parameter_names=space.names,
        n_samples=n_trajectories * (dimension + 1),
        extra={
            "n_trajectories": n_trajectories,
            "n_levels": n_levels,
            "delta": delta,
            "oversample": oversample,
        },
    )
    return MorrisDesign(
        space=space,
        trajectories=trajectories,
        delta=delta,
        n_levels=n_levels,
        manifest=manifest,
    )


def _greedy_spread_selection(candidates: np.ndarray, keep: int) -> list[int]:
    """Greedily select well-separated trajectories (farthest-point traversal).

    Campolongo's exhaustive search over all ``C(candidates, keep)`` subsets is
    combinatorial; this greedy traversal is ``O(candidates * keep)`` and keeps
    most of the spread benefit.

    Args:
        candidates: ``(m, D + 1, D)`` candidate trajectories.
        keep: How many to keep.

    Returns:
        Indices of the selected trajectories, in selection order.
    """
    flat = candidates.reshape(candidates.shape[0], -1)
    centroid = flat.mean(axis=0)
    diff = flat - centroid
    # ⚡ Bolt: np.einsum is used to calculate squared magnitude for argmax, avoiding square root and intermediate allocations.
    chosen = [int(np.argmax(np.einsum("ij,ij->i", diff, diff)))]
    distance = np.linalg.norm(flat - flat[chosen[0]], axis=1)
    while len(chosen) < keep:
        nxt = int(np.argmax(distance))
        chosen.append(nxt)
        distance = np.minimum(distance, np.linalg.norm(flat - flat[nxt], axis=1))
        distance[nxt] = -np.inf
    return chosen


def morris_statistics(
    design: MorrisDesign,
    outputs: np.ndarray,
) -> MorrisResult:
    """Reduce trajectory outputs to Morris statistics.

    Args:
        design: The design the outputs were produced from.
        outputs: ``(r (D + 1),)`` model outputs in the row order of
            :meth:`MorrisDesign.design_matrix`.

    Returns:
        The screening statistics.

    Raises:
        ValueError: If ``outputs`` has the wrong length or is non-finite.
    """
    flat = np.asarray(outputs, dtype=float).ravel()
    if flat.size != design.n_evaluations:
        raise ValueError(f"expected {design.n_evaluations} outputs, got {flat.size}")
    # Safety-critical: NaN would propagate into a silently plausible ranking.
    if not np.all(np.isfinite(flat)):
        raise ValueError("model outputs contain NaN or inf; effects are undefined")

    dimension = design.space.dimension
    n_traj = design.n_trajectories
    values = flat.reshape(n_traj, dimension + 1)
    points = design.trajectories

    effects = np.zeros((n_traj, dimension), dtype=float)
    for t in range(n_traj):
        steps = np.diff(points[t], axis=0)
        changed = np.argmax(np.abs(steps), axis=1)
        step_sizes = steps[np.arange(dimension), changed]
        if np.any(step_sizes == 0.0):
            raise ValueError(
                f"trajectory {t} has a zero-length step; the design is degenerate"
            )
        effects[t, changed] = np.diff(values[t]) / step_sizes

    mu = effects.mean(axis=0)
    mu_star = np.abs(effects).mean(axis=0)
    sigma = effects.std(axis=0, ddof=1) if n_traj > 1 else np.zeros(dimension)
    conf = (
        1.96 * np.abs(effects).std(axis=0, ddof=1) / np.sqrt(n_traj)
        if n_traj > 1
        else np.zeros(dimension)
    )
    return MorrisResult(
        names=design.space.names,
        mu=mu,
        mu_star=mu_star,
        sigma=sigma,
        mu_star_conf=conf,
        elementary_effects=effects,
        n_trajectories=n_traj,
        n_evaluations=design.n_evaluations,
        manifest=design.manifest,
    )


def morris_screening(
    space: DesignSpace,
    model: ModelFunction,
    n_trajectories: int = 10,
    n_levels: int = 4,
    seed: int | SeedRecord | None = None,
    oversample: int = 1,
) -> MorrisResult:
    """Run a complete Morris screening of a model.

    Args:
        space: The design space to screen.
        model: Vectorised model, mapping ``(m, D)`` design points to ``(m,)``
            outputs.
        n_trajectories: Number of trajectories ``r``.
        n_levels: Number of grid levels ``p`` (even).
        seed: Explicit entropy or seed record; ``None`` draws fresh entropy.
        oversample: Candidate oversampling factor for trajectory selection.

    Returns:
        The screening statistics.

    Raises:
        ValueError: If the model returns the wrong number of outputs.
    """
    design = morris_design(
        space,
        n_trajectories,
        n_levels=n_levels,
        seed=seed,
        oversample=oversample,
    )
    outputs = np.asarray(model(design.design_matrix()), dtype=float).ravel()
    if outputs.size != design.n_evaluations:
        raise ValueError(
            f"model returned {outputs.size} outputs, expected {design.n_evaluations}"
        )
    return morris_statistics(design, outputs)
