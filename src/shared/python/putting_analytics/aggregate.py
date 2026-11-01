"""Combine per-class putting results with encounter weights (issue #14174).

Each class gets its own outcome model (a green with that slope, orientation and
stimp) from a factory, its own optimal curve and best action. Weighted sums
treat classes as independent (distinct seed streams): with weights ``w``
normalised over the group, ``se = sqrt(sum (w_i * se_i)^2)``. As in
:mod:`.optimize`, the best action is chosen and scored on one sample.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from src.shared.python.contracts import require
from src.shared.python.putting_analytics.expected_putts import (
    ExpectedPuttsCurve,
    PuttAction,
    PuttOutcomeModel,
    distance_seed,
    validated_grid,
)
from src.shared.python.putting_analytics.library import ClassLibrary, PuttClass
from src.shared.python.putting_analytics import optimize as opt
from src.shared.python.putting_analytics import report as rpt
from src.shared.python.putting_analytics.report import Estimate

DEFAULT_GRID_FT = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 45.0)
DEFAULT_BAND_EDGES_FT = (2.0, 5.0, 10.0, 20.0, 60.0)

ModelFactory = Callable[[PuttClass], PuttOutcomeModel]


@dataclass(frozen=True)
class AggregateConfig:
    """Settings for one aggregate run."""

    actions: tuple[PuttAction, ...]
    tour: ExpectedPuttsCurve
    n: int
    seed: int
    grid_ft: tuple[float, ...] = DEFAULT_GRID_FT
    objective: str = "sg"
    holes_per_round: int = 18
    band_edges_ft: tuple[float, ...] = DEFAULT_BAND_EDGES_FT

    def __post_init__(self) -> None:
        require(len(self.actions) > 0 and self.n > 1, "need actions and n > 1")
        require(
            self.objective in opt.OBJECTIVES, f"objective must be in {opt.OBJECTIVES}"
        )
        require(self.holes_per_round > 0, "holes_per_round must be positive")
        edges = self.band_edges_ft
        require(len(edges) > 1 and bool(np.all(np.diff(edges) > 0)), "bad band edges")
        validated_grid(self.grid_ft)


def combine(
    estimates: Sequence[Estimate], weights: Sequence[float], scale: float = 1.0
) -> Estimate:
    """``scale`` times the weighted mean of independent estimates (weights renormalised)."""
    w = np.asarray(weights, dtype=float)
    require(w.size == len(estimates) and float(w.sum()) > 0.0, "need positive weight")
    w = w / w.sum()
    mean = float(w @ [e.mean for e in estimates])
    se = float(np.sqrt(np.sum((w * [e.se for e in estimates]) ** 2)))
    return Estimate(scale * mean, abs(scale) * se)


def evaluate_class(
    putt_class: PuttClass, model: PuttOutcomeModel, cfg: AggregateConfig, seed: int
) -> rpt.ClassResult:
    """Optimal curve for ``model``, then the best action and SG at the class distance."""
    grid = validated_grid(cfg.grid_ft)
    d = putt_class.distance_ft
    require(grid[0] <= d <= grid[-1], f"grid must cover class distance {d} ft")
    curve, _ = opt.optimal_expected_putts(model, grid, cfg.actions, cfg.n, seed)
    eval_seed = distance_seed(seed, grid.size)
    best = opt.best_action(
        opt.evaluate_actions(model, d, cfg.actions, curve, cfg.n, eval_seed),
        cfg.objective,
    )
    outcomes = model.simulate(d, best.action, cfg.n, eval_seed)
    sg = opt.strokes_gained(d, outcomes, curve, cfg.tour)
    return rpt.ClassResult(
        putt_class=putt_class,
        weight=0.0,
        aim_offset_deg=best.action.aim_offset_deg,
        pace_ft_past=best.action.pace_ft_past,
        make_rate=Estimate(best.make_rate, best.make_rate_se),
        expected_putts=Estimate(best.expected_strokes, best.expected_strokes_se),
        sg_player=Estimate(sg.sg_player, sg.sg_player_se),
        sg_tour=Estimate(sg.sg_tour, sg.sg_tour_se),
    )


def _band_results(
    results: Sequence[rpt.ClassResult], edges: Sequence[float]
) -> tuple[rpt.BandResult, ...]:
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        inside = [
            r
            for r in results
            if lo <= r.putt_class.distance_ft < hi
            or r.putt_class.distance_ft == hi == edges[-1]
        ]
        weights = [r.weight for r in inside]
        if inside:
            make = combine([r.make_rate for r in inside], weights)
            bands.append(rpt.BandResult(lo, hi, sum(weights), make))
    return tuple(bands)


def _pace_results(results: Sequence[rpt.ClassResult]) -> tuple[rpt.PaceResult, ...]:
    groups: defaultdict[tuple[float, float], list[rpt.ClassResult]] = defaultdict(list)
    for r in results:
        groups[(r.putt_class.distance_ft, r.putt_class.stimp)].append(r)
    return tuple(
        rpt.PaceResult(
            d,
            v,
            sum(r.weight for r in rows),
            float(
                np.average(
                    [r.pace_ft_past for r in rows], weights=[r.weight for r in rows]
                )
            ),
        )
        for (d, v), rows in sorted(groups.items())
    )


def _totals(results: Sequence[rpt.ClassResult], cfg: AggregateConfig) -> rpt.Totals:
    weights, rounds = [r.weight for r in results], float(cfg.holes_per_round)

    def total(field: str, scale: float = 1.0) -> Estimate:
        return combine([getattr(r, field) for r in results], weights, scale)

    return rpt.Totals(
        total("make_rate"),
        total("expected_putts", rounds),
        total("sg_player", rounds),
        total("sg_tour", rounds),
    )


def aggregate(
    library: ClassLibrary, factory: ModelFactory, cfg: AggregateConfig
) -> rpt.PuttingReport:
    """Run every class and combine with the library's encounter weights.

    Class ``i`` is seeded by ``distance_seed(cfg.seed, i)``, so a run is
    reproducible and adding classes at the end leaves earlier ones unchanged.
    """
    results = tuple(
        replace(
            evaluate_class(c, factory(c), cfg, distance_seed(cfg.seed, i)), weight=w
        )
        for i, (c, w) in enumerate(zip(library.classes, library.weights, strict=True))
    )
    return rpt.PuttingReport(
        seed=cfg.seed,
        samples_per_action=cfg.n,
        objective=cfg.objective,
        holes_per_round=cfg.holes_per_round,
        classes=results,
        bands=_band_results(results, cfg.band_edges_ft),
        pace=_pace_results(results),
        totals=_totals(results, cfg),
    )
