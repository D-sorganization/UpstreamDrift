"""Cached, versioned Monte-Carlo dispersion tables (SHOW-B3/B3b, #14567/#14610).

Per ``(skill, club, lie, wind, slope)`` cell: mean / sd / correlation of the
aim-frame landing offset (yards) from an injected landing simulator. Every cell
shares one standard-normal block (common random numbers, prefix-stable in ``n``).
The hash covers cells + ``(family, seed, n)`` and the injected ground/lie provenance; wall-clock runtime is never hashed.
Scope: carry by default; an injected ``ground`` model (B3b, ``ground_runout``) turns
landing states into rest positions and is recorded in the provenance; slope defaults
to flat, other bins come from an injected ``slope_modifier`` (B6); ``club`` repeats
``skill``; lie is an identity unless a ``lie_modifier`` is injected.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import ensure, require
from src.shared.python.sg_optimizer.shot_model.skill_family import (
    IMPACT_FIELDS,
    SkillFamily,
)

# (impacts, skill, wind mph pair, slope bin, lie) -> rest position (n, 2) yards;
# a simulator with a ``spin_factor`` (n,) parameter also gets the lie spin
# (#14569); one without it models no spin and gets the rows alone.
LandingSimulator = Callable[..., Any]
# (impacts, lie) -> impacts; a modifier with ``with_spin`` also returns spin.
LieModifier = Callable[[NDArray[np.float64], str], NDArray[np.float64]]


class GroundModel(Protocol):
    """Landing states ``(n, k)`` + ``(lie, slope, skill)`` -> rest offsets ``(n, 2)`` yards."""

    @property
    def provenance(self) -> dict[str, Any]: ...

    def __call__(
        self, landing: NDArray[np.float64], lie: str, slope: str, skill: str
    ) -> NDArray[np.float64]: ...


WIND_BINS: dict[str, tuple[float, float]] = {
    "calm": (0.0, 0.0),
    "head_5": (5.0, 0.0),
    "tail_5": (-5.0, 0.0),
    "cross_5": (0.0, 5.0),
    "head_15": (15.0, 0.0),
    "tail_15": (-15.0, 0.0),
    "cross_15": (0.0, 15.0),
}
DEFAULT_WINDS = ("calm", "head_5", "tail_5", "cross_5")  # the original B3 bins
DEFAULT_SLOPES = (
    "flat",
)  # default slope; other bins come from ``slope_wind_bins`` (B6)
CONVERGENCE_TOL = 0.02
# Real pipeline with runout: n=12000 worst skill ~1.7% vs n/2 at seed 0 (B3b).
DEFAULT_N = 12000
SCHEMA = 1


def standard_normals(seed: int, n: int) -> NDArray[np.float64]:
    """``(n, 6)`` standard normals; the first ``k`` rows never depend on ``n``."""
    require(n > 0, "n must be > 0", n)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed])))
    return np.asarray(rng.standard_normal((n, len(IMPACT_FIELDS))), dtype=np.float64)


def _impacts(dist: Any, z: NDArray[np.float64]) -> NDArray[np.float64]:
    vals, vecs = np.linalg.eigh(dist.covariance())
    factor = vecs * np.sqrt(np.clip(vals, 0.0, None))
    return np.asarray(np.array(dist.mean) + z @ factor.T, dtype=np.float64)


@dataclass(frozen=True)
class DispersionCell:
    skill: str
    club: str
    lie: str
    wind: str
    slope: str
    mean: tuple[float, float]
    sd: tuple[float, float]
    corr: float
    n: int

    @property
    def key(self) -> tuple[str, str, str, str, str]:
        return (self.skill, self.club, self.lie, self.wind, self.slope)


def _cell_stats(land: NDArray[np.float64]) -> tuple[Any, Any, float]:
    mean, sd = land.mean(axis=0), land.std(axis=0, ddof=1)
    denom = float(sd[0] * sd[1])
    corr = float(np.asarray(np.corrcoef(land.T))[0, 1]) if denom > 0 else 0.0
    return (
        (float(mean[0]), float(mean[1])),
        (float(sd[0]), float(sd[1])),
        corr,
    )


def _hash(cells: tuple[DispersionCell, ...], meta: dict[str, Any]) -> str:
    body = [[*c.key, *map(float.hex, (*c.mean, *c.sd, c.corr)), c.n] for c in cells]
    blob = json.dumps({"meta": meta, "cells": body}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Convergence:
    converged: bool
    max_rel_change: float


@dataclass(frozen=True)
class DispersionTable:
    cells: tuple[DispersionCell, ...]
    provenance: dict[str, Any]
    runtime: dict[str, float] = field(default_factory=dict, compare=False)

    @property
    def table_hash(self) -> str:
        return str(self.provenance["table_hash"])

    def club_label(self, skill: str | None) -> str | None:
        """The club label recorded for ``skill``, else the skill name itself.

        Law of Demeter: callers ask the table rather than reach into ``provenance``.
        """
        return self.provenance.get("clubs", {}).get(skill, skill)

    def lookup(
        self, skill: str, club: str, lie: str, wind: str, slope: str
    ) -> DispersionCell:
        key = (skill, club, lie, wind, slope)
        for c in self.cells:
            if c.key == key:
                return c
        require(False, f"no dispersion cell for {key}")
        raise AssertionError  # unreachable: require raises

    def with_convergence(self, conv: Convergence) -> DispersionTable:
        """Record the check in the hashed provenance (it is deterministic)."""
        meta = {k: v for k, v in self.provenance.items() if k != "table_hash"}
        meta["convergence"] = {
            "converged": conv.converged,
            "max_rel_change": round(conv.max_rel_change, 6),
        }
        prov = {**meta, "table_hash": _hash(self.cells, meta)}
        return DispersionTable(self.cells, prov, self.runtime)

    def to_json(self, path: str | Path) -> None:
        raw = {
            "schema": SCHEMA,
            "provenance": self.provenance,
            "runtime": self.runtime,
            "cells": [
                {**c.__dict__, "mean": list(c.mean), "sd": list(c.sd)}
                for c in self.cells
            ],
        }
        Path(path).write_text(json.dumps(raw, indent=1) + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> DispersionTable:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        require(raw.get("schema") == SCHEMA, "unsupported dispersion-table schema")
        cells = tuple(
            DispersionCell(**{**c, "mean": tuple(c["mean"]), "sd": tuple(c["sd"])})
            for c in raw["cells"]
        )
        prov = dict(raw["provenance"])
        meta = {k: v for k, v in prov.items() if k != "table_hash"}
        require(_hash(cells, meta) == prov["table_hash"], "table hash mismatch")
        return cls(cells, prov, dict(raw.get("runtime", {})))


@dataclass(frozen=True)
class TableGrid:
    """The cell grid of a table: skills (``None`` = all), lies, wind and slope bins."""

    skills: tuple[str, ...] | None = None
    lies: tuple[str, ...] = ("fairway",)
    winds: tuple[str, ...] = DEFAULT_WINDS
    slopes: tuple[str, ...] = DEFAULT_SLOPES
    clubs: dict[str, str] | None = None  # skill -> club label; None = the skill name


SlopeModifier = Callable[[str], Callable[[NDArray[np.float64]], NDArray[np.float64]]]


def build_table(
    family: SkillFamily,
    simulate: LandingSimulator,
    *,
    skills: tuple[str, ...] | None = None,
    lies: tuple[str, ...] = ("fairway",),
    winds: tuple[str, ...] = DEFAULT_WINDS,
    n: int = DEFAULT_N,
    seed: int = 0,
    lie_modifier: LieModifier | None = None,
) -> DispersionTable:
    """Flat-slope table over ``skills x lies x winds`` (see :func:`build_grid_table`)."""
    grid = TableGrid(skills, lies, winds)
    return build_grid_table(
        family, simulate, grid, n=n, seed=seed, lie_modifier=lie_modifier
    )


def _lie_rows(
    modifier: LieModifier | None, rows: NDArray[np.float64], lie: str
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Lie-modified rows and the per-shot spin factor, None when it is identity."""
    if modifier is None:
        return rows, None
    with_spin = getattr(modifier, "with_spin", None)
    if with_spin is None:
        return modifier(rows, lie), None
    out, spin = with_spin(rows, lie)
    spin = np.asarray(spin, dtype=float)
    return out, None if bool(np.all(spin == 1.0)) else spin


def _simulator_flags(simulate: LandingSimulator) -> tuple[bool, bool]:
    """Simulator capability flags: (takes_spin, takes_slope_lie)."""
    sig = inspect.signature(simulate)
    takes_spin = "spin_factor" in sig.parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    takes_slope_lie = (
        any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
        or ("slope" in sig.parameters and "lie" in sig.parameters)
        or len(
            [
                p
                for p in sig.parameters.values()
                if p.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
        )
        >= 5
    )
    return takes_spin, takes_slope_lie


def _build_meta(
    family: SkillFamily,
    names: tuple[str, ...],
    clubs: dict[str, str] | None,
    seed: int,
    n: int,
    ground: GroundModel | None,
    lie_modifier: LieModifier | None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "family": family.name,
        "family_version": family.provenance.version,
        "seed": seed,
        "n": n,
    }
    if clubs is not None:
        meta["clubs"] = {k: clubs[k] for k in names}
    for part in (
        ground.provenance if ground is not None else {},
        getattr(lie_modifier, "provenance", {}),
    ):
        require(
            not (set(part) & set(meta)),
            "provenance key collision",
            set(part) & set(meta),
        )
        meta.update(part)
    return meta


def build_grid_table(
    family: SkillFamily,
    simulate: LandingSimulator,
    grid: TableGrid,
    *,
    n: int = DEFAULT_N,
    seed: int = 0,
    lie_modifier: LieModifier | None = None,
    slope_modifier: SlopeModifier | None = None,
    ground: GroundModel | None = None,
) -> DispersionTable:
    """Simulate every grid cell with shared (CRN) draws; fail closed on bad inputs.

    Non-flat slope bins need ``slope_modifier`` (``slope_wind_bins.make_slope_modifier``).
    With ``ground`` the simulator must return landing states (``ground_runout``
    columns), and cells hold rest positions; its ``provenance`` joins the hashed meta.
    """
    names = grid.skills if grid.skills is not None else tuple(family.skills)
    require(set(names) <= set(family.skills), f"skills not in {family.name!r}")
    winds = grid.winds
    require(set(winds) <= set(WIND_BINS), f"unknown wind bins; have {tuple(WIND_BINS)}")
    require(
        grid.slopes == DEFAULT_SLOPES or slope_modifier is not None,
        "slope bins need a slope_modifier",
    )
    clubs = grid.clubs
    require(
        clubs is None or set(names) <= set(clubs),
        "club mapping must cover every skill in the grid",
    )
    takes_spin, takes_slope_lie = _simulator_flags(simulate)
    t0, z, cells = time.perf_counter(), standard_normals(seed, n), []
    for skill in names:
        base = _impacts(family.skills[skill].distribution, z)
        for lie in grid.lies:
            imp, spin = _lie_rows(lie_modifier, base, lie)
            extra = {"spin_factor": spin} if takes_spin and spin is not None else {}
            for slope in grid.slopes:
                sloped = imp if slope_modifier is None else slope_modifier(slope)(imp)
                for wind in winds:
                    args: tuple[Any, ...] = (
                        (sloped, skill, WIND_BINS[wind], slope, lie)
                        if takes_slope_lie
                        else (sloped, skill, WIND_BINS[wind])
                    )
                    land = np.asarray(simulate(*args, **extra), dtype=float)
                    if ground is not None:
                        require(land.ndim == 2 and len(land) == n, "landing states")
                        land = np.asarray(ground(land, lie, slope, skill), dtype=float)
                    require(land.shape == (n, 2), f"simulator returned {land.shape}")
                    require(
                        bool(np.isfinite(land).all()), "simulator returned non-finite"
                    )
                    mean, sd, corr = _cell_stats(land)
                    cells.append(
                        DispersionCell(
                            skill,
                            skill if clubs is None else clubs[skill],
                            lie,
                            wind,
                            slope,
                            mean,
                            sd,
                            corr,
                            n,
                        )
                    )
    meta = _build_meta(family, names, clubs, seed, n, ground, lie_modifier)
    table = DispersionTable(
        tuple(cells),
        {**meta, "table_hash": _hash(tuple(cells), meta)},
        {"total_s": time.perf_counter() - t0, "cells": float(len(cells))},
    )
    ensure(len(table.cells) > 0, "table must not be empty")
    return table


def check_convergence(
    family: SkillFamily,
    simulate: LandingSimulator,
    skill: str,
    *,
    n: int,
    seed: int,
    ground: GroundModel | None = None,
) -> Convergence:
    """Compare n vs n/2 shared draws; stable within 2% of the sd means converged.

    With ``ground`` the check runs on rest positions (fairway, flat).
    """
    require(n >= 4, "n must be >= 4", n)
    z = standard_normals(seed, n)
    imp = _impacts(family.skills[skill].distribution, z)
    _, takes_slope_lie = _simulator_flags(simulate)
    args: tuple[Any, ...] = (
        (imp, skill, WIND_BINS["calm"], "flat", "fairway")
        if takes_slope_lie
        else (imp, skill, WIND_BINS["calm"])
    )
    full = np.asarray(simulate(*args), dtype=float)
    if ground is not None:
        full = np.asarray(ground(full, "fairway", "flat", skill), dtype=float)
    half = full[: n // 2]
    (m1, s1, _), (m2, s2, _) = _cell_stats(full), _cell_stats(half)
    scale = np.maximum(np.array(s1), 1e-12)
    change = np.maximum(
        np.abs(np.array(m1) - np.array(m2)) / scale,
        np.abs(np.array(s1) - np.array(s2)) / scale,
    )
    worst = float(change.max())
    return Convergence(worst < CONVERGENCE_TOL, worst)


def _csv(text: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    return default if text is None else tuple(t for t in text.split(",") if t)


def _cli_table(a: Any, family: SkillFamily) -> tuple[DispersionTable, Convergence]:
    """Carry table, or with ``--ground`` the rest-position table (see ``main``)."""
    from src.shared.python.sg_optimizer.shot_model.physics_simulator import (
        REPRESENTATIVE_CLUB,
        PipelineLandingSimulator,
    )

    if not a.ground:
        sim = PipelineLandingSimulator()
        table = build_table(family, sim, n=a.n, seed=a.seed)
        ground = None
    else:
        from src.shared.python.sg_optimizer.shot_model.dispersion_lie import (
            lie_modifier_for_table,
        )
        from src.shared.python.sg_optimizer.shot_model.ground_runout import (
            SlopeAwareGround,
        )
        from src.shared.python.sg_optimizer.shot_model.slope_wind_bins import (
            SLOPE_BINS,
            make_slope_modifier,
        )

        sim, ground = PipelineLandingSimulator(landing_state=True), SlopeAwareGround()
        grid = TableGrid(
            None,
            _csv(a.lies, ("tee", "fairway", "rough", "sand")),
            _csv(a.winds, DEFAULT_WINDS),
            _csv(a.slopes, tuple(SLOPE_BINS)),
            clubs=REPRESENTATIVE_CLUB,
        )
        table = build_grid_table(
            family,
            sim,
            grid,
            n=a.n,
            seed=a.seed,
            lie_modifier=lie_modifier_for_table(a.seed),
            slope_modifier=make_slope_modifier,
            ground=ground,
        )
    worst = max(
        check_convergence(
            family, sim, s, n=a.n, seed=a.seed, ground=ground
        ).max_rel_change
        for s in family.skills
    )
    return table, Convergence(worst < CONVERGENCE_TOL, worst)


def main(argv: list[str] | None = None) -> int:
    """Regenerate a table: ``python -m ...dispersion_table --family tour --out t.json``.

    ``--ground`` builds the rest-position table instead of the carry table: bounce
    and roll on every slope bin, the lie modifier, a club label per skill, and the
    convergence of the rest positions, all in the hashed provenance.
    """
    import argparse

    from src.shared.python.sg_optimizer.shot_model.skill_registry import (
        get_skill_family,
    )

    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--family", default="tour")
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--ground", action="store_true", help="rest-position table")
    p.add_argument("--lies", help="comma list (with --ground; default all)")
    p.add_argument("--slopes", help="comma list (with --ground; default all)")
    p.add_argument("--winds", help="comma list (with --ground; default B3 bins)")
    a = p.parse_args(argv)
    table, conv = _cli_table(a, get_skill_family(a.family))
    table.with_convergence(conv).to_json(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
