"""Cached, versioned Monte-Carlo dispersion tables (SHOW-B3, #14567).

Per ``(skill, club, lie, wind, slope)`` cell: mean / sd / correlation of the
aim-frame landing offset (yards) from an injected landing simulator. Every cell
shares one standard-normal block (common random numbers, prefix-stable in ``n``).
The hash covers cells + ``(family, seed, n)``; wall-clock runtime is never hashed.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import ensure, require
from src.shared.python.sg_optimizer.shot_model.skill_family import (
    IMPACT_FIELDS,
    SkillFamily,
)

# (impacts (n, 6), skill, (headwind_mph, crosswind_mph)) -> landing (n, 2) yards
LandingSimulator = Callable[[NDArray[np.float64], str, tuple[float, float]], Any]
LieModifier = Callable[[NDArray[np.float64], str], NDArray[np.float64]]

WIND_BINS: dict[str, tuple[float, float]] = {
    "calm": (0.0, 0.0),
    "head_5": (5.0, 0.0),
    "tail_5": (-5.0, 0.0),
    "cross_5": (0.0, 5.0),
}
CONVERGENCE_TOL = 0.02
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
    corr = float(np.corrcoef(land.T)[0, 1]) if denom > 0 else 0.0
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
class DispersionTable:
    cells: tuple[DispersionCell, ...]
    provenance: dict[str, Any]
    runtime: dict[str, float] = field(default_factory=dict, compare=False)

    @property
    def table_hash(self) -> str:
        return str(self.provenance["table_hash"])

    def has_cell(self, *key: str) -> bool:
        return any(c.key == key for c in self.cells)

    def lookup(
        self, skill: str, club: str, lie: str, wind: str, slope: str
    ) -> DispersionCell:
        key = (skill, club, lie, wind, slope)
        for c in self.cells:
            if c.key == key:
                return c
        require(False, f"no dispersion cell for {key}")
        raise AssertionError  # unreachable: require raises

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


def build_table(
    family: SkillFamily,
    simulate: LandingSimulator,
    *,
    skills: tuple[str, ...] | None = None,
    lies: tuple[str, ...] = ("fairway",),
    winds: tuple[str, ...] = tuple(WIND_BINS),
    slopes: tuple[str, ...] = ("flat",),
    n: int = 2000,
    seed: int = 0,
    lie_modifier: LieModifier | None = None,
) -> DispersionTable:
    """Simulate every cell with shared (CRN) draws; fail closed on bad inputs."""
    names = skills if skills is not None else tuple(family.skills)
    require(set(names) <= set(family.skills), f"skills not in {family.name!r}")
    require(set(winds) <= set(WIND_BINS), f"unknown wind bins; have {tuple(WIND_BINS)}")
    t0, z, cells = time.perf_counter(), standard_normals(seed, n), []
    for skill in names:
        base = _impacts(family.skills[skill].distribution, z)
        for lie in lies:
            imp = lie_modifier(base, lie) if lie_modifier is not None else base
            for wind in winds:
                land = np.asarray(simulate(imp, skill, WIND_BINS[wind]), dtype=float)
                require(land.shape == (n, 2), f"simulator returned {land.shape}")
                require(bool(np.isfinite(land).all()), "simulator returned non-finite")
                mean, sd, corr = _cell_stats(land)
                for slope in slopes:
                    cells.append(
                        DispersionCell(
                            skill, skill, lie, wind, slope, mean, sd, corr, n
                        )
                    )
    meta = {
        "family": family.name,
        "family_version": family.provenance.version,
        "seed": seed,
        "n": n,
    }
    table = DispersionTable(
        tuple(cells),
        {**meta, "table_hash": _hash(tuple(cells), meta)},
        {"total_s": time.perf_counter() - t0, "cells": float(len(cells))},
    )
    ensure(len(table.cells) > 0, "table must not be empty")
    return table


@dataclass(frozen=True)
class Convergence:
    converged: bool
    max_rel_change: float


def check_convergence(
    family: SkillFamily,
    simulate: LandingSimulator,
    skill: str,
    *,
    n: int,
    seed: int,
) -> Convergence:
    """Compare n vs n/2 shared draws; stable within 2% of the sd means converged."""
    require(n >= 4, "n must be >= 4", n)
    z = standard_normals(seed, n)
    imp = _impacts(family.skills[skill].distribution, z)
    full = np.asarray(simulate(imp, skill, WIND_BINS["calm"]), dtype=float)
    half = full[: n // 2]
    (m1, s1, _), (m2, s2, _) = _cell_stats(full), _cell_stats(half)
    scale = np.maximum(np.array(s1), 1e-12)
    change = np.maximum(
        np.abs(np.array(m1) - np.array(m2)) / scale,
        np.abs(np.array(s1) - np.array(s2)) / scale,
    )
    worst = float(change.max())
    return Convergence(worst < CONVERGENCE_TOL, worst)


def main(argv: list[str] | None = None) -> int:
    """Regenerate a table: ``python -m ...dispersion_table --family tour --out t.json``."""
    import argparse

    from src.shared.python.sg_optimizer.shot_model.physics_simulator import (
        PipelineLandingSimulator,
    )
    from src.shared.python.sg_optimizer.shot_model.skill_registry import (
        get_skill_family,
    )

    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--family", default="tour")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    table = build_table(
        get_skill_family(a.family), PipelineLandingSimulator(), n=a.n, seed=a.seed
    )
    table.to_json(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
