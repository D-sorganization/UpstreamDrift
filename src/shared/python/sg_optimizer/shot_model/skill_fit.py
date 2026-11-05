"""Player matching: fit a ``SkillFamily`` from shot data, shrunk toward a preset.

Input is a table ``{skill: (n, 6) array}`` in ``IMPACT_FIELDS`` order (``NaN`` =
not measured), read from a launch-monitor CSV (``read_shot_csv``) or built from
optimised model swings. Per skill the sample mean and covariance are blended
with the prior (preset) family by ``w = n / (n + PRIOR_STRENGTH)``:

    mean = w * sample_mean + (1 - w) * prior_mean
    cov  = w * sample_cov  + (1 - w) * prior_cov      (a convex mix of PSD is PSD)

so a handful of shots barely moves a preset, and a large sample dominates it.
Fields a monitor does not report (all ``NaN``) keep the prior's mean and sd; skills
with fewer than ``MIN_SHOTS`` complete shots keep the preset unchanged.
``PRIOR_STRENGTH`` is a pseudo-count *assumption* (like the presets), not a
measured quantity.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import require
from src.shared.python.sg_optimizer.shot_model.skill_family import (
    IMPACT_FIELDS,
    SKILL_NAMES,
    ImpactDistribution,
    Provenance,
    SkillFamily,
    SkillSpec,
)

PRIOR_STRENGTH = 10.0  # pseudo-shots worth of belief in the preset (assumption)
MIN_SHOTS = 2  # a covariance needs at least two complete shots
_N = len(IMPACT_FIELDS)
ShotTable = dict[str, NDArray[np.float64]]


def shrinkage_weight(n: int, prior_strength: float = PRIOR_STRENGTH) -> float:
    """Weight on the sample: 0 with no data, 0.5 at ``prior_strength`` shots."""
    require(n >= 0 and prior_strength > 0, "need n >= 0 and prior_strength > 0")
    return n / (n + prior_strength)


def club_to_skill(club: str) -> str | None:
    """Map a launch-monitor club label to a ``SKILL_NAMES`` entry (``None`` = unmapped)."""
    c = club.strip().lower()
    if c in ("driver", "dr", "1w"):
        return "driver"
    if "wood" in c or re.fullmatch(r"[2-9]w", c):
        return "fairway_wood"
    if "wedge" in c or c in ("pw", "gw", "aw", "sw", "lw"):
        return "wedge"
    if "hybrid" in c or re.fullmatch(r"[2-9]h", c):
        return "long_iron"
    m = re.fullmatch(r"([1-9])\s*-?\s*i(?:ron)?", c)
    if m:
        n = int(m.group(1))
        return "long_iron" if n <= 4 else "mid_iron" if n <= 7 else "short_iron"
    return None


def read_shot_csv(path: str | Path) -> ShotTable:
    """Read ``skill`` + ``IMPACT_FIELDS`` columns; ``#`` lines skipped, blanks = NaN."""
    p = Path(path)
    require(p.exists(), f"shot file not found: {p}")
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln[:1] != "#"]
    rows = list(csv.DictReader(lines))
    require(bool(rows) and "skill" in rows[0], "shot csv needs a 'skill' column")
    groups: dict[str, list[list[float]]] = {}
    for r in rows:
        skill = r["skill"].strip()
        require(skill in SKILL_NAMES, f"unknown skill {skill!r}; have {SKILL_NAMES}")
        vals = [(r.get(f) or "").strip() for f in IMPACT_FIELDS]
        groups.setdefault(skill, []).append([float(v) if v else np.nan for v in vals])
    return {k: np.array(v, dtype=float) for k, v in groups.items()}


def _fit_one(
    data: NDArray[np.float64], prior: ImpactDistribution, strength: float
) -> tuple[ImpactDistribution, int, float] | None:
    """Shrunk distribution, shot count and weight; ``None`` if too few shots."""
    require(data.ndim == 2 and data.shape[1] == _N, f"shots must be (n, {_N})")
    require(not np.isinf(data).any(), "shot values must be finite or NaN")
    seen = ~np.isnan(data).all(axis=0)
    rows = data[~np.isnan(data[:, seen]).any(axis=1)]
    if len(rows) < MIN_SHOTS:
        return None
    w = shrinkage_weight(len(rows), strength)
    p_cov = prior.covariance()
    s_cov = np.zeros((_N, _N))
    s_cov[np.ix_(~seen, ~seen)] = p_cov[np.ix_(~seen, ~seen)]
    s_cov[np.ix_(seen, seen)] = np.atleast_2d(np.cov(rows[:, seen], rowvar=False))
    s_mean = np.array(prior.mean)
    s_mean[seen] = rows[:, seen].mean(axis=0)
    mean = w * s_mean + (1 - w) * np.array(prior.mean)
    cov = w * s_cov + (1 - w) * p_cov
    sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    scale = np.outer(np.where(sd > 0, sd, 1.0), np.where(sd > 0, sd, 1.0))
    corr = np.clip(cov / scale, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    dist = ImpactDistribution(
        tuple(float(v) for v in mean),
        tuple(float(v) for v in sd),
        tuple(tuple(float(v) for v in r) for r in (corr + corr.T) / 2),
    )
    return dist, len(rows), w


def fit_skill_family(
    table: ShotTable,
    prior: SkillFamily,
    *,
    name: str,
    source: str = "shot data",
    prior_strength: float = PRIOR_STRENGTH,
) -> SkillFamily:
    """Fit a family from ``table``; unobserved or sparse skills keep ``prior``."""
    require(prior_strength > 0, "prior_strength must be > 0", prior_strength)
    unknown = set(table) - set(SKILL_NAMES)
    require(not unknown, f"unknown skills {sorted(unknown)}; have {SKILL_NAMES}")
    skills = dict(prior.skills)
    for skill, data in table.items():
        fit = _fit_one(
            np.asarray(data, dtype=float), _prior_dist(prior, skill), prior_strength
        )
        if fit is None:
            continue
        dist, n, w = fit
        note = f"fit: {source}, n={n}, w={w:.2f} toward {prior.name} (strength {prior_strength:g})"
        skills[skill] = SkillSpec(dist, Provenance("fit", note, "1"))
    note = f"fit from {source}, shrunk toward preset {prior.name!r}"
    return SkillFamily(name, skills, Provenance("fit", note, "1"))


def _prior_dist(prior: SkillFamily, skill: str) -> ImpactDistribution:
    require(skill in prior.skills, f"prior {prior.name!r} has no skill {skill!r}")
    return prior.skills[skill].distribution
