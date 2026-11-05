"""Preset skill families (tour, scratch, 10-hcp, 20-hcp) and a name registry.

Presets live in ``data/sg_optimizer/skill_families/presets.yaml``: one explicit
``tour`` table plus levels *derived* from it by a speed scale and a dispersion
(sd) scale. Every number there is marked as an assumption until B7 (#14571)
replaces it with a fit; none is presented as measured here.
"""

from __future__ import annotations

import math
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src.shared.python.contracts import require
from src.shared.python.sg_optimizer.shot_model.skill_family import (
    IMPACT_FIELDS,
    ImpactDistribution,
    Provenance,
    SkillFamily,
    SkillSpec,
    check_units,
)

PRESETS_PATH = (
    Path(__file__).resolve().parents[5]
    / "data"
    / "sg_optimizer"
    / "skill_families"
    / "presets.yaml"
)
_CUSTOM: dict[str, SkillFamily] = {}


def _corr_matrix(pairs: dict[str, float]) -> tuple[tuple[float, ...], ...]:
    n = len(IMPACT_FIELDS)
    m = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for key, rho in pairs.items():
        a, b = (IMPACT_FIELDS.index(k.strip()) for k in key.split(","))
        m[a][b] = m[b][a] = float(rho)
    return tuple(tuple(r) for r in m)


def derive_family(base: SkillFamily, name: str, spec: dict[str, Any]) -> SkillFamily:
    """Scale a family's speed and sd; the result is a ``preset`` with its source."""
    prov = Provenance("preset", str(spec["source"]), str(spec["version"]))
    speed, disp = float(spec["speed_scale"]), float(spec["dispersion_scale"])
    require(
        math.isfinite(speed) and math.isfinite(disp) and speed > 0 and disp > 0,
        "speed_scale and dispersion_scale must be finite and > 0",
    )
    skills = {}
    for k, s in base.skills.items():
        d = s.distribution
        mean = (d.mean[0] * speed, *d.mean[1:])
        sd = tuple(v * disp for v in d.sd)
        skills[k] = SkillSpec(ImpactDistribution(mean, sd, d.corr), prov)
    return SkillFamily(name, skills, prov)


@lru_cache(maxsize=1)
def _presets() -> dict[str, SkillFamily]:
    require(PRESETS_PATH.exists(), f"preset file not found: {PRESETS_PATH}")
    raw = yaml.safe_load(PRESETS_PATH.read_text(encoding="utf-8"))
    check_units(raw)
    corr = _corr_matrix(raw.get("correlation", {}))
    base_raw = raw["tour"]
    prov = Provenance("preset", str(base_raw["source"]), str(base_raw["version"]))
    skills = {
        k: SkillSpec(
            ImpactDistribution(
                tuple(float(v) for v in s[0]),  # [mean, sd] pair
                tuple(float(v) for v in s[1]),
                corr,
            ),
            prov,
        )
        for k, s in base_raw["skills"].items()
    }
    out = {"tour": SkillFamily("tour", skills, prov)}
    for name, spec in raw["derived"].items():
        out[name] = derive_family(out["tour"], name, spec)
    return out


def available_presets() -> tuple[str, ...]:
    return tuple(sorted({*_presets(), *_CUSTOM}))


def register_skill_family(family: SkillFamily) -> None:
    require(family.name not in available_presets(), f"{family.name!r} already exists")
    _CUSTOM[family.name] = family


def get_skill_family(name: str) -> SkillFamily:
    """Look a family up by name; unknown names fail closed."""
    fams = {**_presets(), **_CUSTOM}
    require(name in fams, f"unknown skill family {name!r}; have {sorted(fams)}")
    return replace(fams[name])
