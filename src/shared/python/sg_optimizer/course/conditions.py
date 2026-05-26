"""Course-conditions models — rough severity, tree penalization, green speed.

Conditions are **orthogonal to geometry**: the same hole raster can be played
under benign or punitive conditions and produce different strategies. Do not
encode "this is heavy rough" as a separate lie code (spec pitfall #14).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.shared.python.contracts import require


# ---------------------------------------------------------------------------
# Rough
# ---------------------------------------------------------------------------


_ROUGH_PRESETS: dict[str, float] = {
    "light": 0.20,
    "medium": 0.50,
    "heavy": 0.75,
    "us_open": 0.95,
}


@dataclass(frozen=True)
class RoughModel:
    """Severity ∈ [0, 1] modulates any shot played *from* a rough lie."""

    severity: float

    def __post_init__(self) -> None:
        require(
            0.0 <= self.severity <= 1.0, "severity must lie in [0,1]", self.severity
        )

    def distance_multiplier(self) -> float:
        r = self.severity
        return 1.0 - 0.08 * r - 0.12 * r * r

    def dispersion_multiplier(self) -> float:
        return 1.0 + 0.4 * self.severity

    def flyer_probability(self) -> float:
        # Peaks around medium rough (the 'between lies' regime).
        r = self.severity
        return 4.0 * r * (1.0 - r) * 0.25  # max 0.25 at r=0.5

    def spin_reduction(self) -> float:
        return 0.5 * self.severity

    @classmethod
    def preset(cls, name: str) -> RoughModel:
        require(name in _ROUGH_PRESETS, f"unknown rough preset {name!r}")
        return cls(severity=_ROUGH_PRESETS[name])


# ---------------------------------------------------------------------------
# Trees (Phase-1 stub: treats trees ≈ heavy rough; full model in Phase 2)
# ---------------------------------------------------------------------------


_TREE_PRESETS: dict[str, float] = {
    "decorative": 0.10,
    "typical": 0.50,
    "dense": 0.80,
    "jail": 1.00,
}


@dataclass(frozen=True)
class TreeModel:
    """Tree penalization ∈ [0, 1]. Phase 1: behaves like a scaled rough.

    Full recovery-distribution implementation lands in Phase 2 (#6271).
    """

    penalization: float

    def __post_init__(self) -> None:
        require(0.0 <= self.penalization <= 1.0, "penalization must lie in [0,1]")

    def is_forced_punch_out(self) -> bool:
        return self.penalization > 0.85

    def distance_multiplier(self) -> float:
        # Phase-1 stub: penalization shrinks effective advance distance.
        return max(0.05, 1.0 - 0.9 * self.penalization)

    def dispersion_multiplier(self) -> float:
        return 1.0 + 0.6 * self.penalization

    @classmethod
    def preset(cls, name: str) -> TreeModel:
        require(name in _TREE_PRESETS, f"unknown tree preset {name!r}")
        return cls(penalization=_TREE_PRESETS[name])


# ---------------------------------------------------------------------------
# Greens
# ---------------------------------------------------------------------------


_GREEN_PRESETS: dict[str, float] = {
    "slow": 9.0,
    "medium": 10.5,
    "fast": 12.0,
    "tournament": 13.0,
    "masters": 13.5,
}


@dataclass(frozen=True)
class GreenModel:
    """Stimpmeter ∈ [8, 14]. Affects putting + approach-shot holding."""

    stimp: float

    def __post_init__(self) -> None:
        require(8.0 <= self.stimp <= 14.0, "stimp must lie in [8,14]", self.stimp)

    def make_pct_modifier(self, distance_ft: float) -> float:
        alpha = 0.015
        g = min(distance_ft / 10.0, 3.0)
        return 1.0 - alpha * (self.stimp - 10.0) * g

    def leave_distribution_modifier(self, distance_ft: float) -> float:
        return 1.0 + 0.08 * max(0.0, self.stimp - 10.0)

    def effective_green_depth_multiplier(self) -> float:
        return 1.0 - 0.06 * max(0.0, self.stimp - 10.0)

    @classmethod
    def preset(cls, name: str) -> GreenModel:
        require(name in _GREEN_PRESETS, f"unknown green preset {name!r}")
        return cls(stimp=_GREEN_PRESETS[name])


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CourseConditions:
    """Composite condition object attached to a hole or round."""

    rough: RoughModel
    trees: TreeModel
    greens: GreenModel
    pin_position_difficulty: float = 0.5

    def __post_init__(self) -> None:
        require(
            0.0 <= self.pin_position_difficulty <= 1.0,
            "pin_position_difficulty must lie in [0,1]",
            self.pin_position_difficulty,
        )

    @classmethod
    def benign(cls) -> CourseConditions:
        return cls(
            RoughModel.preset("light"),
            TreeModel.preset("decorative"),
            GreenModel.preset("medium"),
        )

    @classmethod
    def tournament(cls) -> CourseConditions:
        return cls(
            RoughModel.preset("medium"),
            TreeModel.preset("typical"),
            GreenModel.preset("fast"),
        )

    @classmethod
    def major_championship(cls) -> CourseConditions:
        return cls(
            RoughModel.preset("us_open"),
            TreeModel.preset("dense"),
            GreenModel.preset("tournament"),
            pin_position_difficulty=0.85,
        )

    # --- I/O -------------------------------------------------------------
    def to_yaml(self, path: str | Path) -> None:
        payload: dict[str, Any] = {
            "rough": {"severity": self.rough.severity},
            "trees": {"penalization": self.trees.penalization},
            "greens": {"stimp": self.greens.stimp},
            "pin_position_difficulty": self.pin_position_difficulty,
        }
        Path(path).write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> CourseConditions:
        p = Path(path)
        require(p.exists(), f"conditions file not found: {p}")
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        rough = _build_rough(raw.get("rough", {}))
        trees = _build_trees(raw.get("trees", {}))
        greens = _build_greens(raw.get("greens", {}))
        return cls(
            rough=rough,
            trees=trees,
            greens=greens,
            pin_position_difficulty=float(raw.get("pin_position_difficulty", 0.5)),
        )


def _build_rough(spec: dict[str, Any]) -> RoughModel:
    if "preset" in spec:
        return RoughModel.preset(str(spec["preset"]))
    return RoughModel(severity=float(spec.get("severity", 0.5)))


def _build_trees(spec: dict[str, Any]) -> TreeModel:
    if "preset" in spec:
        return TreeModel.preset(str(spec["preset"]))
    return TreeModel(penalization=float(spec.get("penalization", 0.5)))


def _build_greens(spec: dict[str, Any]) -> GreenModel:
    if "preset" in spec:
        return GreenModel.preset(str(spec["preset"]))
    return GreenModel(stimp=float(spec.get("stimp", 10.5)))


# Silence unused-field lint under empty defaults — keeps dataclass hashable.
_ = field
