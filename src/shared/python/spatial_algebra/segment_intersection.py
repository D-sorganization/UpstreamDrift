"""Declarative segment intersection points for whole-body mobility (HCM-5, #14006)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from src.shared.python.core.handedness import subject_handedness
from src.shared.python.model_appearance.club_assembly import assembly_from_spec
from src.shared.python.model_appearance.grip_hands import grip_hand_frame
from src.shared.python.model_appearance.grip_pose import DEFAULT_GRIP_POSE, Hand

_GROUPS = (
    ("hands", (0.95, 0.75, 0.10, 0.80), ("lead_hand", "trail_hand")),
    ("wrists", (0.20, 0.80, 0.30, 0.80), ("lead_wrist", "trail_wrist")),
    ("elbows", (0.20, 0.60, 0.95, 0.80), ("lead_elbow", "trail_elbow")),
    ("shoulders", (0.95, 0.30, 0.30, 0.80), ("lead_shoulder", "trail_shoulder")),
    ("pelvis", (0.75, 0.30, 0.95, 0.80), ("pelvis",)),
    ("feet", (0.85, 0.50, 0.15, 0.80), ("lead_foot", "trail_foot")),
)
POINT_GROUPS: dict[str, tuple[str, ...]] = {g[0]: g[2] for g in _GROUPS}
GROUP_COLORS: dict[str, tuple[float, float, float, float]] = {
    g[0]: g[1] for g in _GROUPS
}


@dataclass(frozen=True)
class IntersectionPoint:
    name: str
    group: str
    body: str
    anchor_local: tuple[float, float, float]
    color: tuple[float, float, float, float]
    parent_body: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> IntersectionPoint:
        a, c = d["anchor_local"], d["color"]
        return cls(
            str(d["name"]),
            str(d["group"]),
            str(d["body"]),
            (float(a[0]), float(a[1]), float(a[2])),
            (float(c[0]), float(c[1]), float(c[2]), float(c[3])),
            d.get("parent_body"),
        )


def _pt(
    name: str, grp: str, body: str, anc: tuple[float, float, float]
) -> IntersectionPoint:
    return IntersectionPoint(
        name, grp, body, anc, GROUP_COLORS.get(grp, (0.7, 0.7, 0.7, 0.8))
    )


def _extract_anchor(f: Mapping[str, Any]) -> tuple[float, float, float]:
    p = f.get("placement")
    if p is not None and hasattr(p, "shape"):
        return (float(p[0, 3]), float(p[1, 3]), float(p[2, 3]))
    if p is not None and isinstance(p, (list, tuple)):
        return (float(p[0][3]), float(p[1][3]), float(p[2][3]))
    raw = f.get("anchor", f.get("offset", (0.0, 0.0, 0.0)))
    return (float(raw[0]), float(raw[1]), float(raw[2]))


def extract_intersection_points(
    spec: Mapping[str, Any],
) -> dict[str, IntersectionPoint]:
    """Extract declarative intersection points from specification (HCM-5, #14006)."""
    pts: dict[str, IntersectionPoint] = {}
    lead, trail = (
        ("trail", "lead") if subject_handedness(spec) == "left" else ("lead", "trail")
    )

    # 1. Hands from club assembly
    try:
        if (club := assembly_from_spec(spec)) is not None:
            cb = spec.get("closure", {}).get("follower", {}).get("body") or "club"
            for role in ("lead", "trail"):
                m = grip_hand_frame(club, Hand(role), DEFAULT_GRIP_POSE)
                pts[f"{role}_hand"] = _pt(
                    f"{role}_hand",
                    "hands",
                    cb,
                    (float(m[0, 3]), float(m[1, 3]), float(m[2, 3])),
                )
    except (KeyError, ValueError, TypeError):
        pass

    # 2. Frames in spec["frames"]
    raw = spec.get("frames", [])
    frames = (
        {k: v if isinstance(v, dict) else {"body": str(v)} for k, v in raw.items()}
        if isinstance(raw, dict)
        else {f["name"]: f for f in raw if isinstance(f, dict) and "name" in f}
        if isinstance(raw, list)
        else {}
    )
    for tag, side in (("W", "wrist"), ("E", "elbow"), ("S", "shoulder")):
        for prefix, role in (("L", lead), ("R", trail)):
            if (k := f"{prefix}{tag}") in frames:
                f, name, grp = frames[k], f"{role}_{side}", f"{side}s"
                pts[name] = _pt(name, grp, f.get("body", k), _extract_anchor(f))
    if "Hip" in frames:
        f = frames["Hip"]
        pts["pelvis"] = _pt(
            "pelvis", "pelvis", f.get("body", "Hip"), _extract_anchor(f)
        )

    # 3. Feet from contact parameters
    for p in spec.get("contact", {}).get("points", []):
        pn = p.get("name", "").lower()
        fk: str | None = (
            "lead_foot"
            if ("lead" in pn or "left" in pn)
            else "trail_foot"
            if ("trail" in pn or "right" in pn)
            else None
        )
        if fk and fk not in pts:
            pts[fk] = _pt(fk, "feet", p.get("body", fk), _extract_anchor(p))
    return pts


def resolve_intersection_points(
    spec: Mapping[str, Any],
    selector: str | Sequence[str] = "hands",
    *,
    max_points: int | None = None,
    max_condition: float | None = None,
) -> list[IntersectionPoint]:
    """Resolve intersection points from selector string or sequence (HCM-5, #14006)."""
    all_pts = extract_intersection_points(spec)
    seq = selector.split(",") if isinstance(selector, str) else selector
    tokens = [t.strip() for t in seq if str(t).strip()] or ["hands"]
    valid = set(all_pts) | set(POINT_GROUPS) | {"all"}
    names: list[str] = []
    for tok in tokens:
        if tok not in valid:
            raise ValueError(
                f"unknown mobility point {tok!r}; available: {sorted(all_pts)}"
            )
        items = list(all_pts) if tok == "all" else POINT_GROUPS.get(tok, (tok,))
        for name in items:
            if name in all_pts and name not in names:
                names.append(name)
    res = [all_pts[n] for n in names]
    return res[:max_points] if max_points and max_points > 0 else res
