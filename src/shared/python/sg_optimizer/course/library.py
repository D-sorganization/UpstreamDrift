"""Classic holes library — load famous par-3/4/5 GeoJSON specs by slug.

Slugs are stable identifiers so callers can hard-code them in tests and
scripts. The backing ``.geojson`` files live at::

    data/sg_optimizer/courses/classics/{slug}.geojson

Phase 2 (#6271).
"""

from __future__ import annotations

from pathlib import Path

from src.shared.python.contracts import require
from src.shared.python.sg_optimizer.course.course_io import (
    HoleGeometry,
    load_hole_geojson,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Stable slug → human-readable description mapping.
_CLASSIC_REGISTRY: dict[str, str] = {
    "sawgrass_17": "TPC Sawgrass Hole 17 — island-green par 3 (137 yd)",
    "augusta_13": "Augusta National Hole 13 — Azalea par 5 (510 yd)",
    "pebble_7": "Pebble Beach Hole 7 — clifftop par 3 (106 yd)",
    "road_hole_17": "St Andrews Old Course Hole 17 — Road Hole par 4 (455 yd)",
    "cypress_16": "Cypress Point Hole 16 — clifftop par 3 over ocean (231 yd)",
}

# Resolve data path relative to the repo root (two ``parents`` above this
# file's package root ``src/``).
_REPO_ROOT = Path(__file__).resolve().parents[5]
_CLASSICS_DIR = _REPO_ROOT / "data" / "sg_optimizer" / "courses" / "classics"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_classics() -> list[str]:
    """Return the sorted list of available classic-hole slugs.

    The slugs are stable identifiers; they will not change between releases
    without a deprecation notice.
    """
    return sorted(_CLASSIC_REGISTRY.keys())


def load_classic(slug: str) -> HoleGeometry:
    """Load a classic hole by slug.

    Parameters
    ----------
    slug :
        One of the slugs returned by :func:`list_classics`.

    Returns
    -------
    HoleGeometry
        The parsed hole geometry.

    Raises
    ------
    ValueError
        If ``slug`` is not a recognised classic slug.
    FileNotFoundError
        If the backing ``.geojson`` file is missing from the data directory.
    """
    require(
        slug in _CLASSIC_REGISTRY,
        f"unknown classic slug {slug!r}; available: {sorted(_CLASSIC_REGISTRY)}",
        slug,
    )

    path = _CLASSICS_DIR / f"{slug}.geojson"
    if not path.exists():
        raise FileNotFoundError(
            f"Classic hole GeoJSON not found: {path}\n"
            f"Expected at data/sg_optimizer/courses/classics/{slug}.geojson"
        )

    return load_hole_geojson(path)
