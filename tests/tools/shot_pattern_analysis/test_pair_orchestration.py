"""Pair export stages preserve common-index populations and format order."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_pair_common_points_intersect_all_six_populations(monkeypatch):
    from src.tools.shot_pattern_analysis import matched_supplement_figures as figures

    maps = [
        {
            name: [(2, 20.0, 2.0), (0, 0.0, 0.0), (1, 10.0, 1.0)]
            for name in figures._PATTERNS
        },
        {name: [(2, 22.0, 2.0), (0, 2.0, 0.0)] for name in figures._PATTERNS},
    ]
    maps[1]["Fade"] = [(0, 2.0, 0.0)]
    seen = []

    def load(path, total):
        seen.append((path, total))
        return maps[len(seen) - 1], {}

    monkeypatch.setattr(figures, "_supported_endpoints", load)
    first = {"source_root": "/synthetic", "cell": "a", "config": {"n_shots": 3}}
    second = {"source_root": "/synthetic", "cell": "b", "config": {"n_shots": 3}}
    context = figures.PairFigureContext(first, second, "across_lean", ("A", "B"), 125.0)
    points, common_n = figures._load_pair_common_points(context)
    assert common_n == 1
    assert points == {
        f"{p} · {label}": [xy]
        for label, xy in [("A", (0.0, 0.0)), ("B", (2.0, 0.0))]
        for p in figures._PATTERNS
    }
    assert seen == [
        (Path("/synthetic/a/shots.csv.gz"), 3),
        (Path("/synthetic/b/shots.csv.gz"), 3),
    ]


@pytest.mark.parametrize(
    "kinds,expected",
    [
        (None, ("png", "pdf", "svg")),
        (frozenset({"density_svg", "density_png", "overhead_pdf"}), ("png", "svg")),
        (frozenset({"overhead_pdf"}), ()),
    ],
)
def test_pair_export_formats_keep_canonical_order(kinds, expected):
    from src.tools.shot_pattern_analysis.matched_supplement_figures import (
        _pair_export_formats,
    )

    assert _pair_export_formats("density", kinds) == expected
