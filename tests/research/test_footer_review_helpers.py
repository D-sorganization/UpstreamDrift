"""Synthetic integration checks for the archived footer-review renderer."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from docs.research.shot_pattern_analysis.phase2.review_followups import (
    footer_review_helpers as helpers,
)
from src.tools.shot_pattern_analysis.matched_pair_social import (
    write_pair_social_density,
)
from src.tools.shot_pattern_analysis.matched_pair_social import PairSocialContext




@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.timeout(240)
def test_render_club_uses_typed_pair_context_without_changing_shared_points(
    tmp_path, monkeypatch
):
    data = tmp_path / "saved-data"
    output = tmp_path / "output"
    output.mkdir()
    captured: dict[str, object] = {}
    native_writer = write_pair_social_density

    def load_cell(directory, expected_config, expected_execution):
        rotation = 1 if "rotation1" in directory.name else 2
        return {
            "config": {"noise": {"rotation": rotation}},
            "target_x_m": 180.0,
            "rotation": rotation,
        }

    def endpoints(path, total_n):
        del path
        assert total_n == 10000
        rows = [(2, 3.0, 4.0), (5, 6.0, 7.0)]
        return (dict.fromkeys(("Straight", "Draw", "Fade"), rows), {})

    def write_pairs(points, stem, *, context):
        assert isinstance(context, PairSocialContext)
        captured["points"] = points
        captured["context"] = context
        return native_writer(points, stem, context=context)

    monkeypatch.setattr(helpers, "load_validated_cell", load_cell)
    monkeypatch.setattr(helpers.figures, "_supported_endpoints", endpoints)
    monkeypatch.setattr(
        helpers.figures,
        "_pair_condition_label",
        lambda comparison, ref: f"Rotation {ref['summary']['rotation']}°",
    )
    monkeypatch.setattr(
        helpers.figures,
        "_pair_social_scenario",
        lambda comparison, *refs: "fixed scenario",
    )
    monkeypatch.setattr(
        helpers.figures,
        "_noise_profile_description",
        lambda profile, noise: "serialized noise",
    )
    monkeypatch.setattr(helpers, "write_pair_social_density", write_pairs)

    def hash_file(path: Path) -> str:
        payload = path.read_bytes() if path.exists() else path.name.encode()
        return hashlib.sha256(payload).hexdigest()

    inputs: dict[str, dict[str, str]] = {}
    artifacts: dict[str, str] = {}
    previews: dict[str, str] = {}

    helpers.render_club(
        "seven_iron",
        data,
        output,
        {
            "cells": {
                f"seven_iron_lean0_curve1_rotation{rotation}_combined2": {}
                for rotation in (1, 2)
            },
            "execution": {},
        },
        inputs,
        artifacts,
        previews,
        hash_file,
    )

    context = captured["context"]
    assert isinstance(context, PairSocialContext)
    assert context.condition_labels == ("Rotation 1°", "Rotation 2°")
    assert context.scenario == "fixed scenario"
    assert context.social_club == "Seven Iron"
    assert context.target_point_m == (180.0, 0)
    assert context.total_n == 10000
    assert context.delivery_noise == "serialized noise"
    assert captured["points"] == {
        f"{pattern} · {label}": [(3.0, 4.0), (6.0, 7.0)]
        for label in context.condition_labels
        for pattern in ("Straight", "Draw", "Fade")
    }
    assert len(inputs) == 2
    assert len(artifacts) == 9
    assert len(previews) == 3
    assert all((output / path).is_file() for path in artifacts)
    assert all((output / path).is_file() for path in previews)


@pytest.mark.unit
def test_footer_geometry_helper_reports_clearance_and_overlap(tmp_path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(8, 5))
    FigureCanvasAgg(figure)
    for index in range(7):
        bottom = 0.02 if index == 0 else 0.4
        figure.add_axes([0.1 + index * 0.01, bottom, 0.2, 0.3])
    figure.text(0.2, 0.12, "review footer", fontsize=10)

    report = helpers.inspect_figure(figure, tmp_path / "plot.png")

    assert report["path"] == str(tmp_path / "plot.png")
    assert "footer overlaps decorated axes" in report["issues"]
    assert report["minimum_font_em_px_at_1080"] > 0
    assert isinstance(report["minimum_footer_clearance_pt"], float)
