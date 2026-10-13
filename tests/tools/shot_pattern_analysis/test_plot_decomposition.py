"""Exact pixels and function budgets for saved shot-pattern plots."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path

import pytest

from src.tools.shot_pattern_analysis import overview, reporting, sensitivity
from src.tools.shot_pattern_analysis.core import AnalysisConfig, run_analysis
from src.tools.shot_pattern_analysis.physics import ShotOutcome

pytestmark = pytest.mark.unit


class _FakePhysics:
    def simulate(self, *, face_deg, path_deg, config, **kwargs):  # noqa: ANN001
        lateral = 12 * face_deg - 2 * path_deg
        return ShotOutcome(
            200.0,
            lateral,
            math.hypot(200.0, lateral),
            face_deg,
            2500.0,
            face_deg - path_deg,
            ((0.0, 0.0), (100.0, lateral / 2), (200.0, lateral)),
        )


def test_plot_writers_fit_architecture_budget() -> None:
    for writer in (
        overview._plot_scoring_forest,
        reporting._save_dispersion,
        sensitivity.render_sensitivity,
    ):
        assert len(inspect.getsourcelines(writer)[0]) <= 100


def _forest_rows() -> list[dict]:
    rows = []
    for mode in ("fixed_loft", "shaft_rotation"):
        for club in overview.CLUBS:
            for sd, scale in ((1.0, 1.0), (2.0, 1.0), (1.0, 2.0), (2.0, 2.0)):
                for pattern, sign in (("Draw", 1), ("Fade", -1)):
                    estimate = sign * sd * scale * 0.015
                    rows.append(
                        {
                            "delivery_mode": mode,
                            "club_id": club,
                            "face_sd_deg": sd,
                            "curve_scale": scale,
                            "pattern": pattern,
                            "paired_delta_sg_vs_straight": estimate,
                            "paired_delta_sg_lower_95": estimate - 0.02,
                            "paired_delta_sg_upper_95": estimate + 0.02,
                        }
                    )
    return rows


@pytest.mark.slow
def test_synthetic_plot_pixels_match_pre_refactor(tmp_path: Path) -> None:
    from PIL import Image

    rows = _forest_rows()
    for mode in ("fixed_loft", "shaft_rotation"):
        overview._plot_scoring_forest(
            rows, delivery_mode=mode, output=tmp_path / f"{mode}_forest.png"
        )
    result = run_analysis(AnalysisConfig(n_shots=3, seed=71), physics=_FakePhysics())
    reporting._save_dispersion(result, tmp_path / "dispersion.png")
    reporting._save_dispersion(
        result, tmp_path / "dispersion_equal_range.png", equal_range=True
    )
    controls = sensitivity.build_sensitivity(
        sensitivity.preset_configs(), engine=_FakePhysics()
    )
    sensitivity.render_sensitivity(controls, tmp_path / "sensitivity.png")
    reference = json.loads(
        (Path(__file__).parent / "fixtures/plot_decomposition_rgba.json").read_text()
    )["outputs"]
    observed = {
        path.name: {
            "size": list(Image.open(path).size),
            "rgba_sha256": hashlib.sha256(
                Image.open(path).convert("RGBA").tobytes()
            ).hexdigest(),
        }
        for path in sorted(tmp_path.glob("*.png"))
    }
    assert observed == reference
