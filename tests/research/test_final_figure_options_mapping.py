"""Issue #12309: final-figure keyword-to-FigureOptions mapping is exact.

The final builder historically passed ``presentation=`` and ``dpi=`` to the
academic figure writers. Those writers now take ``options=FigureOptions(...)``.
These tests pin three things:

* the builder hands every writer exactly ``FigureOptions(presentation, dpi)``
  and never a legacy keyword (all three call sites, via test doubles);
* both public writers still accept the frozen keyword surface, and it renders
  decoded-pixel and text-artist identical figures to the options surface;
* mixing the two surfaces fails fast.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.unit

_PATTERNS = ("Straight", "Draw", "Fade")


def _plots() -> Any:
    try:
        import matplotlib  # noqa: F401
    except (ImportError, OSError) as exc:  # pragma: no cover - optional extra
        pytest.skip(f"matplotlib unavailable: {exc}")
    from src.tools.shot_pattern_analysis import academic_plots

    return academic_plots


def _metadata() -> dict[str, Any]:
    return {
        "social_club": "Driver",
        "protocol": "Original Matched-Delivery Protocol",
        "noise": "Shaft Twist SD 2° • Path SD 0.5° • Speed SD 0.9 m/s",
        "valid_shots_by_pattern": dict.fromkeys(_PATTERNS, 12),
        "total_shots_by_pattern": dict.fromkeys(_PATTERNS, 12),
    }


def _endpoints() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(12309)
    return {
        name: np.column_stack(
            (180.0 + rng.normal(0.0, 3.0, 12), offset + rng.normal(0.0, 1.5, 12))
        )
        for name, offset in zip(_PATTERNS, (0.0, -4.0, 4.0), strict=True)
    }


def _trajectories() -> dict[str, list[np.ndarray]]:
    return {
        name: [np.array([[0.0, 0.0], [90.0, 0.5 * end[1]], end]) for end in points[:4]]
        for name, points in _endpoints().items()
    }


class _Rendered:
    """Decoded raster plus the visible text artists of one exported figure."""

    def __init__(self, figure: Any, dpi: int) -> None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        figure.set_dpi(dpi)
        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        self.dpi = dpi
        self.rgba = np.asarray(canvas.buffer_rgba()).copy()
        self.texts = sorted(
            text.get_text()
            for artist in (figure, *figure.axes)
            for text in artist.texts
            if text.get_visible()
        )
        self.size_inches = tuple(figure.get_size_inches())


@pytest.fixture
def rendered(monkeypatch: pytest.MonkeyPatch) -> list[_Rendered]:
    """Replace file export with an in-memory decode of the finished figure."""
    plots = _plots()
    captured: list[_Rendered] = []

    def decode(figure: Any, output_stem: Path, *, dpi: int) -> dict[str, Path]:
        captured.append(_Rendered(figure, dpi))
        return {"png": output_stem.with_suffix(".png")}

    monkeypatch.setattr(plots, "_save_formats", decode)
    return captured


def _assert_identical(first: _Rendered, second: _Rendered) -> None:
    assert first.dpi == second.dpi
    assert first.size_inches == second.size_inches
    assert first.texts == second.texts
    assert first.rgba.shape == second.rgba.shape
    assert np.array_equal(first.rgba, second.rgba)


@pytest.mark.parametrize("presentation", ["academic", "social"])
def test_overhead_writer_legacy_keywords_render_identically_to_options(
    tmp_path: Path, rendered: list[_Rendered], presentation: str
) -> None:
    plots = _plots()
    common = {
        "target_x_m": 180.0,
        "scenario": "Synthetic Mapping Control",
        "metadata": _metadata(),
    }
    plots.write_overhead_trajectory_figure(
        _trajectories(),
        tmp_path / "legacy",
        presentation=presentation,
        dpi=72,
        **common,
    )
    plots.write_overhead_trajectory_figure(
        _trajectories(),
        tmp_path / "options",
        options=plots.FigureOptions(presentation=presentation, dpi=72),
        **common,
    )
    assert len(rendered) == 2
    _assert_identical(*rendered)


@pytest.mark.parametrize("presentation", ["academic", "social"])
def test_density_writer_legacy_keywords_render_identically_to_options(
    tmp_path: Path, rendered: list[_Rendered], presentation: str
) -> None:
    plots = _plots()
    common = {
        "scenario": "Synthetic Mapping Control",
        "metadata": _metadata(),
        "target_point_m": (180.0, 0.0),
    }
    plots.write_academic_dispersion_figure(
        _endpoints(), tmp_path / "legacy", presentation=presentation, dpi=72, **common
    )
    plots.write_academic_dispersion_figure(
        _endpoints(),
        tmp_path / "options",
        options=plots.FigureOptions(presentation=presentation, dpi=72),
        **common,
    )
    assert len(rendered) == 2
    _assert_identical(*rendered)


def test_overhead_writer_rejects_mixed_options_and_legacy_keywords(
    tmp_path: Path,
) -> None:
    plots = _plots()
    with pytest.raises(ValueError, match="either options or legacy"):
        plots.write_overhead_trajectory_figure(
            _trajectories(),
            tmp_path / "mixed",
            target_x_m=180.0,
            scenario="Synthetic Mapping Control",
            metadata=_metadata(),
            options=plots.FigureOptions(),
            dpi=72,
        )


@pytest.mark.parametrize("presentation", ["academic", "social"])
def test_builder_option_mapping_is_exact(presentation: str) -> None:
    from scripts.research.shot_pattern_analysis import build_final_figures as builder

    plots = _plots()
    mapped = builder._figure_options(presentation, 137)
    assert mapped == plots.FigureOptions(presentation=presentation, dpi=137)
    defaults = plots.FigureOptions()
    assert (mapped.bins, tuple(mapped.region_masses)) == (
        defaults.bins,
        tuple(defaults.region_masses),
    )


def test_builder_passes_only_figure_options_at_all_three_call_sites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.research.shot_pattern_analysis import build_final_figures as builder

    plots = _plots()
    calls: list[tuple[str, dict[str, Any]]] = []

    def double(name: str):
        def write(_data: Any, stem: Path, **kwargs: Any) -> dict[str, Path]:
            calls.append((name, kwargs))
            return {"png": Path(stem).with_suffix(".png")}

        return write

    monkeypatch.setattr(plots, "write_academic_dispersion_figure", double("density"))
    monkeypatch.setattr(plots, "write_overhead_trajectory_figure", double("overhead"))
    endpoints = {name: points.tolist() for name, points in _endpoints().items()}
    monkeypatch.setattr(
        builder,
        "_supported_points",
        lambda _path: (endpoints, dict.fromkeys(_PATTERNS, 12)),
    )
    cells = [
        {
            "cell": f"driver_sd{sd:g}",
            "status": "completed",
            "target_x_m": 180.0,
            "config": {
                "club": {"club_id": "driver", "shaft_lean_deg": 0.0},
                "noise": {"standard_deviations": [sd, 0, 0, 0, 0, 0, 0]},
                "curve_scale": 1.0,
                "n_shots": 12,
                "seed": 4,
            },
            "nominals": {},
        }
        for sd in (1.0, 2.0)
    ]
    source = tmp_path / "source"
    (source / cells[0]["cell"]).mkdir(parents=True)
    (source / cells[0]["cell"] / "flight_samples.json").write_text(
        json.dumps(
            {
                "target_x_m": 180.0,
                "trajectories": {
                    name: [[[0.0, 0.0], [180.0, 0.0]]] for name in _PATTERNS
                },
            }
        ),
        encoding="utf-8",
    )
    for presentation in ("academic", "social"):
        builder._render_cell_figures(
            cells[0],
            source,
            tmp_path / "out",
            dpi=91,
            preview=False,
            presentation=presentation,
        )
    pair = next(
        item
        for item in builder._comparison_pair_specs(cells)
        if item["comparison"] == "shaft_rotation_sd" and item["pattern"] == "Draw"
    )
    row, _paths = builder._render_density_comparison(
        pair, source, tmp_path / "out", dpi=91, preview=False
    )
    assert row["status"] == "completed"
    assert [(name, kwargs["options"]) for name, kwargs in calls] == [
        ("density", plots.FigureOptions(presentation="academic", dpi=91)),
        ("overhead", plots.FigureOptions(presentation="academic", dpi=91)),
        ("density", plots.FigureOptions(presentation="social", dpi=91)),
        ("overhead", plots.FigureOptions(presentation="social", dpi=91)),
        ("density", plots.FigureOptions(presentation="social", dpi=91)),
    ]
    legacy = {"presentation", "dpi", "bins", "region_masses"}
    assert all(not legacy & kwargs.keys() for _name, kwargs in calls)
