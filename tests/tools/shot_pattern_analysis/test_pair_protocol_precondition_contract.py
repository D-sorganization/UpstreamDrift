"""Renderer protocol precondition holds before any I/O or figure work (#12169).

The regression: a completed pair reference missing ``source_protocol`` but carrying
the serialized ``protocol_id`` passed the general identity adapter, then raised
``KeyError: source_protocol`` while building the academic density caption. The
identity adapter must keep accepting ``protocol_id`` for serialized comparison
records, while both renderer entry points reject such references with
``ValueError`` before reading sources, creating directories or allocating figures,
for every selectable figure kind.
"""

from __future__ import annotations

import builtins
import gzip
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

from src.tools.shot_pattern_analysis import matched_supplement_figures as figures
from src.tools.shot_pattern_analysis.matched_supplement_figures import (
    CLOSED_SELECTED_FIGURE_KINDS,
    _render_pair_figures,
    pair_comparison_identity,
    render_selected_pair_figures,
)

pytestmark = pytest.mark.unit

_MESSAGE = "pair references lack source protocol"


def _reference(cell: str, protocol: str, source_root: Path) -> dict[str, Any]:
    return {
        "status": "completed",
        "cell": cell,
        "source_protocol": protocol,
        "source_root": str(source_root),
        "config": {"n_shots": 10},
        "summary": {"target_x_m": 180.0},
    }


def _protocol_less_pair(
    tmp_path: Path, position: str, *, missing: bool
) -> dict[str, Any]:
    """Build the #12169 shape: one reference keeps only the serialized protocol_id."""
    pair: dict[str, Any] = {
        "comparison_kind": "curve_strength",
        "target_x_m": 180.0,
        "first": _reference(
            "pitching_wedge_lean0_curve2_rotation1_combined2",
            "original_primary/1",
            tmp_path / "sources",
        ),
        "second": _reference(
            "pitching_wedge_lean5_curve2_rotation1_combined2",
            "pw-control-face-bounds/1",
            tmp_path / "sources",
        ),
    }
    reference = pair[position]
    reference["protocol_id"] = reference["source_protocol"]
    if missing:
        del reference["source_protocol"]
    else:
        reference["source_protocol"] = None
    return pair


def _forbid_side_effects(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    touched: list[str] = []

    def forbid(name: str) -> Callable[..., Any]:
        def guard(*args: Any, **kwargs: Any) -> Any:
            touched.append(name)
            raise AssertionError(f"{name} reached before protocol precondition")

        return guard

    for name in (
        "_supported_endpoints",
        "_load_pair_common_points",
        "_load_pair_trajectories",
        "_write_pair_density_figure",
        "_write_pair_overhead_figure",
        "write_pair_social_density",
        "sha256_file",
    ):
        monkeypatch.setattr(figures, name, forbid(name))
    monkeypatch.setattr(Figure, "__init__", forbid("Figure"))
    monkeypatch.setattr(plt, "figure", forbid("pyplot.figure"))
    monkeypatch.setattr(plt, "subplots", forbid("pyplot.subplots"))
    for attribute in ("mkdir", "open", "read_text", "read_bytes"):
        monkeypatch.setattr(Path, attribute, forbid(f"Path.{attribute}"))
    monkeypatch.setattr(gzip, "open", forbid("gzip.open"))
    monkeypatch.setattr(builtins, "open", forbid("open"))
    return touched


def test_identity_adapter_still_accepts_serialized_protocol_id(tmp_path: Path) -> None:
    pair = _protocol_less_pair(tmp_path, "first", missing=True)
    identity = pair_comparison_identity(pair)
    assert identity[1] == "original_primary/1"
    assert identity[3] == "pw-control-face-bounds/1"


@pytest.mark.parametrize("position", ["first", "second"])
@pytest.mark.parametrize("missing", [True, False], ids=["missing", "none"])
@pytest.mark.parametrize(
    "figure_kinds",
    [None, *(frozenset({kind}) for kind in sorted(CLOSED_SELECTED_FIGURE_KINDS))],
    ids=lambda kinds: "canonical" if kinds is None else "-".join(sorted(kinds)),
)
def test_renderer_rejects_protocol_less_reference_before_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    position: str,
    missing: bool,
    figure_kinds: frozenset[str] | None,
) -> None:
    pair = _protocol_less_pair(tmp_path, position, missing=missing)
    output_dir = tmp_path / "out"
    open_figures = plt.get_fignums()
    touched = _forbid_side_effects(monkeypatch)
    with pytest.raises(ValueError, match=_MESSAGE):
        if figure_kinds is None:
            _render_pair_figures(pair, output_dir)
        else:
            render_selected_pair_figures(pair, output_dir, figure_kinds=figure_kinds)
    monkeypatch.undo()
    assert touched == []
    assert plt.get_fignums() == open_figures
    assert not output_dir.exists()
