"""Unit tests for pair protocol preconditions in matched supplement figures.

Verifies that:
1. Missing or None source_protocol on either pair reference is rejected with ValueError
   containing 'pair references lack source protocol' before figure generation or data access,
   even when generic serialized fallback protocol_id is present.
2. The precondition holds for both canonical (_render_pair_figures) and selective
   (render_selected_pair_figures) rendering entry points.
3. No endpoints, files, or figure rendering helpers are accessed when the precondition fails.
4. Incomplete cell status (status != 'completed') is skipped cleanly (returns empty list)
   before protocol precondition checks.
5. Non-None invalid values (empty strings, whitespace, wrong types) continue to be handled
   by pair_comparison_identity validation.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from src.tools.shot_pattern_analysis import matched_supplement_figures as figures
from src.tools.shot_pattern_analysis.matched_supplement_figures import (
    CLOSED_SELECTED_FIGURE_KINDS,
    _render_pair_figures,
    render_selected_pair_figures,
)

pytestmark = pytest.mark.unit


def _create_synthetic_cell_ref(
    cell_dir: Path,
    *,
    club_id: str = "pitching_wedge",
    shaft_lean_deg: float = 0.0,
    curve_scale: float = 1.0,
    rotation_idx: int = 1,
    profile: str = "combined2",
    source_protocol: str | None = "original_primary/1",
    status: str = "completed",
    target_x_m: float = 180.0,
    n_shots: int = 10,
) -> dict[str, Any]:
    """Create synthetic cell directory and ref dictionary for tests."""
    cell_name = (
        f"{club_id}_lean{int(shaft_lean_deg)}_curve{int(curve_scale)}_"
        f"rotation{rotation_idx}_{profile}"
    )
    cell_path = cell_dir / cell_name
    cell_path.mkdir(parents=True, exist_ok=True)

    # Write shots.csv.gz
    shots_file = cell_path / "shots.csv.gz"
    lines = ["shot_index,pattern,status,carry_x_m,carry_y_m"]
    for i in range(n_shots):
        for pattern in ("Straight", "Draw", "Fade"):
            lines.append(f"{i},{pattern},supported,{target_x_m + i * 0.1},{i * 0.05}")
    with gzip.open(shots_file, "wb") as gz:
        gz.write("\n".join(lines).encode("utf-8"))

    # Write flight samples
    flight_data = {
        "trajectories": {
            pattern: [[[0.0, 0.0], [target_x_m, 0.0]]]
            for pattern in ("Straight", "Draw", "Fade")
        }
    }
    (cell_path / "flight_samples.json").write_text(json.dumps(flight_data))

    ref: dict[str, Any] = {
        "cell": cell_name,
        "source_root": str(cell_dir),
        "status": status,
        "config": {
            "club": {"club_id": club_id, "shaft_lean_deg": shaft_lean_deg},
            "curve_scale": curve_scale,
            "noise": {
                "profile": profile,
                "standard_deviations": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "correlation": None,
            },
            "n_shots": n_shots,
            "target_x_m": target_x_m,
        },
        "summary": {"target_x_m": target_x_m},
    }
    if source_protocol is not None:
        ref["source_protocol"] = source_protocol
    return ref


def _build_valid_pair(tmp_path: Path) -> dict[str, Any]:
    """Build a complete valid pair dictionary with synthetic directories."""
    first = _create_synthetic_cell_ref(
        tmp_path / "cell_a",
        shaft_lean_deg=0.0,
        curve_scale=1.0,
        source_protocol="original_primary/1",
    )
    second = _create_synthetic_cell_ref(
        tmp_path / "cell_b",
        shaft_lean_deg=5.0,
        curve_scale=2.0,
        rotation_idx=2,
        source_protocol="pw-control-face-bounds/1",
    )
    return {
        "comparison_kind": "curve_strength",
        "first": first,
        "second": second,
        "target_x_m": 180.0,
    }


@pytest.mark.parametrize(
    ("first_mod", "second_mod"),
    [
        # First missing source_protocol, has protocol_id fallback
        ({"source_protocol": None, "protocol_id": "fallback_a/1", "pop": True}, {}),
        # First source_protocol is None, has protocol_id fallback
        ({"source_protocol": None, "protocol_id": "fallback_a/1", "pop": False}, {}),
        # Second missing source_protocol, has protocol_id fallback
        ({}, {"source_protocol": None, "protocol_id": "fallback_b/1", "pop": True}),
        # Second source_protocol is None, has protocol_id fallback
        ({}, {"source_protocol": None, "protocol_id": "fallback_b/1", "pop": False}),
        # Both missing source_protocol, both have protocol_id fallback
        (
            {"source_protocol": None, "protocol_id": "fallback_a/1", "pop": True},
            {"source_protocol": None, "protocol_id": "fallback_b/1", "pop": True},
        ),
        # Both source_protocol are None, both have protocol_id fallback
        (
            {"source_protocol": None, "protocol_id": "fallback_a/1", "pop": False},
            {"source_protocol": None, "protocol_id": "fallback_b/1", "pop": False},
        ),
    ],
)
@pytest.mark.parametrize("renderer_kind", ["canonical", "selective"])
def test_pair_missing_or_none_source_protocol_rejected_with_value_error(
    tmp_path: Path,
    first_mod: dict[str, Any],
    second_mod: dict[str, Any],
    renderer_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing or None source_protocol must be rejected with ValueError.

    Even if generic protocol_id is present as fallback, actual renderer refs
    require source_protocol and must not fall through to KeyError or None caption.
    """
    monkeypatch.setattr(
        figures,
        "_save_figure",
        lambda fig, stem, formats=("png", "pdf", "svg"): {
            f: stem.with_suffix(f".{f}") for f in formats
        },
    )
    monkeypatch.setattr(
        figures,
        "write_pair_social_density",
        lambda *args, **kwargs: {},
    )
    pair = _build_valid_pair(tmp_path)
    for ref_key, mod in (("first", first_mod), ("second", second_mod)):
        if not mod:
            continue
        ref = dict(pair[ref_key])
        if mod.get("pop"):
            ref.pop("source_protocol", None)
        else:
            ref["source_protocol"] = None
        if "protocol_id" in mod:
            ref["protocol_id"] = mod["protocol_id"]
        pair[ref_key] = ref

    out_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="pair references lack source protocol"):
        if renderer_kind == "canonical":
            _render_pair_figures(pair, out_dir)
        else:
            render_selected_pair_figures(
                pair, out_dir, figure_kinds=CLOSED_SELECTED_FIGURE_KINDS
            )


@pytest.mark.parametrize("ref_key", ["first", "second"])
@pytest.mark.parametrize("missing", [True, False])
@pytest.mark.parametrize("renderer_kind", ["canonical", "selective"])
def test_no_endpoints_or_figures_accessed_when_protocol_precondition_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ref_key: str,
    missing: bool,
    renderer_kind: str,
) -> None:
    """Reject either malformed reference before any data, figure or directory access."""
    pair: dict[str, Any] = {
        "comparison_kind": "curve_strength",
        "target_x_m": 180.0,
        "first": {
            "status": "completed",
            "cell": "first",
            "source_protocol": "original_primary/1",
        },
        "second": {
            "status": "completed",
            "cell": "second",
            "source_protocol": "pw-primary-face-bounds/1",
        },
    }
    pair[ref_key]["protocol_id"] = "serialized-fallback/1"
    if missing:
        pair[ref_key].pop("source_protocol")
    else:
        pair[ref_key]["source_protocol"] = None
    accessed: list[str] = []

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        accessed.append("side effect")
        raise AssertionError(
            "Malformed renderer references must fail before side effects"
        )

    for name in (
        "_supported_endpoints",
        "_write_pair_density_figure",
        "_write_pair_overhead_figure",
        "write_pair_social_density",
    ):
        monkeypatch.setattr(figures, name, forbidden)
    from matplotlib.figure import Figure

    monkeypatch.setattr(Figure, "__init__", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    out_dir = tmp_path / "out"
    with pytest.raises(ValueError, match="pair references lack source protocol"):
        if renderer_kind == "canonical":
            _render_pair_figures(pair, out_dir)
        else:
            render_selected_pair_figures(
                pair, out_dir, figure_kinds=CLOSED_SELECTED_FIGURE_KINDS
            )
    assert accessed == []
    assert not out_dir.exists()


@pytest.mark.parametrize(
    ("first_status", "second_status"),
    [
        ("failed", "completed"),
        ("completed", "failed"),
        ("pending", "completed"),
        ("incomplete", "incomplete"),
        (None, "completed"),
    ],
)
def test_retained_incomplete_status_skip(
    tmp_path: Path, first_status: str | None, second_status: str | None
) -> None:
    """Incomplete cells must return [] without error, even if source_protocol is missing."""
    pair = _build_valid_pair(tmp_path)
    bad_first = dict(pair["first"])
    bad_first.pop("source_protocol", None)
    if first_status is not None:
        bad_first["status"] = first_status
    else:
        bad_first.pop("status", None)

    bad_second = dict(pair["second"])
    bad_second.pop("source_protocol", None)
    if second_status is not None:
        bad_second["status"] = second_status
    else:
        bad_second.pop("status", None)

    pair["first"] = bad_first
    pair["second"] = bad_second

    out_dir = tmp_path / "out"
    assert _render_pair_figures(pair, out_dir) == []
    assert (
        render_selected_pair_figures(
            pair, out_dir, figure_kinds=CLOSED_SELECTED_FIGURE_KINDS
        )
        == []
    )


@pytest.mark.parametrize(
    ("bad_protocol", "expected_exc", "match_str"),
    [
        ("", ValueError, "cannot be empty"),
        ("   ", ValueError, "whitespace"),
        (12345, TypeError, "must be a string"),
    ],
)
def test_identity_helper_validates_non_none_source_protocols(
    tmp_path: Path,
    bad_protocol: Any,
    expected_exc: type[Exception],
    match_str: str,
) -> None:
    """Non-None invalid values are forwarded to and validated by identity helper."""
    pair = _build_valid_pair(tmp_path)
    pair["first"]["source_protocol"] = bad_protocol
    out_dir = tmp_path / "out"

    with pytest.raises(expected_exc, match=match_str):
        _render_pair_figures(pair, out_dir)
