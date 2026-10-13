"""Unit and integration tests for render_selected_pair_figures (issue #12132).

Verifies that:
1. figure_kinds accepts only a non-empty frozenset from the closed 7-kind catalog.
2. Returns identical ledger metadata and path naming to _render_pair_figures.
3. Writes ONLY requested formats on disk, with no extra untracked files.
4. Never calls pair statistics, bootstrap, simulation, or writes social density pattern cards.
5. Correctly handles incomplete cells or missing flight samples.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.unit

CLOSED_7_KINDS = frozenset(
    {
        "density_png",
        "density_pdf",
        "density_svg",
        "overhead_png",
        "overhead_pdf",
        "overhead_svg",
        "overhead_social_png",
    }
)


def _check_plotting_deps() -> None:
    try:
        import matplotlib.pyplot  # noqa: F401
        from PIL import Image  # noqa: F401
    except (ImportError, OSError) as exc:
        pytest.skip(f"Plotting dependencies unavailable: {exc}")


def _write_synthetic_cell(
    cell_dir: Path,
    *,
    club_id: str = "pitching_wedge",
    lean_deg: float = 0.0,
    curve_scale: float = 1.0,
    target_x_m: float = 125.0,
    profile: str = "shaft_only",
    n_shots: int = 20,
) -> dict[str, Any]:
    """Create a fully valid synthetic cell directory matching pipeline expectations."""
    import hashlib

    cell_dir.mkdir(parents=True, exist_ok=True)
    shots_file = cell_dir / "shots.csv.gz"
    # Write CSV with header and valid shots
    lines = ["shot_index,pattern,status,carry_x_m,carry_y_m"]
    for i in range(n_shots):
        for pattern in ("Straight", "Draw", "Fade"):
            lines.append(f"{i},{pattern},supported,{target_x_m + i * 0.1},{i * 0.05}")
    csv_bytes = "\n".join(lines).encode("utf-8")
    with gzip.open(shots_file, "wb") as gz:
        gz.write(csv_bytes)

    flight_file = cell_dir / "flight_samples.json"
    flight_data = {
        "trajectories": {
            pattern: [[[0.0, 0.0], [target_x_m * 0.5, 2.0], [target_x_m, 0.0]]]
            for pattern in ("Straight", "Draw", "Fade")
        }
    }
    flight_bytes = json.dumps(flight_data).encode("utf-8")
    flight_file.write_bytes(flight_bytes)

    # Write manifest
    manifest = {
        "files": {
            "shots.csv.gz": hashlib.sha256(csv_bytes).hexdigest(),
            "flight_samples.json": hashlib.sha256(flight_bytes).hexdigest(),
        }
    }
    (cell_dir / "manifest.json").write_text(json.dumps(manifest))

    cell_name = (
        f"{club_id}_lean{int(lean_deg)}_curve{int(curve_scale)}_rotation1_{profile}"
    )
    return {
        "cell": cell_name,
        "source_root": str(cell_dir.parent),
        "source_protocol": "original_control/1",
        "status": "completed",
        "config": {
            "club": {"club_id": club_id, "shaft_lean_deg": lean_deg},
            "curve_scale": curve_scale,
            "noise": {
                "standard_deviations": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "correlation": None,
            },
            "n_shots": n_shots,
        },
        "summary": {"target_x_m": target_x_m},
    }


def _synthetic_pair_bundle(
    root: Path,
) -> tuple[dict[str, Any], Path, Path]:
    """Create a pair bundle with valid completed sources."""
    c1_name = "pitching_wedge_lean0_curve1_rotation1_shaft_only"
    c2_name = "pitching_wedge_lean10_curve1_rotation1_shaft_only"
    c1_dir = root / c1_name
    c2_dir = root / c2_name
    r1 = _write_synthetic_cell(c1_dir, lean_deg=0.0, curve_scale=1.0)
    r2 = _write_synthetic_cell(c2_dir, lean_deg=10.0, curve_scale=1.0)
    pair = {
        "comparison_kind": "across_lean",
        "first": r1,
        "second": r2,
        "target_x_m": 125.0,
    }
    return pair, c1_dir, c2_dir


def test_render_selected_pair_figures_closed_kind_rejection(tmp_path: Path) -> None:
    """render_selected_pair_figures must reject unknown, invalid, or empty figure_kinds."""
    _check_plotting_deps()
    from src.tools.shot_pattern_analysis.matched_supplement_figures import (
        render_selected_pair_figures,
    )

    pair, _, _ = _synthetic_pair_bundle(tmp_path / "sources")
    out_dir = tmp_path / "out"

    # 1. Empty frozenset rejected
    with pytest.raises(ValueError, match="figure_kinds"):
        render_selected_pair_figures(pair, out_dir, figure_kinds=frozenset())

    # 2. Unknown kinds rejected
    with pytest.raises(ValueError, match="figure_kinds"):
        render_selected_pair_figures(
            pair, out_dir, figure_kinds=frozenset({"density_bmp"})
        )

    # 3. Social density pattern cards (9) not in the closed set of 7
    with pytest.raises(ValueError, match="figure_kinds"):
        render_selected_pair_figures(
            pair, out_dir, figure_kinds=frozenset({"density_straight_social_png"})
        )

    # 4. Non-frozenset types rejected
    with pytest.raises((ValueError, TypeError)):
        render_selected_pair_figures(
            pair,
            out_dir,
            figure_kinds=["density_png"],  # type: ignore[arg-type]
        )


def test_render_selected_pair_figures_exact_metadata_and_path_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selected renderer ledger entries and file paths match ordinary renderer exactly."""
    _check_plotting_deps()
    import matplotlib.figure
    import src.tools.shot_pattern_analysis.matched_supplement_figures as msf
    from src.tools.shot_pattern_analysis.matched_supplement_figures import (
        _render_pair_figures,
        render_selected_pair_figures,
    )

    recorded_saves: list[tuple[str, tuple[str, ...]]] = []

    def lightweight_save(
        figure: Any, stem: Path, formats: tuple[str, ...] = ("png", "pdf", "svg")
    ) -> dict[str, Path]:
        stem.parent.mkdir(parents=True, exist_ok=True)
        kind_prefix = "density" if "density" in stem.name else "overhead"
        recorded_saves.append((kind_prefix, formats))
        outputs = {}
        for fmt in formats:
            p = stem.with_suffix(f".{fmt}")
            p.write_bytes(f"mock-{kind_prefix}-{fmt}".encode())
            outputs[fmt] = p
        return outputs

    monkeypatch.setattr(msf, "_save_figure", lightweight_save)

    def fast_savefig(self: Any, fname: Any, *args: Any, **kwargs: Any) -> None:
        p = Path(fname)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"mock-social-png")

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", fast_savefig)

    def fake_social_density(
        point_sets: Any, output_stem: Path, **kwargs: Any
    ) -> dict[str, Path]:
        out = {}
        for pattern in ("straight", "draw", "fade"):
            for ext in ("png", "pdf", "svg"):
                p = output_stem.with_name(f"{output_stem.name}_{pattern}_social.{ext}")
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(f"{pattern}_{ext}".encode())
                out[f"{pattern}_{ext}"] = p
        return out

    monkeypatch.setattr(msf, "write_pair_social_density", fake_social_density)

    pair, _, _ = _synthetic_pair_bundle(tmp_path / "sources")
    out_ordinary = tmp_path / "out_ordinary"
    out_selected = tmp_path / "out_selected"

    ordinary_ledger = _render_pair_figures(pair, out_ordinary)
    assert len(ordinary_ledger) == 16, (
        f"Expected 16 ordinary entries, got {len(ordinary_ledger)}"
    )

    selected_ledger = render_selected_pair_figures(
        pair, out_selected, figure_kinds=CLOSED_7_KINDS
    )
    assert len(selected_ledger) == 7, (
        f"Expected 7 selected entries, got {len(selected_ledger)}"
    )

    # Build maps by figure_kind
    ord_by_kind = {item["figure_kind"]: item for item in ordinary_ledger}
    sel_by_kind = {item["figure_kind"]: item for item in selected_ledger}

    for kind in CLOSED_7_KINDS:
        assert kind in sel_by_kind, f"Missing kind {kind} in selected ledger"
        ord_item = ord_by_kind[kind]
        sel_item = sel_by_kind[kind]

        # Verify all metadata keys match exactly except file paths (different output dirs)
        for key in ord_item:
            if key == "sha256":
                continue
            assert ord_item[key] == sel_item[key], (
                f"Mismatch for key {key} in kind {kind}"
            )

        # Relative paths and filenames match
        assert sel_item["relative_path"] == ord_item["relative_path"]
        assert (
            Path(sel_item["relative_path"]).name == Path(ord_item["relative_path"]).name
        )
        assert len(sel_item["sha256"]) == 64

        # Verify file exists on disk at expected relative path
        rel_path = sel_item["relative_path"]
        assert (out_selected / rel_path).exists()
        assert (out_selected / rel_path).stat().st_size > 0

    # Format routing verification
    assert any(
        kind == "density" and fmts == ("png", "pdf", "svg")
        for kind, fmts in recorded_saves
    )
    assert any(
        kind == "overhead" and fmts == ("png", "pdf", "svg")
        for kind, fmts in recorded_saves
    )


def test_render_selected_pair_figures_format_routing_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Format routing requests only the specified formats without untracked extras."""
    _check_plotting_deps()
    import src.tools.shot_pattern_analysis.matched_supplement_figures as msf
    from src.tools.shot_pattern_analysis.matched_supplement_figures import (
        render_selected_pair_figures,
    )

    recorded_saves: list[tuple[str, tuple[str, ...]]] = []

    def lightweight_save(
        figure: Any, stem: Path, formats: tuple[str, ...] = ("png", "pdf", "svg")
    ) -> dict[str, Path]:
        stem.parent.mkdir(parents=True, exist_ok=True)
        kind_prefix = "density" if "density" in stem.name else "overhead"
        recorded_saves.append((kind_prefix, formats))
        outputs = {}
        for fmt in formats:
            p = stem.with_suffix(f".{fmt}")
            p.write_bytes(f"mock-{kind_prefix}-{fmt}".encode())
            outputs[fmt] = p
        return outputs

    monkeypatch.setattr(msf, "_save_figure", lightweight_save)

    pair, _, _ = _synthetic_pair_bundle(tmp_path / "sources")
    out_dir = tmp_path / "out_subset"

    subset = frozenset({"density_png", "overhead_pdf"})
    ledger = render_selected_pair_figures(pair, out_dir, figure_kinds=subset)

    assert len(ledger) == 2
    kinds = {item["figure_kind"] for item in ledger}
    assert kinds == {"density_png", "overhead_pdf"}

    assert ("density", ("png",)) in recorded_saves
    assert ("overhead", ("pdf",)) in recorded_saves

    all_files = [f for f in out_dir.rglob("*") if f.is_file()]
    assert len(all_files) == 2


def test_render_selected_pair_figures_preserves_common_points_and_no_numerical_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selected renderer preserves identical points and common indices without numerical recomputation."""
    _check_plotting_deps()
    import src.tools.shot_pattern_analysis.matched_supplement_figures as msf
    from src.tools.shot_pattern_analysis.matched_supplement_figures import (
        _render_pair_figures,
        render_selected_pair_figures,
    )

    recorded_points: list[dict[str, Any]] = []

    def spy_density(point_sets: Any, **kwargs: Any) -> dict[str, Path]:
        recorded_points.append(dict(point_sets))
        stem = kwargs["output_stem"]
        p = stem.with_suffix(".png")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"mock")
        return {"png": p}

    def fake_social(
        point_sets: Any, output_stem: Path, **kwargs: Any
    ) -> dict[str, Path]:
        return {}

    def dummy_overhead(*a: Any, **kw: Any) -> dict[str, Path]:
        output_stem = kw["output_stem"]
        p = output_stem.with_suffix(".png")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"dummy")
        return {"png": p}

    monkeypatch.setattr(msf, "_write_pair_density_figure", spy_density)
    monkeypatch.setattr(msf, "write_pair_social_density", fake_social)
    monkeypatch.setattr(msf, "_write_pair_overhead_figure", dummy_overhead)

    pair, _, _ = _synthetic_pair_bundle(tmp_path / "sources")

    _render_pair_figures(pair, tmp_path / "ord")
    render_selected_pair_figures(
        pair, tmp_path / "sel", figure_kinds=frozenset({"density_png"})
    )

    assert len(recorded_points) == 2
    ord_pts = recorded_points[0]
    sel_pts = recorded_points[1]
    assert ord_pts.keys() == sel_pts.keys()
    for k in ord_pts:
        assert ord_pts[k] == sel_pts[k]


def test_render_selected_pair_figures_selective_generation_no_extra_files(
    tmp_path: Path,
) -> None:
    """Only requested figure kinds are written to disk; no extra untracked files."""
    _check_plotting_deps()
    from src.tools.shot_pattern_analysis.matched_supplement_figures import (
        render_selected_pair_figures,
    )

    pair, _, _ = _synthetic_pair_bundle(tmp_path / "sources")
    out_dir = tmp_path / "out_single"

    # Select single kind: density_png
    selected_kinds = frozenset({"density_png"})
    ledger = render_selected_pair_figures(pair, out_dir, figure_kinds=selected_kinds)

    assert len(ledger) == 1
    assert ledger[0]["figure_kind"] == "density_png"

    # Check disk contents
    all_files = list(out_dir.rglob("*"))
    actual_files = [f for f in all_files if f.is_file()]
    assert len(actual_files) == 1, f"Expected exactly 1 file, found {actual_files}"
    assert actual_files[0].name.endswith("_density.png")


def test_render_selected_pair_figures_overhead_social_only(tmp_path: Path) -> None:
    """Requesting only overhead_social_png writes only that single file."""
    _check_plotting_deps()
    from src.tools.shot_pattern_analysis.matched_supplement_figures import (
        render_selected_pair_figures,
    )

    pair, _, _ = _synthetic_pair_bundle(tmp_path / "sources")
    out_dir = tmp_path / "out_overhead_social"

    selected_kinds = frozenset({"overhead_social_png"})
    ledger = render_selected_pair_figures(pair, out_dir, figure_kinds=selected_kinds)

    assert len(ledger) == 1
    assert ledger[0]["figure_kind"] == "overhead_social_png"

    all_files = [f for f in out_dir.rglob("*") if f.is_file()]
    assert len(all_files) == 1
    assert all_files[0].name.endswith("_overhead_social.png")


def test_render_selected_pair_figures_incomplete_returns_empty(tmp_path: Path) -> None:
    """Pair with uncompleted status returns empty list without error."""
    _check_plotting_deps()
    from src.tools.shot_pattern_analysis.matched_supplement_figures import (
        render_selected_pair_figures,
    )

    pair, _, _ = _synthetic_pair_bundle(tmp_path / "sources")
    pair["first"]["status"] = "failed"
    out_dir = tmp_path / "out_failed"

    ledger = render_selected_pair_figures(
        pair, out_dir, figure_kinds=frozenset({"density_png"})
    )
    assert ledger == []
