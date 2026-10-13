"""Focused regression tests for canonical social density and overhead asset integration."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from src.tools.shot_pattern_analysis import (
    matched_supplement_figures as figures,
    matched_supplement_pipeline as pipeline,
)

pytestmark = pytest.mark.unit


def _fixture_ref(
    cell: str,
    club: str = "driver",
    lean: float = 5.0,
    curve: float = 2.0,
    rotation: float = 1.0,
    profile: str = "combined2",
    target: float = 180.0,
    protocol: str = "original_primary/1",
) -> dict:
    return {
        "cell": f"{club}_lean{lean:g}_curve{curve:g}_rotation{rotation:g}_{profile}",
        "source_protocol": protocol,
        "execution_kind": "unique",
        "status": "completed",
        "source_root": f"/{protocol}",
        "config": {
            "club": {"club_id": club, "shaft_lean_deg": lean},
            "curve_scale": curve,
            "noise": {
                "standard_deviations": [
                    rotation,
                    0.5,
                    0.89408,
                    0.00635,
                    0.00635,
                    0.0,
                    0.0,
                ]
            },
            "n_shots": 10_000,
            "seed": 7,
            "target_x_m": target,
        },
        "summary": {"status": "completed", "target_x_m": target},
        "source_csv_sha256": cell,
    }


def test_report_source_hashes_includes_matched_pair_social():
    hashes = pipeline._report_source_hashes()
    assert "matched_pair_social.py" in hashes
    package_dir = Path(pipeline.__file__).parent
    social_path = package_dir / "matched_pair_social.py"
    assert hashes["matched_pair_social.py"] == pipeline.sha256_file(social_path)


def test_condition_caption_uses_readable_physical_units_and_preserves_title_case():
    ref = _fixture_ref("cell-c2", profile="combined2")
    caption = figures._condition_caption(ref)
    assert "Combined 2" in caption
    assert "Path SD 0.5°" in caption
    assert "Speed SD 2 mph" in caption
    assert "Strike SD 0.25 in on Each Axis" in caption
    assert "Shaft Twist SD 1°" in caption
    assert "Lean 5°" in caption
    assert "Curve ×2" in caption

    # Test nonzero correlation profile
    ref_cov = _fixture_ref("cell-cov", profile="cov_facepath_positive")
    correlation = np.eye(7)
    correlation[0, 1] = correlation[1, 0] = -0.6
    ref_cov["config"]["noise"]["correlation"] = correlation.tolist()
    caption_cov = figures._condition_caption(ref_cov)
    assert "Shaft Twist–Path Correlation r = -0.6" in caption_cov


def test_pair_social_scenario_distinguishes_varying_panels_from_fixed_conditions():
    first = _fixture_ref("c1", rotation=1.0)
    second = _fixture_ref("c2", rotation=2.0)
    scenario = figures._pair_social_scenario("shaft_rotation_sd", first, second)

    # Varying condition in panels
    assert (
        "Varying Shaft Rotation SD in Panels" in scenario
        or "Varying Shaft Rotation SD Across Panels" in scenario
    )
    # Fixed conditions
    assert "Fixed:" in scenario
    assert "Lean 5°" in scenario
    assert "Curve ×2" in scenario
    from matplotlib.textpath import TextPath

    assert TextPath((0, 0), scenario, size=14).get_extents().width < 0.88 * 16 * 72
    # Noise units belong in wrapped metadata; the canonical footer supplies
    # the model-conditional simulation caveat.
    assert "Path SD" not in scenario


def test_render_cell_figures_exports_dedicated_canonical_social_assets(
    tmp_path, monkeypatch
):
    root = tmp_path / "cell_source"
    cell_name = "driver_lean5_curve2_rotation1_combined2"
    cell_dir = root / cell_name
    cell_dir.mkdir(parents=True)
    (cell_dir / "manifest.json").write_text("{}")
    with gzip.open(cell_dir / "shots.csv.gz", "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("pattern", "shot_index", "status", "carry_x_m", "carry_y_m"))
        for pattern in ("Straight", "Draw", "Fade"):
            for i in range(10):
                writer.writerow((pattern, i, "supported", 180.0 + i, 0.1 * i))
    (cell_dir / "flight_samples.json").write_text(
        json.dumps(
            {
                "trajectories": {
                    p: [[[0.0, 0.0], [180.0, 0.0]]]
                    for p in ("Straight", "Draw", "Fade")
                }
            }
        )
    )
    ref = _fixture_ref(cell_name, profile="combined2")
    ref["source_root"] = str(root)
    ref["config"]["n_shots"] = 10
    ref["summary"] = {"target_x_m": 180.0, "status": "completed"}

    monkeypatch.setattr(figures, "manifest_files_match", lambda path: True)

    from typing import Any

    calls: dict[str, list[Any]] = {"dispersion": [], "overhead": []}

    def capture_dispersion(arrays, stem, **kwargs):
        calls["dispersion"].append((stem, kwargs))
        out = {}
        for ext in ("png", "pdf", "svg"):
            p = stem.with_suffix(f".{ext}")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(f"{stem.name}_{ext}".encode())
            out[ext] = p
        return out

    def capture_overhead(trajectories, stem, **kwargs):
        calls["overhead"].append((stem, kwargs))
        out = {}
        for ext in ("png", "pdf", "svg"):
            p = stem.with_suffix(f".{ext}")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(f"{stem.name}_{ext}".encode())
            out[ext] = p
        return out

    monkeypatch.setattr(figures, "write_academic_dispersion_figure", capture_dispersion)
    monkeypatch.setattr(figures, "write_overhead_trajectory_figure", capture_overhead)

    output_dir = tmp_path / "report_figures"
    ledger = figures._render_cell_figures(ref, output_dir)
    for _, kwargs in calls["dispersion"] + calls["overhead"]:
        assert "Supplement Cell" not in kwargs["scenario"]
        assert "stage" not in kwargs["metadata"]
    for _, kwargs in calls["overhead"]:
        assert "valid_shots_by_pattern" not in kwargs["metadata"]
        assert "total_shots_by_pattern" not in kwargs["metadata"]
        assert kwargs["metadata"]["figure_scope"] == "Saved Raw Flight Samples"
        if kwargs["options"].presentation == "social":
            assert (
                kwargs["metadata"]["Population"]
                == "Saved Raw Trajectories; Not Paired by Shot Index"
            )

    # Must contain both academic and social exports for density and overhead
    kinds = {item["figure_kind"] for item in ledger}
    assert "density_png" in kinds
    assert "density_pdf" in kinds
    assert "density_svg" in kinds
    assert "density_social_png" in kinds
    social_call = next(
        kwargs
        for _, kwargs in calls["dispersion"]
        if kwargs["options"].presentation == "social"
    )
    assert "Path SD 0.5°" in social_call["metadata"]["Delivery Noise"]
    assert "Speed SD 2 mph" in social_call["metadata"]["Delivery Noise"]
    assert "Strike SD 0.25 in" in social_call["metadata"]["Delivery Noise"]
    assert "Protocol" in social_call["metadata"]
    assert social_call["metadata"]["Protocol"] == ref["source_protocol"]
    assert "Matched Target" in social_call["metadata"]
    assert "180.00 m downrange" in social_call["metadata"]["Matched Target"]
    assert "Fixed-Target Control" in social_call["metadata"]["Matched Target"]

    overhead_social_call = next(
        kwargs
        for _, kwargs in calls["overhead"]
        if kwargs["options"].presentation == "social"
    )
    assert "Protocol" in overhead_social_call["metadata"]
    assert overhead_social_call["metadata"]["Protocol"] == ref["source_protocol"]
    assert "Matched Target" in overhead_social_call["metadata"]
    assert "180.00 m downrange" in overhead_social_call["metadata"]["Matched Target"]

    assert "density_social_pdf" in kinds
    assert "density_social_svg" in kinds
    assert "overhead_png" in kinds
    assert "overhead_pdf" in kinds
    assert "overhead_svg" in kinds
    assert "overhead_social_png" in kinds
    assert "overhead_social_pdf" in kinds
    assert "overhead_social_svg" in kinds
    assert len(ledger) == 12

    # Presentation must be explicitly labeled
    for item in ledger:
        assert item["presentation"] in ("academic", "social")
        if "social" in item["figure_kind"]:
            assert item["presentation"] == "social"
        else:
            assert item["presentation"] == "academic"

    # Overhead must NOT have common-six or supported counts
    for item in ledger:
        if "overhead" in item["figure_kind"]:
            assert item["valid_n_by_pattern"] is None
            assert item["total_n_by_pattern"] is None
            assert item["saved_trajectory_samples_by_pattern"] == {
                "Straight": 1,
                "Draw": 1,
                "Fade": 1,
            }
        else:
            assert item["valid_n_by_pattern"] == {
                "Straight": 10,
                "Draw": 10,
                "Fade": 10,
            }
            assert item["saved_trajectory_samples_by_pattern"] is None


def test_render_pair_figures_exports_canonical_social_density_with_physical_patterns(
    tmp_path, monkeypatch
):
    first = _fixture_ref("cell1", rotation=1.0)
    second = _fixture_ref("cell2", rotation=2.0)

    for ref in (first, second):
        root = tmp_path / ref["cell"]
        cell_dir = root / ref["cell"]
        cell_dir.mkdir(parents=True)
        (cell_dir / "flight_samples.json").write_text(
            json.dumps(
                {
                    "trajectories": {
                        p: [[[0.0, 0.0], [180.0, 0.0]]]
                        for p in ("Straight", "Draw", "Fade")
                    }
                }
            )
        )
        ref["source_root"] = str(root)

    # 10 common shot indices
    common_endpoints = {
        p: [(i, 180.0 + i, 0.1 * i) for i in range(10)]
        for p in ("Straight", "Draw", "Fade")
    }
    monkeypatch.setattr(
        figures, "_supported_endpoints", lambda *args: (common_endpoints, {})
    )

    social_calls = []

    def fake_social_density(point_sets, output_stem, **kwargs):
        social_calls.append((point_sets, output_stem, kwargs))
        out = {}
        for pattern in ("straight", "draw", "fade"):
            for ext in ("png", "pdf", "svg"):
                p = output_stem.with_name(f"{output_stem.name}_{pattern}_social.{ext}")
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(f"{pattern}_{ext}".encode())
                out[f"{pattern}_{ext}"] = p
        return out

    monkeypatch.setattr(figures, "write_pair_social_density", fake_social_density)

    pair = {
        "comparison_kind": "shaft_rotation_sd",
        "first": first,
        "second": second,
        "target_x_m": 180.0,
    }
    output_dir = tmp_path / "report_figures"
    ledger = figures._render_pair_figures(pair, output_dir)

    # Check social density call
    assert len(social_calls) == 1
    passed_points, passed_stem, kwargs = social_calls[0]
    assert set(kwargs) == {"context"}
    context = kwargs["context"]
    assert context.condition_labels == ("Rotation SD 1°", "Rotation SD 2°")
    assert context.social_club == "Driver"
    assert context.total_n == 10_000
    assert context.target_point_m == (180.0, 0.0)
    assert "Path SD 0.5°" in context.delivery_noise
    assert "Speed SD 2 mph" in context.delivery_noise
    assert "Strike SD 0.25 in" in context.delivery_noise
    assert "Fixed:" in context.scenario
    assert context.source_protocols == (
        first["source_protocol"],
        second["source_protocol"],
    )
    assert context.target_modes == (
        "Fixed-Target Control",
        "Fixed-Target Control",
    )

    # Check that point_sets passed to social wrapper use exact common coordinates
    for pattern in ("Straight", "Draw", "Fade"):
        for label in ("Rotation SD 1°", "Rotation SD 2°"):
            key = f"{pattern} · {label}"
            assert key in passed_points
            assert len(passed_points[key]) == 10

    # Check ledger items
    kinds = {item["figure_kind"] for item in ledger}
    # Academic density
    assert {"density_png", "density_pdf", "density_svg"} <= kinds
    # Pair overhead
    assert {
        "overhead_png",
        "overhead_pdf",
        "overhead_svg",
        "overhead_social_png",
    } <= kinds
    # Pair social density for the 3 physical patterns
    for p in ("straight", "draw", "fade"):
        for ext in ("png", "pdf", "svg"):
            assert f"density_{p}_social_{ext}" in kinds

    # Verify explicit labeling for pair social density
    social_density_items = [
        item
        for item in ledger
        if item["figure_kind"].startswith("density_")
        and "social" in item["figure_kind"]
    ]
    assert len(social_density_items) == 9
    for item in social_density_items:
        assert item["presentation"] == "social"
        assert item["physical_pattern"] in ("Straight", "Draw", "Fade")
        assert item["common_supported_n"] == 10
        assert item["total_n_per_population"] == 10_000
        assert (
            item["conditioning"] == "same shot indices supported in all six populations"
        )

    # Overhead items must NEVER have common-six counts
    overhead_items = [
        item for item in ledger if item["figure_kind"].startswith("overhead_")
    ]
    assert len(overhead_items) == 4
    for item in overhead_items:
        assert "common_supported_n" not in item or item["common_supported_n"] is None
        assert (
            "total_n_per_population" not in item
            or item["total_n_per_population"] is None
        )
        assert item["saved_trajectory_samples_by_pattern"] is not None


def test_pair_figure_ledger_preserves_unavailable_row_and_keeps_overhead(
    tmp_path, monkeypatch
):
    first = _fixture_ref("c1", rotation=1.0)
    second = _fixture_ref("c2", rotation=2.0)

    for ref in (first, second):
        root = tmp_path / ref["cell"]
        cell_dir = root / ref["cell"]
        cell_dir.mkdir(parents=True)
        (cell_dir / "flight_samples.json").write_text(
            json.dumps(
                {
                    "trajectories": {
                        p: [[[0.0, 0.0], [180.0, 0.0]]]
                        for p in ("Straight", "Draw", "Fade")
                    }
                }
            )
        )
        ref["source_root"] = str(root)

    # 1 common shot index (fewer than 2 -> unavailable density)
    common_endpoints = {p: [(0, 180.0, 0.0)] for p in ("Straight", "Draw", "Fade")}
    monkeypatch.setattr(
        figures, "_supported_endpoints", lambda *args: (common_endpoints, {})
    )

    pair = {
        "comparison_kind": "shaft_rotation_sd",
        "first": first,
        "second": second,
        "target_x_m": 180.0,
    }
    output_dir = tmp_path / "report_figures"
    ledger = figures._render_pair_figures(pair, output_dir)

    density_unavailable = [
        item for item in ledger if item.get("status") == "unavailable"
    ]
    assert len(density_unavailable) == 1
    assert density_unavailable[0]["figure_kind"] == "paired_density"
    assert (
        density_unavailable[0]["reason"]
        == "fewer than two indices supported in all six populations"
    )

    # Overheads are still available and not hidden
    overhead_items = [
        item for item in ledger if item["figure_kind"].startswith("overhead_")
    ]
    assert len(overhead_items) == 4


def test_render_cell_figures_social_metadata_distinguishes_within_lean_and_control_alias(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(figures, "manifest_files_match", lambda path: True)
    monkeypatch.setattr(
        figures,
        "load_individually_supported_endpoints",
        lambda *args: (
            {p: [(77.4, 0.0), (77.5, 0.1)] for p in ("Straight", "Draw", "Fade")},
            {},
        ),
    )

    captured_metadata = []

    def capture_dispersion(arrays, stem, **kwargs):
        if kwargs["options"].presentation == "social":
            captured_metadata.append(kwargs["metadata"])
        out = {}
        for ext in ("png", "pdf", "svg"):
            p = stem.with_suffix(f".{ext}")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(f"{stem.name}_{ext}".encode())
            out[ext] = p
        return out

    def capture_overhead(trajectories, stem, **kwargs):
        out = {}
        for ext in ("png", "pdf", "svg"):
            p = stem.with_suffix(f".{ext}")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(f"{stem.name}_{ext}".encode())
            out[ext] = p
        return out

    monkeypatch.setattr(figures, "write_academic_dispersion_figure", capture_dispersion)
    monkeypatch.setattr(figures, "write_overhead_trajectory_figure", capture_overhead)

    # 1. Primary cell: within-lean nominal target (config target_x_m is None)
    root = tmp_path / "primary_cell"
    primary_ref = _fixture_ref(
        "pitching_wedge_lean0_curve2_rotation1_combined2",
        club="pitching_wedge",
        lean=0.0,
        curve=2.0,
        rotation=1.0,
        profile="combined2",
        target=77.402477248663,
        protocol="pw-primary-face-bounds/1",
    )
    primary_ref["source_root"] = str(root)
    primary_ref["config"]["target_x_m"] = None  # Within-lean matching
    primary_ref["summary"] = {"target_x_m": 77.402477248663, "status": "completed"}

    cell_dir = root / primary_ref["cell"]
    cell_dir.mkdir(parents=True)
    (cell_dir / "manifest.json").write_text("{}")
    (cell_dir / "flight_samples.json").write_text(
        json.dumps(
            {
                "trajectories": {
                    p: [[[0.0, 0.0], [77.4, 0.0]]] for p in ("Straight", "Draw", "Fade")
                }
            }
        )
    )

    figures._render_cell_figures(primary_ref, tmp_path / "figures_primary")
    assert len(captured_metadata) == 1
    m1 = captured_metadata[-1]
    assert m1["Protocol"] == "pw-primary-face-bounds/1"
    assert m1["Matched Target"] == "77.40 m downrange (Within-Lean Nominal Target)"

    # 2. Control cell: fixed-target control (config target_x_m is explicitly set)
    control_ref = _fixture_ref(
        "pitching_wedge_lean5_curve2_rotation1_combined2",
        club="pitching_wedge",
        lean=5.0,
        curve=2.0,
        rotation=1.0,
        profile="combined2",
        target=77.402477248663,
        protocol="pw-control-face-bounds/1",
    )
    control_ref["source_root"] = str(root)
    control_ref["config"]["target_x_m"] = 77.402477248663
    control_ref["summary"] = {"target_x_m": 77.402477248663, "status": "completed"}

    ctrl_dir = root / control_ref["cell"]
    ctrl_dir.mkdir(parents=True)
    (ctrl_dir / "manifest.json").write_text("{}")
    (ctrl_dir / "flight_samples.json").write_text(
        json.dumps(
            {
                "trajectories": {
                    p: [[[0.0, 0.0], [77.4, 0.0]]] for p in ("Straight", "Draw", "Fade")
                }
            }
        )
    )

    figures._render_cell_figures(control_ref, tmp_path / "figures_control")
    assert len(captured_metadata) == 2
    m2 = captured_metadata[-1]
    assert m2["Protocol"] == "pw-control-face-bounds/1"
    assert m2["Matched Target"] == "77.40 m downrange (Fixed-Target Control)"

    # 3. Provenance alias: real PW control aliases are same-cell zero-lean reuse
    # of zero-lean primary with requested_config target set.
    alias_ref = dict(primary_ref)
    alias_ref["execution_kind"] = "provenance_alias"
    alias_ref["alias_of_primary"] = primary_ref["cell"]
    alias_ref["alias_cell"] = primary_ref["cell"]
    alias_ref["requested_config"] = {
        **primary_ref["config"],
        "target_x_m": 77.402477248663,
    }

    figures._render_cell_figures(alias_ref, tmp_path / "figures_alias")
    assert len(captured_metadata) == 3
    m3 = captured_metadata[-1]
    assert m3["Protocol"] == "pw-primary-face-bounds/1"
    assert m3["Matched Target"] == "77.40 m downrange (Fixed-Target Control (Alias))"

    # Missing actual protocol should fail clearly without protocol_id fallback
    bad_cell_ref = dict(primary_ref)
    bad_cell_ref.pop("source_protocol", None)
    bad_cell_ref["protocol_id"] = "pw-control-face-bounds/1"
    with pytest.raises(ValueError, match="cell reference lacks source protocol"):
        figures._render_cell_figures(bad_cell_ref, tmp_path / "figures_bad")


def test_render_pair_figures_cross_protocol_passes_ordered_protocols(
    tmp_path, monkeypatch
):
    first = _fixture_ref(
        "pitching_wedge_lean0_curve2_rotation1_combined2",
        club="pitching_wedge",
        lean=0.0,
        rotation=1.0,
        protocol="original_primary/1",
    )
    first["config"]["target_x_m"] = None  # Within-Lean Nominal Target

    second = _fixture_ref(
        "pitching_wedge_lean5_curve2_rotation1_combined2",
        club="pitching_wedge",
        lean=5.0,
        rotation=2.0,
        protocol="pw-control-face-bounds/1",
    )
    second["config"]["target_x_m"] = 180.0  # Fixed-Target Control

    for ref in (first, second):
        root = tmp_path / ref["cell"]
        cell_dir = root / ref["cell"]
        cell_dir.mkdir(parents=True)
        (cell_dir / "flight_samples.json").write_text(
            json.dumps(
                {
                    "trajectories": {
                        p: [[[0.0, 0.0], [180.0, 0.0]]]
                        for p in ("Straight", "Draw", "Fade")
                    }
                }
            )
        )
        ref["source_root"] = str(root)

    common_endpoints = {
        p: [(i, 180.0 + i, 0.1 * i) for i in range(10)]
        for p in ("Straight", "Draw", "Fade")
    }
    monkeypatch.setattr(
        figures, "_supported_endpoints", lambda *args: (common_endpoints, {})
    )

    social_calls = []

    def fake_social_density(point_sets, output_stem, **kwargs):
        social_calls.append(kwargs)
        out = {}
        for p in ("straight", "draw", "fade"):
            for ext in ("png", "pdf", "svg"):
                path = output_stem.with_name(f"{output_stem.name}_{p}.{ext}")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"dummy")
                out[f"{p}_{ext}"] = path
        return out

    monkeypatch.setattr(figures, "write_pair_social_density", fake_social_density)

    pair = {
        "comparison_kind": "curve_strength",
        "first": first,
        "second": second,
        "target_x_m": 180.0,
    }
    figures._render_pair_figures(pair, tmp_path / "comparisons")

    assert len(social_calls) == 1
    kwargs = social_calls[0]
    assert set(kwargs) == {"context"}
    context = kwargs["context"]
    assert context.source_protocols == (
        "original_primary/1",
        "pw-control-face-bounds/1",
    )
    assert context.target_modes == (
        "Within-Lean Nominal Target",
        "Fixed-Target Control",
    )
    assert context.target_point_m == (180.0, 0.0)

    # Missing actual protocol should fail clearly without protocol_id fallback
    bad_pair_first = dict(first)
    bad_pair_first.pop("source_protocol", None)
    bad_pair_first["protocol_id"] = "original_primary/1"
    with pytest.raises(ValueError, match="pair references lack source protocol"):
        figures._render_pair_figures(
            {
                "comparison_kind": "curve_strength",
                "first": bad_pair_first,
                "second": second,
                "target_x_m": 180.0,
            },
            tmp_path / "comparisons_bad",
        )
