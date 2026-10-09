"""Small contracts for final matched-study figure selection."""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.unit


def _api():
    from scripts.research.shot_pattern_analysis.build_final_figures import (
        _detail_profile,
        _facebook_selection,
        _heatmap_cells,
        _comparison_pair_specs,
        _render_density_comparison,
    )

    return (
        _detail_profile,
        _facebook_selection,
        _heatmap_cells,
        _comparison_pair_specs,
        _render_density_comparison,
    )


def test_detail_selection_keeps_only_declared_shaft_only_and_combined_profiles() -> None:
    detail_profile, _, _, _, _ = _api()

    assert detail_profile([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) == "shaft_only"
    assert detail_profile([2.0, 0.5, 0.89408, 0.00635, 0.00635, 0.0, 0.0]) == "combined2"
    assert detail_profile([1.0, 0.0, 0.44704, 0.0, 0.0, 0.0, 0.0]) is None


def test_facebook_selection_falls_back_to_pitching_wedge_lean5_when_lean0_unavailable() -> None:
    _, facebook_selection, _, _, _ = _api()
    cells = [
        {"cell": "driver_lean0", "status": "completed", "config": {"club": {"club_id": "driver", "shaft_lean_deg": 0}, "curve_scale": 1, "noise": {"standard_deviations": [1, 0, 0, 0, 0, 0, 0]}}},
        {"cell": "pitching_wedge_lean0", "status": "unavailable", "reason": "unsupported", "config": {"club": {"club_id": "pitching_wedge", "shaft_lean_deg": 0}, "curve_scale": 1, "noise": {"standard_deviations": [1, 0, 0, 0, 0, 0, 0]}}},
        {"cell": "pitching_wedge_lean5", "status": "completed", "config": {"club": {"club_id": "pitching_wedge", "shaft_lean_deg": 5}, "curve_scale": 1, "noise": {"standard_deviations": [1, 0, 0, 0, 0, 0, 0]}}},
    ]

    selected = facebook_selection(cells)

    assert {item["cell"] for item in selected} == {"driver_lean0", "pitching_wedge_lean5"}
    wedge = next(item for item in selected if item["config"]["club"]["club_id"] == "pitching_wedge")
    assert wedge["selection_note"] == "lean 0° unavailable; showing lean 5°"


def test_report_cell_ids_are_adapted_to_heatmap_scenario_names() -> None:
    _, _, heatmap_cells, _, _ = _api()
    cells = [{"cell": "driver_lean0_curve1", "status": "completed"}]

    assert heatmap_cells(cells)[0]["scenario"] == "driver_lean0_curve1"


def test_direct_density_pairs_hold_all_but_face_sd_or_curve_scale_constant() -> None:
    _, _, _, comparison_pairs, _ = _api()
    cells = []
    for sd in (1.0, 2.0):
        for curve in (1.0, 2.0):
            cells.append(
                {
                    "cell": f"driver_lean0_sd{sd:g}_curve{curve:g}",
                    "status": "completed",
                    "target_x_m": 180.0,
                    "config": {
                        "club": {
                            "club_id": "driver",
                            "shaft_lean_deg": 0.0,
                            "contact_law": "incremental_coulomb_energetic",
                        },
                        "noise": {
                            "standard_deviations": [sd, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                        },
                        "curve_scale": curve,
                    },
                }
            )

    pairs = comparison_pairs(cells)
    face_pair = next(pair for pair in pairs if pair["comparison"] == "shaft_rotation_sd")
    curve_pair = next(pair for pair in pairs if pair["comparison"] == "curve_scale")

    assert {item["cell"] for item in face_pair["cells"]} == {
        "driver_lean0_sd1_curve1",
        "driver_lean0_sd2_curve1",
    }
    assert {item["cell"] for item in curve_pair["cells"]} == {
        "driver_lean0_sd2_curve1",
        "driver_lean0_sd2_curve2",
    }
    assert all(item["target_x_m"] == 180.0 for item in face_pair["cells"])


def test_direct_density_comparison_rejects_different_targets(tmp_path) -> None:
    _, _, _, comparison_pairs, render_comparison = _api()
    cells = []
    for sd in (1.0, 2.0):
        cells.append(
            {
                "cell": f"driver_lean0_sd{sd:g}_curve1",
                "status": "completed",
                "target_x_m": 180.0 + (sd - 1.0) * 0.01,
                "config": {
                    "club": {
                        "club_id": "driver",
                        "shaft_lean_deg": 0.0,
                        "contact_law": "incremental_coulomb_energetic",
                    },
                    "noise": {
                        "standard_deviations": [sd, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    },
                    "curve_scale": 1.0,
                },
            }
        )

    pair = next(
        item
        for item in comparison_pairs(cells)
        if item["comparison"] == "shaft_rotation_sd" and item["pattern"] == "Draw"
    )
    result, paths = render_comparison(
        pair,
        tmp_path / "source",
        tmp_path / "presentation_preview",
        dpi=90,
        preview=True,
    )

    assert result["status"] == "unavailable"
    assert result["reason"] == "comparison cells do not share a common target"
    assert paths == []


def test_pilot_preview_writes_separate_manifest_and_facebook_pngs(tmp_path, monkeypatch) -> None:
    from scripts.research.shot_pattern_analysis import build_final_figures as builder
    from src.tools.shot_pattern_analysis import matched_report, matched_report_plots

    source_root = tmp_path / "pilot_report"
    source_root.mkdir()
    output_root = tmp_path / "presentation_preview"
    cell = {
        "cell": "driver_lean0_curve1_rotation1_shaft_only",
        "status": "completed",
        "target_x_m": 180.0,
        "config": {
            "club": {
                "club_id": "driver",
                "shaft_lean_deg": 0.0,
                "contact_law": "incremental_coulomb_energetic",
            },
            "noise": {"standard_deviations": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
            "curve_scale": 1.0,
            "n_shots": 2,
            "seed": 4,
        },
        "summary": {},
        "nominals": {},
    }
    report = {
        "stage": "pilot",
        "planned_cells": 1,
        "counts": {"completed": 1, "unavailable": 0, "not_run": 0},
        "plan_sha256": "plan",
        "cell_manifest_sha256": {cell["cell"]: "manifest"},
        "report_source_sha256": "source",
        "cells": [cell],
    }
    monkeypatch.setattr(matched_report, "load_matrix_report", lambda *_args, **_kwargs: report)

    def fake_render(_cell, _source_root, destination, *, dpi, preview):
        assert dpi == 90
        assert preview is True
        output = destination / "details" / cell["cell"]
        output.mkdir(parents=True)
        paths = [output / "density.png", output / "overhead.png"]
        for path in paths:
            path.write_bytes(b"preview")
        return paths

    monkeypatch.setattr(builder, "_render_detail", fake_render)
    comparison_index = 0

    def fake_render_comparison(pair, _source_root, destination, *, dpi, preview):
        nonlocal comparison_index
        comparison_index += 1
        assert dpi == 90
        assert preview is True
        path = destination / "comparisons" / f"pilot-{comparison_index}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"paired density")
        row = {
            "comparison": pair["comparison"],
            "pattern": pair["pattern"],
            "club_id": pair["club_id"],
            "profile": pair["profile"],
            "first_cell": "pilot-first",
            "second_cell": "pilot-second",
            "first_condition": pair["first_label"],
            "second_condition": pair["second_label"],
            "status": "completed",
            "reason": "",
            "target_x_m": 180.0,
            "png": path.relative_to(destination).as_posix(),
            "pdf": "",
            "svg": "",
        }
        return row, [path]

    monkeypatch.setattr(builder, "_render_density_comparison", fake_render_comparison)
    observed = {}

    def fake_heatmaps(cells, destination, **kwargs):
        observed["scenario"] = cells[0]["scenario"]
        observed["selection_label"] = kwargs["selection_label"]
        destination.mkdir(parents=True)
        image = destination / "overview.png"
        image.write_bytes(b"overview")
        return {"png": image}

    monkeypatch.setattr(matched_report_plots, "export_matched_matrix_heatmaps", fake_heatmaps)

    manifest_path = builder.build_final_figures(
        source_root,
        output_root,
        preview=True,
        selected_cells=(cell["cell"],),
        dpi=90,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["kind"] == "pilot_preview"
    assert manifest["rendered_detail_cells"] == [cell["cell"]]
    assert observed["scenario"] == cell["cell"]
    assert "PILOT PREVIEW" in observed["selection_label"]
    assert manifest["facebook_ready_image_count"] == 6
    assert manifest["facebook_overhead_image_count"] == 1
    assert all((output_root / path).is_file() for path in manifest["outputs"])


def test_detail_metadata_identifies_static_loft() -> None:
    from scripts.research.shot_pattern_analysis.build_final_figures import _cell_metadata
    metadata = _cell_metadata({'config': {'club': {'loft_deg': 34, 'shaft_lean_deg': 10}}}, {}, {})
    assert metadata['static loft deg'] == 34
    assert 'nominal delivered loft deg' not in metadata


@pytest.mark.parametrize('stage,count,shots', [('unknown',324,10000), ('final',32,10000), ('final',324,32)])
def test_final_protocol_rejects_wrong_stage_size_or_population(stage, count, shots) -> None:
    from scripts.research.shot_pattern_analysis.build_final_figures import _validate_final_protocol
    report = {'stage': stage, 'planned_cells': count, 'cells': [{'config': {'n_shots': shots, 'target_x_m': None}}] * count}
    with pytest.raises(ValueError):
        _validate_final_protocol(report, 'final_figures_v1')


def test_final_publication_rechecks_input_bytes(monkeypatch, tmp_path) -> None:
    from scripts.research.shot_pattern_analysis.build_final_figures import _assert_inputs_unchanged
    from src.tools.shot_pattern_analysis import matched_report
    def changed(*args, **kwargs):
        raise ValueError('cell integrity failure: changed CSV')
    monkeypatch.setattr(matched_report, 'load_matrix_report', changed)
    with pytest.raises(ValueError, match='integrity failure'):
        _assert_inputs_unchanged(tmp_path, {}, preview=False)


@pytest.mark.parametrize('rewrite_manifest', [False, True])
def test_publication_rejects_changed_saved_bytes_even_with_new_manifest(tmp_path, rewrite_manifest) -> None:
    import hashlib
    from scripts.research.shot_pattern_analysis.build_final_figures import _assert_inputs_unchanged
    from src.tools.shot_pattern_analysis.matched_report import load_matrix_report
    config = {'n_shots': 1}
    cell = tmp_path / 'cell'
    cell.mkdir()
    (tmp_path / 'matrix_plan.json').write_text(json.dumps({'stage': 'pilot', 'execution': {}, 'cells': {'cell': config}}))
    (cell / 'run_start.json').write_text(json.dumps({'config': config, 'execution': {}}))
    (cell / 'summary.json').write_text(json.dumps({'config': config, 'status': 'unavailable'}))
    (cell / 'shots.csv.gz').write_bytes(b'original saved bytes')
    def write_manifest():
        (cell / 'manifest.json').write_text(json.dumps({'files_sha256': {name: hashlib.sha256((cell / name).read_bytes()).hexdigest() for name in ('run_start.json', 'summary.json', 'shots.csv.gz')}}))
    write_manifest()
    report = load_matrix_report(tmp_path)
    (cell / 'shots.csv.gz').write_bytes(b'changed after initial validation')
    if rewrite_manifest:
        write_manifest()
    with pytest.raises((ValueError, RuntimeError)):
        _assert_inputs_unchanged(tmp_path, report, preview=True)


def test_detail_metadata_names_matched_raw_landing_and_pattern_nominals() -> None:
    from scripts.research.shot_pattern_analysis.build_final_figures import _cell_metadata

    cell = {
        "config": {"club": {"club_id": "driver", "loft_deg": 10.0}},
        "nominals": {
            "Straight": {"face_deg": 0.0, "path_deg": 0.0, "speed_mps": 45.0},
            "Draw": {"face_deg": 2.1, "path_deg": 3.6, "speed_mps": 45.2},
            "Fade": {"face_deg": -2.1, "path_deg": -3.6, "speed_mps": 45.2},
        },
    }

    metadata = _cell_metadata(cell, {}, {})

    assert metadata["delivery mode"] == "Matched Raw Landing"
    assert "Straight" in metadata["nominal pattern solutions"]
    assert "face +2.100°" in metadata["nominal pattern solutions"]
    assert "path +3.600°" in metadata["nominal pattern solutions"]
    assert "speed 45.200 m/s" in metadata["nominal pattern solutions"]
