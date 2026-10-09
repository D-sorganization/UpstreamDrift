"""Publication orchestration stays separate from frozen matrix inventories."""

from __future__ import annotations

import json
import pytest
import csv
import gzip

pytestmark = pytest.mark.unit


def _ref(cell, club, lean, curve, rotation, target=100.0, protocol="original"):
    profile = "shaft_only"
    return {
        "cell": f"{club}_lean{lean:g}_curve{curve:g}_rotation{rotation:g}_{profile}",
        "source_protocol": protocol,
        "execution_kind": "unique",
        "status": "completed",
        "source_root": f"/{protocol}",
        "config": {
            "club": {"club_id": club, "shaft_lean_deg": lean},
            "curve_scale": curve,
            "noise": {"standard_deviations": [rotation]},
            "n_shots": 10_000,
            "seed": 7,
            "target_x_m": target,
        },
        "summary": {"status": "completed", "target_x_m": target},
        "source_csv_sha256": cell,
    }


def test_selects_curve_lean_and_sd_pairs_without_reflection_or_deduping_patterns():
    from src.tools.shot_pattern_analysis.matched_supplement_pipeline import (
        select_explicit_pairs,
    )

    cells = [
        _ref("pw0c1r1", "pitching_wedge", 0, 1, 1),
        _ref("pw0c2r1", "pitching_wedge", 0, 2, 1, protocol="supplement"),
        _ref("pw5c2r1", "pitching_wedge", 5, 2, 1, protocol="supplement"),
        _ref("pw10c2r1", "pitching_wedge", 10, 2, 1, protocol="supplement"),
        _ref("pw0c2r2", "pitching_wedge", 0, 2, 2, protocol="supplement"),
        _ref("pw0c1r2", "pitching_wedge", 0, 1, 2),
    ]

    pairs = select_explicit_pairs(cells)
    identities = {
        (item["comparison_kind"], item["first"]["cell"], item["second"]["cell"])
        for item in pairs
    }
    assert ("curve_strength", cells[0]["cell"], cells[1]["cell"]) in identities
    assert ("across_lean", cells[1]["cell"], cells[2]["cell"]) in identities
    assert ("across_lean", cells[2]["cell"], cells[3]["cell"]) in identities
    assert ("across_lean", cells[1]["cell"], cells[3]["cell"]) in identities
    assert ("shaft_rotation_sd", cells[1]["cell"], cells[4]["cell"]) in identities
    assert all(
        item["physical_patterns"] == ("Straight", "Draw", "Fade") for item in pairs
    )


def test_final_root_validation_requires_final_stage_expected_population_and_10k(
    tmp_path,
):
    from src.tools.shot_pattern_analysis.matched_supplement_pipeline import (
        validate_final_matrix_report,
    )

    def report(stage="final", planned=324, n=10_000, not_run=0):
        return {
            "stage": stage,
            "planned_cells": planned,
            "counts": {"not_run": not_run},
            "cells": [
                {"cell": f"cell-{i}", "config": {"n_shots": n}} for i in range(planned)
            ],
        }

    validate_final_matrix_report(report(), expected_cells=324)
    with pytest.raises(ValueError, match="stage"):
        validate_final_matrix_report(report(stage="pilot"), expected_cells=324)
    with pytest.raises(ValueError, match="planned cell count"):
        validate_final_matrix_report(report(planned=323), expected_cells=324)
    with pytest.raises(ValueError, match="10,000"):
        validate_final_matrix_report(report(n=32), expected_cells=324)
    with pytest.raises(ValueError, match="incomplete"):
        validate_final_matrix_report(report(not_run=1), expected_cells=324)


def test_original_source_inventory_keeps_every_availability_state():
    import csv
    from io import StringIO

    from src.tools.shot_pattern_analysis.matched_supplement_pipeline import (
        render_original_source_inventory,
    )

    reports = {
        "original_primary": {
            "plan_sha256": "primary-plan",
            "cells": [
                {
                    "cell": f"primary-{index}",
                    "status": "completed" if index % 2 else "unavailable",
                    "reason": "unsupported nominal" if index % 2 == 0 else "",
                    "config": {"n_shots": 10_000},
                }
                for index in range(324)
            ],
        },
        "original_control": {
            "plan_sha256": "control-plan",
            "cells": [
                {
                    "cell": f"control-{index}",
                    "status": "unavailable",
                    "unavailable_reason": "no airborne nominal",
                    "config": {"n_shots": 10_000},
                }
                for index in range(72)
            ],
        },
    }
    rows = list(csv.DictReader(StringIO(render_original_source_inventory(reports))))
    assert len(rows) == 396
    assert rows[0]["source_protocol"] == "original_primary/1"
    assert rows[0]["unavailable_reason"] == "unsupported nominal"
    assert rows[-1]["source_protocol"] == "original_control/1"
    assert rows[-1]["unavailable_reason"] == "no airborne nominal"


def test_original_protocol_validation_checks_exact_factorial_configs():
    from dataclasses import asdict

    from src.tools.shot_pattern_analysis.matched_matrix import matched_matrix_cells
    from src.tools.shot_pattern_analysis.matched_report import (
        validate_original_matrix_protocol,
    )

    cells = [
        {"cell": name, "config": json.loads(json.dumps(asdict(config)))}
        for name, config in matched_matrix_cells(n_shots=10_000, seed=20261009)
    ]
    validate_original_matrix_protocol(
        {"cells": cells}, protocol="original_primary"
    )
    altered = [dict(cell) for cell in cells]
    altered[0] = {**altered[0], "config": {**altered[0]["config"], "seed": 17}}
    with pytest.raises(ValueError, match="differ from the reviewed protocol"):
        validate_original_matrix_protocol(
            {"cells": altered}, protocol="original_primary"
        )

    target_map = {
        "driver": 178.94875782337786,
        "seven_iron": 127.88194166336235,
        "pitching_wedge": 77.40247724866299,
    }
    control_cells = [
        {"cell": name, "config": json.loads(json.dumps(asdict(config)))}
        for name, config in matched_matrix_cells(
            n_shots=10_000,
            seed=20261009,
            profiles=("shaft_only", "combined2"),
            common_targets=target_map,
        )
    ]
    validate_original_matrix_protocol(
        {"cells": control_cells},
        protocol="original_control",
        common_targets=target_map,
    )


def test_common_targets_are_bound_to_accepted_control_pilot():
    from pathlib import Path

    from src.tools.shot_pattern_analysis.matched_report import (
        load_reviewed_common_targets,
    )

    phase2 = (
        Path(__file__).resolve().parents[3]
        / "docs/research/shot_pattern_analysis/phase2"
    )
    targets, sources = load_reviewed_common_targets(phase2)
    assert targets["driver"] == 178.94875782337786
    assert targets["seven_iron"] == 127.88194166336235
    assert targets["pitching_wedge"] == 77.40247724866299
    assert set(sources) == {
        "acceptance_sha256",
        "audit_sha256",
        "pilot_plan_sha256",
        "common_targets_sha256",
    }


def test_markdown_comparison_table_keeps_population_scope_and_unavailable_sg():
    from src.tools.shot_pattern_analysis.matched_supplement_pipeline import (
        render_comparison_markdown,
    )

    rendered = render_comparison_markdown(
        [
            {
                "comparison_kind": "curve_strength",
                "status": "conditional_only",
                "common_supported_n": 19,
                "total_n_per_population": 20,
                "sources": [
                    {"protocol_id": "primary", "cell": "a"},
                    {"protocol_id": "supplement", "cell": "b"},
                ],
                "patterns": {
                    pattern: {
                        "radial_error_difference_m": {
                            "estimate": 0.1,
                            "lower_95": 0.0,
                            "upper_95": 0.2,
                        },
                        "course_sg_difference": {
                            "estimate": 0.03,
                            "lower_95": 0.01,
                            "upper_95": 0.05,
                        },
                    }
                    for pattern in ("Straight", "Draw", "Fade")
                },
                "course_scoring_effect_modification": {
                    "status": "unavailable",
                    "reason": "outside baseline",
                    "Draw": {
                        "estimate": 0.02,
                        "lower_95": 0.0,
                        "upper_95": 0.04,
                    },
                },
            }
        ]
    )
    assert "19 / 20" in rendered
    assert "Unavailable" in rendered
    assert "results do not isolate a causal curvature effect" in rendered
    assert "Paired Pattern SG Change" in rendered
    assert "+0.0300" in rendered
    assert "+0.0200" in rendered
    assert "not independent experimental replications" in rendered


def test_density_dot_overlay_uses_deterministic_cap_without_truncating_density_data():
    from src.tools.shot_pattern_analysis.matched_supplement_figures import (
        _displayed_sample_indices,
    )

    assert _displayed_sample_indices(4).tolist() == [0, 1, 2, 3]
    indices = _displayed_sample_indices(10_000)
    assert len(indices) == 600
    assert indices[0] == 0
    assert indices[-1] == 9_999
    assert (indices[1:] > indices[:-1]).all()


def test_cell_figure_ledger_keeps_density_and_overhead_exports_distinct(
    tmp_path, monkeypatch
):
    import json

    from src.tools.shot_pattern_analysis import matched_supplement_pipeline as module
    from src.tools.shot_pattern_analysis import matched_supplement_figures as figures

    root = tmp_path / "source"
    cell = "pitching_wedge_lean0_curve2_rotation1_shaft_only"
    cell_dir = root / cell
    cell_dir.mkdir(parents=True)
    (cell_dir / "manifest.json").write_text("{}")
    with gzip.open(cell_dir / "shots.csv.gz", "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("pattern", "shot_index", "status", "carry_x_m", "carry_y_m"))
        for pattern, supported_n in (("Straight", 8), ("Draw", 9), ("Fade", 10)):
            for index in range(10):
                writer.writerow((pattern, index, "supported" if index < supported_n else "off_face", 1.0, 2.0))
    (cell_dir / "flight_samples.json").write_text(
        json.dumps({"trajectories": {p: [[0.0, 0.0]] for p in ("Straight", "Draw", "Fade")}})
    )
    ref = _ref(cell, "pitching_wedge", 0, 2, 1, protocol="pw-primary-face-bounds/1")
    ref["source_root"] = str(root)
    ref["config"]["n_shots"] = 10
    ref["summary"] = {
        "target_x_m": 77.0,
        "summary": {p: {"supported_n": 10} for p in ("Straight", "Draw", "Fade")},
    }
    monkeypatch.setattr(figures, "manifest_files_match", lambda path: True)
    density_metadata = []

    def write_group(stem):
        outputs = {}
        for ext in ("png", "pdf", "svg"):
            path = stem.with_suffix(f".{ext}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(ext)
            outputs[ext] = path
        return outputs

    monkeypatch.setattr(
        figures,
        "write_academic_dispersion_figure",
        lambda *args, **kwargs: density_metadata.append(kwargs["metadata"]) or write_group(args[1]),
    )
    monkeypatch.setattr(figures, "write_overhead_trajectory_figure", lambda *args, **kwargs: write_group(args[1]))
    ledger = module._render_cell_figures(ref, tmp_path / "figures")

    kinds = [entry["figure_kind"] for entry in ledger]
    assert len(ledger) == 6
    assert len(set(kinds)) == 6
    assert {kind.split("_")[0] for kind in kinds} == {"density", "overhead"}
    density_entry = next(item for item in ledger if item["figure_kind"] == "density_png")
    assert density_entry["valid_n_by_pattern"] == {"Straight": 8, "Draw": 9, "Fade": 10}
    overhead_entry = next(item for item in ledger if item["figure_kind"] == "overhead_png")
    assert overhead_entry["source_kind"] == "saved_illustrative_trajectories"
    assert overhead_entry["valid_n_by_pattern"] is None


def test_pair_figure_ledger_keeps_overhead_when_common_density_support_is_missing(
    tmp_path, monkeypatch
):
    from src.tools.shot_pattern_analysis import matched_supplement_figures as figures

    first = _ref("a", "pitching_wedge", 0, 1, 1)
    second = _ref("b", "pitching_wedge", 0, 2, 1, protocol="supplement")
    for ref in (first, second):
        root = tmp_path / ref["cell"]
        cell_dir = root / ref["cell"]
        cell_dir.mkdir(parents=True)
        (cell_dir / "flight_samples.json").write_text(
            json.dumps(
                {
                    "trajectories": {
                        pattern: [[[0.0, 0.0], [1.0, 0.1]]]
                        for pattern in ("Straight", "Draw", "Fade")
                    }
                }
            )
        )
        ref["source_root"] = str(root)
    monkeypatch.setattr(
        figures,
        "_supported_endpoints",
            lambda *args: (
                {
                    pattern: [(pattern_index, 1.0, 0.0)]
                    for pattern_index, pattern in enumerate(
                        ("Straight", "Draw", "Fade")
                    )
                },
                {},
            ),
    )

    def write_overhead(*args, **kwargs):
        stem = kwargs["output_stem"]
        stem.parent.mkdir(parents=True, exist_ok=True)
        outputs = {}
        for suffix in ("png", "pdf", "svg", "social_png"):
            path = stem.with_suffix(f".{suffix}")
            path.write_text("figure")
            outputs[suffix] = path
        return outputs

    monkeypatch.setattr(figures, "_write_pair_overhead_figure", write_overhead)
    ledger = figures._render_pair_figures(
        {
            "comparison_kind": "curve_strength",
            "first": first,
            "second": second,
            "target_x_m": 1.0,
        },
        tmp_path / "report",
    )

    density = next(item for item in ledger if item["figure_kind"] == "paired_density")
    overhead = [item for item in ledger if item["figure_kind"].startswith("overhead_")]
    assert density["status"] == "unavailable"
    assert len(overhead) == 4
    assert all(
        item["source_kind"] == "saved_illustrative_trajectory_samples"
        for item in overhead
    )


def test_pipeline_writes_separate_tables_figures_and_aliases(tmp_path, monkeypatch):
    from src.tools.shot_pattern_analysis import matched_supplement_pipeline as module

    def ref(name, curve, protocol):
        config = {
            "club": {"club_id": "pitching_wedge", "shaft_lean_deg": 0.0},
            "curve_scale": curve,
            "noise": {"standard_deviations": [1.0]},
            "n_shots": 10_000,
            "seed": 7,
            "target_x_m": None,
        }
        return {
            "cell": name,
            "source_protocol": protocol,
            "execution_kind": "unique",
            "status": "completed",
            "source_root": str(tmp_path / protocol),
            "config": config,
            "summary": {"status": "completed", "target_x_m": 77.0},
            "source_plan_sha256": "plan",
            "source_cell_manifest_sha256": "manifest",
            "source_csv_sha256": name,
            "_execution": {"source_sha256": {"source.py": "same"}},
        }

    primary = ref(
        "pitching_wedge_lean0_curve1_rotation1_shaft_only",
        1.0,
        "original_primary/1",
    )
    supplement_cell = ref(
        "pitching_wedge_lean0_curve2_rotation1_shaft_only",
        2.0,
        "pw-primary-face-bounds/1",
    )
    empty_report = {
        "stage": "final",
        "planned_cells": 324,
        "counts": {"not_run": 0},
        "cells": [],
        "plan_sha256": "plan",
        "common_targets_path": str(tmp_path / "common_targets.json"),
        "common_targets_sha256": "targets",
        "common_target_provenance": {
            "acceptance_sha256": "acceptance",
            "audit_sha256": "audit",
            "pilot_plan_sha256": "pilot-plan",
            "common_targets_sha256": "targets",
        },
    }
    monkeypatch.setattr(
        module,
        "_load_original_sources",
        lambda root, protocol: (
            empty_report,
            [primary] if protocol == "original_primary" else [],
        ),
    )
    supplement = {
        "schema": "matched-supplement-report/1",
        "stage": "final",
        "execution": {"source_sha256": {"source.py": "same"}},
        "inventory_sha256": "inventory",
        "unique_cells": [supplement_cell],
        "aliases": [],
        "control_aliases": {
            "schema": "matched-study-control-aliases/1",
            "references": [],
            "status_by_alias": {},
        },
        "protocols": {},
        "protocol_counts": {},
        "unique_executed_cells": 1,
        "issue": 12000,
        "interpretation": "separate supplement",
    }
    monkeypatch.setattr(
        module, "load_supplement_sources", lambda *args, **kwargs: supplement
    )
    monkeypatch.setattr(
        module,
        "_pair_result",
        lambda pair, *args: {
            "comparison_kind": pair["comparison_kind"],
            "status": "available",
            "common_supported_n": 10_000,
            "total_n_per_population": 10_000,
            "sources": [
                {"protocol_id": "a", "cell": "a"},
                {"protocol_id": "b", "cell": "b"},
            ],
            "patterns": {},
            "course_scoring_effect_modification": {},
        },
    )
    monkeypatch.setattr(
        module,
        "_render_cell_figures",
        lambda ref, output: [
            {
                "figure_kind": "density",
                "cell": ref["cell"],
                "relative_path": "figures/cell.png",
                "sha256": "cell-plot",
            }
        ],
    )
    monkeypatch.setattr(
        module,
        "_render_pair_figures",
        lambda pair, output: [
            {
                "figure_kind": "paired-density",
                "first_cell": pair["first"]["cell"],
                "relative_path": "figures/pair.png",
                "sha256": "pair-plot",
            }
        ],
    )
    monkeypatch.setattr(module, "_sha256", lambda path: "receipt-hash")
    monkeypatch.setattr(
        module,
        "_recheck_source_inputs",
        lambda *args: None,
    )
    monkeypatch.setattr(module, "_recheck_transition_evidence", lambda *args: None)
    monkeypatch.setattr(
        module,
        "write_supplement_artifacts",
        lambda report, output: {
            "inventory": output / "REPORT_SUPPLEMENT.csv",
            "control_aliases": output / "control_aliases.json",
        },
    )
    monkeypatch.setattr(
        module,
        "write_supplement_comparisons",
        lambda results, output: {
            "csv": output / "REPORT_SUPPLEMENT_COMPARISONS.csv",
            "json": output / "REPORT_SUPPLEMENT_COMPARISONS.json",
        },
    )
    transition = tmp_path / "transition.json"
    transition.write_text("{}")
    output = tmp_path / "supplement-report"
    paths = module.build_final_supplement_report(
        original_primary_root=tmp_path / "primary",
        original_control_root=tmp_path / "control",
        protocol_roots={},
        inventory_path=tmp_path / "inventory.json",
        output_dir=output,
        transition_receipt_path=transition,
        transition_receipt_sha256="receipt-hash",
    )
    assert {"report", "comparison_markdown", "figure_ledger"}.issubset(paths)
    assert paths["original_source_inventory"].is_file()
    assert paths["comparison_markdown"].is_file()
    assert paths["figure_ledger"].is_file()
    assert not (tmp_path / "primary" / "REPORT_SUPPLEMENT.csv").exists()


@pytest.mark.slow
def test_pair_density_and_overhead_render_shared_condition_panels(
    tmp_path, monkeypatch
):
    try:
        import matplotlib.pyplot  # noqa: F401
        from PIL import Image
    except (ImportError, OSError) as exc:
        pytest.skip(f"matplotlib/Pillow not loadable: {exc}")
    import numpy as np

    from src.tools.shot_pattern_analysis.matched_supplement_figures import (
        _write_pair_density_figure,
        _write_pair_overhead_figure,
        namespace_figure_paths,
    )

    first = _ref("a", "pitching_wedge", 0, 1, 1, target=12.0)
    second = _ref("b", "pitching_wedge", 0, 2, 1, target=12.0)
    rng = np.random.default_rng(3)
    point_sets = {
        f"{pattern} · {condition}": rng.normal(size=(40, 2)).tolist()
        for pattern in ("Straight", "Draw", "Fade")
        for condition in ("First", "Second")
    }
    density_paths = _write_pair_density_figure(
        point_sets,
        first=first,
        second=second,
        comparison="curve_strength",
        target_x=12.0,
        common_n=40,
        output_stem=tmp_path / "curve-density",
    )
    trajectories = {
        f"{pattern} · {condition}": [
            [[0.0, 0.0], [6.0, 0.2], [12.0, 0.0]],
            [[0.0, 0.0], [6.0, -0.2], [12.0, 0.0]],
        ]
        for pattern in ("Straight", "Draw", "Fade")
        for condition in ("First", "Second")
    }
    overhead_paths = _write_pair_overhead_figure(
        trajectories,
        first=first,
        second=second,
        comparison="curve_strength",
        target_x=12.0,
        output_stem=tmp_path / "curve-overhead",
    )
    assert {"png", "pdf", "svg", "social_png"} <= density_paths.keys()
    assert {"png", "pdf", "svg", "social_png"} <= overhead_paths.keys()
    namespaced = {
        **namespace_figure_paths("density", density_paths),
        **namespace_figure_paths("overhead", overhead_paths),
    }
    assert len(namespaced) == 8
    assert {key.split("_", 1)[0] for key in namespaced} == {"density", "overhead"}
    with Image.open(density_paths["social_png"]) as image:
        assert image.size == (1920, 1080)
    with Image.open(overhead_paths["social_png"]) as image:
        assert image.size == (1920, 1080)
