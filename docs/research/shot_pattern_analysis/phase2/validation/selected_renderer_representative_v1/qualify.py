"""Representative saved-data renderer comparison; run only after source freeze."""

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest.mock
import numpy as np

MAIN = Path("/home/dieterolson/.codex/worktrees/shot-pattern-analysis/UpstreamDrift")
CANDIDATE = Path(
    "/home/dieterolson/.codex/worktrees/shot-pattern-presentation/UpstreamDrift"
)
BASELINE = Path(
    "/home/dieterolson/.codex/worktrees/shot-pattern-final-layout/UpstreamDrift"
)
PHASE = MAIN / "docs/research/shot_pattern_analysis/phase2"
KINDS = frozenset(
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


def digest(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def source_map(root):
    return {
        str(p.relative_to(root)): digest(p)
        for folder in [
            "src/tools/shot_pattern_analysis",
            "scripts/research/shot_pattern_analysis",
        ]
        for p in (root / folder).rglob("*.py")
    }


def data_value(value):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return data_value(dataclasses.asdict(value))
    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        return {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): data_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [data_value(v) for v in value]
    return value


def prepare(out):
    sys.path.insert(0, str(CANDIDATE))
    from scripts.research.shot_pattern_analysis.supplement_revision_adapter import (
        reconstruct_pair,
    )

    comparison_path = PHASE / "supplement_report_v1/REPORT_SUPPLEMENT_COMPARISONS.json"
    report_path = PHASE / "supplement_report_v1/REPORT_SUPPLEMENT.json"
    comparisons = json.loads(comparison_path.read_text())
    report = json.loads(report_path.read_text())
    roots = {
        "original_primary/1": PHASE / "final_v1",
        "original_control/1": PHASE / "control_final_v1",
        "pw-primary-face-bounds/1": PHASE / "supplement_final_v1/pw_primary_face",
        "pw-control-face-bounds/1": PHASE / "supplement_final_v1/pw_control_face",
        "driver-control-speed-bounds/1": PHASE
        / "supplement_final_v1/driver_control_speed",
    }
    selectors = [
        ("pw_narrow", "across_lean", "pitching_wedge", "shaft_only"),
        ("iron_narrow", "across_lean", "seven_iron", "shaft_only"),
        ("pw_covariance", "curve_strength", "pitching_wedge", "cov_facepath_negative"),
        ("driver_fixed", "across_lean", "driver", "combined2"),
    ]
    selected = []
    inputs = {
        str(comparison_path): digest(comparison_path),
        str(report_path): digest(report_path),
    }
    for label, kind, club, profile in selectors:
        matches = [
            r
            for r in comparisons["comparisons"]
            if r["comparison_kind"] == kind
            and all(
                s["cell"].startswith(club + "_") and s["cell"].endswith("_" + profile)
                for s in r["sources"]
            )
        ]
        if label == "pw_covariance":
            matches = [
                r
                for r in matches
                if any(
                    s["protocol_id"] == "pw-primary-face-bounds/1" for s in r["sources"]
                )
            ]
        if label == "driver_fixed":
            matches = [
                r
                for r in matches
                if any(
                    s["protocol_id"] == "driver-control-speed-bounds/1"
                    for s in r["sources"]
                )
            ]
        assert matches, label
        row = sorted(matches, key=lambda r: tuple(s["cell"] for s in r["sources"]))[0]
        snapshots = {}
        for proof in row["sources"]:
            root = roots[proof["protocol_id"]]
            directory = root / proof["cell"]
            manifest = directory / "manifest.json"
            m = json.loads(manifest.read_text())
            summary = json.loads((directory / "summary.json").read_text())
            assert digest(manifest) == proof["source_cell_manifest_sha256"]
            assert digest(root / "matrix_plan.json") == proof["source_plan_sha256"]
            inputs[str(manifest)] = digest(manifest)
            inputs[str(root / "matrix_plan.json")] = digest(root / "matrix_plan.json")
            for name, sha in m["files_sha256"].items():
                assert digest(directory / name) == sha
                inputs[str(directory / name)] = sha
            snapshots[f"{proof['protocol_id']}:{proof['cell']}"] = {
                "source_root": str(root),
                "status": summary["status"],
                "config": summary["config"],
                "manifest_sha256": digest(manifest),
                "summary_sha256": m["files_sha256"]["summary.json"],
                "csv_sha256": m["files_sha256"]["shots.csv.gz"],
            }
        a, b = row["sources"]
        record = {
            "comparison_kind": kind,
            "first_protocol": a["protocol_id"],
            "second_protocol": b["protocol_id"],
            "first_cell": a.get("alias_cell") or a["cell"],
            "second_cell": b.get("alias_cell") or b["cell"],
            "target_x_m": row["target_x_m"],
            "source_kind": "six_population_common_index_comparison",
            "common_supported_n": row["common_supported_n"],
            "total_n_per_population": row["total_n_per_population"],
        }
        pair = reconstruct_pair(
            {
                "report": report,
                "comparisons": comparisons,
                "receipt": {"source_cells_sha256": snapshots},
            },
            record,
        )
        selected.append({"label": label, "pair": pair, "expected": record})
    (out / "pairs.json").write_text(json.dumps(selected, indent=2) + "\n")
    return inputs


def worker(root, mode, out, pairs_path):
    sys.path.insert(0, str(root))
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from src.tools.shot_pattern_analysis import matched_supplement_figures as module

    assert (
        Path(module.__file__).resolve()
        == root / "src/tools/shot_pattern_analysis/matched_supplement_figures.py"
    )
    original_save = Figure.savefig
    original_density = module.shared_histogram_density
    captures = {}
    current = {}
    seen = set()

    def density(*args, **kwargs):
        result = original_density(*args, **kwargs)
        current["density_recipe"] = data_value(result)
        return result

    module.shared_histogram_density = density
    # Baseline ordinary path skips the nine unchanged social-density exports.
    module.write_pair_social_density = lambda *a, **k: {}

    def save(fig, path, *args, **kwargs):
        path = Path(path)
        kind = (
            "overhead_social"
            if path.stem.endswith("_overhead_social")
            else ("overhead" if path.stem.endswith("_overhead") else "density")
        )
        key = (current["label"], kind)
        if key not in seen:
            seen.add(key)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            axes = []
            for ax in fig.axes:
                axes.append(
                    {
                        "xlim": list(ax.get_xlim()),
                        "ylim": list(ax.get_ylim()),
                        "lines": [
                            {
                                "x": data_value(np.asarray(line.get_xdata())),
                                "y": data_value(np.asarray(line.get_ydata())),
                            }
                            for line in ax.lines
                        ],
                        "collections": [
                            {
                                "array": data_value(np.asarray(c.get_array()))
                                if c.get_array() is not None
                                else None,
                                "offsets": data_value(np.asarray(c.get_offsets())),
                                "clim": list(c.get_clim()),
                            }
                            for c in ax.collections
                        ],
                    }
                )
            texts = []
            for text in fig.findobj(matplotlib.text.Text):
                if text.get_visible() and text.get_text():
                    box = text.get_window_extent(renderer)
                    texts.append(
                        {
                            "text": text.get_text(),
                            "bbox": list(box.bounds),
                            "font_pt": text.get_fontsize(),
                            "outside_canvas": box.x0 < -0.5
                            or box.y0 < -0.5
                            or box.x1 > fig.bbox.width + 0.5
                            or box.y1 > fig.bbox.height + 0.5,
                        }
                    )
            current.setdefault("figures", {})[kind] = {
                "axes": axes,
                "texts": texts,
                "canvas_px": list(fig.bbox.bounds),
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        if mode == "baseline":
            path.write_bytes(b"BASELINE CAPTURE ONLY - NOT A GRAPHIC")
        else:
            original_save(fig, path, *args, **kwargs)

    with unittest.mock.patch.object(Figure, "savefig", new=save):
        for item in json.loads(pairs_path.read_text()):
            current = {"label": item["label"]}
            directory = out / item["label"]
            directory.mkdir(parents=True)
            if mode == "baseline":
                ledger = module._render_pair_figures(item["pair"], directory)
            else:
                ledger = module.render_selected_pair_figures(
                    item["pair"], directory, figure_kinds=KINDS
                )
            assert len(ledger) == 7 and {r["figure_kind"] for r in ledger} == KINDS
            actual = {
                str(p.relative_to(directory))
                for p in directory.rglob("*")
                if p.is_file()
            }
            assert actual == {r["relative_path"] for r in ledger}
            for record in ledger:
                assert record["target_x_m"] == item["expected"]["target_x_m"]
                for key in [
                    "first_cell",
                    "second_cell",
                    "first_protocol",
                    "second_protocol",
                ]:
                    assert record[key] == item["expected"][key]
                if record["source_kind"] == "six_population_common_index_comparison":
                    assert (
                        record["common_supported_n"]
                        == item["expected"]["common_supported_n"]
                    )
                    assert (
                        record["total_n_per_population"]
                        == item["expected"]["total_n_per_population"]
                    )
                assert digest(directory / record["relative_path"]) == record["sha256"]
            current["ledger"] = ledger
            captures[item["label"]] = current
    (out / "capture.json").write_text(json.dumps(data_value(captures), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=["baseline", "candidate"])
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.root, args.worker, args.out, args.pairs)
        return
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources = {str(r): source_map(r) for r in [CANDIDATE, BASELINE]}
    inputs = prepare(out)
    for mode, root in [("baseline", BASELINE), ("candidate", CANDIDATE)]:
        directory = out / mode
        directory.mkdir()
        with (out / f"{mode}.txt").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    mode,
                    "--root",
                    str(root),
                    "--out",
                    str(directory),
                    "--pairs",
                    str(out / "pairs.json"),
                ],
                env={**os.environ, "MPLBACKEND": "Agg", "QT_QPA_PLATFORM": "offscreen"},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    baseline = json.loads((out / "baseline/capture.json").read_text())
    candidate = json.loads((out / "candidate/capture.json").read_text())
    for label in baseline:
        assert (
            baseline[label]["density_recipe"] == candidate[label]["density_recipe"]
        ), (label, "density recipe")
        for kind in baseline[label]["figures"]:
            assert (
                baseline[label]["figures"][kind]["axes"]
                == candidate[label]["figures"][kind]["axes"]
            ), (label, kind, "artist arrays/limits")

        def clean(ledger):
            return [{k: v for k, v in r.items() if k != "sha256"} for r in ledger]

        assert clean(baseline[label]["ledger"]) == clean(candidate[label]["ledger"]), (
            label,
            "ledger",
        )
    assert sources == {str(r): source_map(r) for r in [CANDIDATE, BASELINE]}
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    receipt = {
        "scope": "Representative Selected Renderer Component Only",
        "source_sha256": sources,
        "input_sha256": inputs,
        "script_sha256": digest(Path(__file__)),
        "pairs": list(candidate),
        "artist_and_density_parity": True,
        "exact_seven_outputs_per_pair": True,
        "output_sha256": {
            str(p.relative_to(out)): digest(p)
            for p in (out / "candidate").rglob("*")
            if p.is_file()
        },
        "visual_acceptance": False,
        "parent_final_acceptance": False,
    }
    (out / "qualification.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
