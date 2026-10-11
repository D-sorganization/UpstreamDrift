"""Snapshot complete publication components; contact creation is not visual acceptance."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_files(
    root: Path, files: list[Path], bound: dict[str, str]
) -> dict[str, str]:
    """Check selected final files against a previously bound hash map."""
    result = {}
    for path in files:
        relative = str(path.relative_to(root))
        if not path.is_file() or relative not in bound:
            raise ValueError(f"missing or unbound completed artifact: {relative}")
        digest = sha(path)
        if digest != bound[relative]:
            raise ValueError(f"completed artifact changed: {relative}")
        result[relative] = digest
    return result


def completion_boundary(
    report: Path,
    exit_path: Path,
    sources: dict[str, str],
    pinned_start: tuple[Path, str] | None = None,
):
    """Return a source/receipt-bound terminal snapshot, or pending for a live/failed run."""
    if not exit_path.is_file():
        return None
    exit_sha = sha(exit_path)
    completed = json.loads(exit_path.read_text())
    if (
        completed.get("status") != "completed_pending_full_independent_acceptance"
        or completed.get("process_exit_code") != 0
    ):
        return None
    if pinned_start is None:
        raise ValueError("terminal review requires a pinned producer start receipt")
    start_path, start_sha = pinned_start
    if not start_path.is_file() or sha(start_path) != start_sha:
        raise ValueError("pinned producer start receipt changed")
    started = json.loads(start_path.read_text())
    if started.get("status") != "running":
        raise ValueError("unexpected producer start status")
    if any(
        completed.get(key) != value for key, value in started.items() if key != "status"
    ):
        raise ValueError("terminal record differs from pinned producer start")
    command = completed.get("command", [])
    if command.count("--output-dir") != 1 or command.index("--output-dir") + 1 >= len(
        command
    ):
        raise ValueError("completion boundary lacks one exact output argument")
    if Path(command[command.index("--output-dir") + 1]).resolve() != report.resolve():
        raise ValueError("completion boundary refers to another output directory")
    path = report / "REPORT_SUPPLEMENT_PIPELINE.json"
    if not path.is_file() or sha(path) != completed.get("pipeline_receipt_sha256"):
        raise ValueError("terminal pipeline receipt is missing or changed")
    receipt = json.loads(path.read_text())
    if sources != completed.get("producer_sha256") or sources != receipt.get(
        "postprocessing_source_sha256"
    ):
        raise ValueError("review source differs from completed producer")
    ledger_path = report / "REPORT_SUPPLEMENT_FIGURES.json"
    if not ledger_path.is_file() or sha(ledger_path) != receipt.get(
        "artifact_sha256", {}
    ).get("figure_ledger"):
        raise ValueError("completed ledger missing or changed")
    ledger = json.loads(ledger_path.read_text())
    bound = {}
    for row in ledger["figures"]:
        if row.get("status") == "unavailable":
            continue
        asset = (report / "figures" / row["relative_path"]).resolve()
        if not asset.is_relative_to(report.resolve() / "figures"):
            raise ValueError("ledger path escapes figure directory")
        relative = str(asset.relative_to(report.resolve()))
        if relative in bound:
            raise ValueError("duplicate completed ledger path")
        bound[relative] = row["sha256"]
    if sha(exit_path) != exit_sha:
        raise ValueError("completion boundary changed during read")
    return {
        "start_sha256": start_sha,
        "exit_sha256": exit_sha,
        "pipeline_receipt_sha256": sha(path),
        "ledger_sha256": sha(ledger_path),
        "asset_sha256": bound,
    }


def verify_social_caption(record: dict, group: str, text: str) -> None:
    """Check displayed identity and counts using the actual saved numeric schema."""
    flat = re.sub(r"\s+", "", text)
    protocols = (
        record["source_protocols"] if group == "pairs" else [record["source_protocol"]]
    )
    if group == "pairs" and protocols != [
        source["protocol_id"] for source in record["sources"]
    ]:
        raise ValueError("pair protocol aliases disagree with ordered source proof")
    if "Protocol:" not in flat or "|MatchedTarget:" not in flat:
        raise ValueError("missing protocol/target caption")
    protocol_text = flat.split("Protocol:", 1)[1].split("|MatchedTarget:", 1)[0]
    position = 0
    for protocol in protocols:
        pattern = r"(?<![A-Za-z0-9_./-])" + re.escape(protocol) + r"(?![A-Za-z0-9_./-])"
        match = re.search(pattern, protocol_text[position:])
        if match is None:
            raise ValueError("incorrect or unordered protocol caption")
        position += match.end()
    target = (
        record["target_x_m"] if group == "pairs" else record["summary"]["target_x_m"]
    )
    target_text = flat.split("|MatchedTarget:", 1)[1].split("|", 1)[0]
    if not re.search(
        r"(?<![0-9.])" + re.escape(f"{target:.2f}m") + r"(?![0-9.])", target_text
    ):
        raise ValueError("incorrect matched target caption")
    if group == "pairs":
        if "AllSixPopulations" not in flat:
            raise ValueError("incorrect population caption")
        count_text = flat.split("Valid/Total:", 1)[-1].split("Model-Conditional", 1)[0]
        count = f"{record['common_supported_n']:,}/{record['total_n_per_population']:,}"
        # A flattened caption places the count directly after labels such as
        # "Curve ×1". Anchor each complete labeled population instead of
        # mistaking the label's final digit for part of the count.
        population = (
            r"(?:Lean\d+(?:\.\d+)?°|Curve×\d+|RotationSD\d+(?:\.\d+)?°)"
            + re.escape(count)
            + r"valid"
        )
        if len(count_text.split(";")) != 2 or not all(
            re.fullmatch(population, part) for part in count_text.split(";")
        ):
            raise ValueError("incorrect common-six counts")
    else:
        if "IndividuallySupportedEndpointsbyPhysicalPattern" not in flat:
            raise ValueError("incorrect individual endpoint population")
        count_text = flat.split("Valid/Total:", 1)[-1].split("Model-Conditional", 1)[0]
        for pattern in ("Straight", "Draw", "Fade"):
            n = record["summary"]["summary"][pattern]["supported_n"]
            total = record["config"]["n_shots"]
            expected = f"{pattern}{n:,}/{total:,}valid"
            if expected not in count_text:
                raise ValueError("incorrect individual endpoint count")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--completion-receipt", type=Path, required=True)
    parser.add_argument("--start-receipt", type=Path, required=True)
    parser.add_argument("--start-sha256", required=True)
    parser.add_argument("--expected-numeric", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group", choices=("pairs", "cells"), required=True)
    parser.add_argument(
        "--kind", choices=("across_lean", "curve_strength", "shaft_rotation_sd")
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    if args.offset < 0 or args.limit < 1 or args.output.exists():
        raise ValueError(
            "Use a fresh evidence directory and a nonnegative bounded selection"
        )
    sys.path.insert(0, str(args.repo.resolve()))
    from src.tools.shot_pattern_analysis.matched_supplement_figures import (
        _slug,
        pair_comparison_identity,
        pair_comparison_slug,
    )
    from src.tools.shot_pattern_analysis.matched_supplement_pipeline import (
        _check_numeric_parity_reference,
        _load_numeric_parity_reference,
        _report_source_hashes,
    )

    source_before = _report_source_hashes()
    boundary = completion_boundary(
        args.report,
        args.completion_receipt,
        source_before,
        (args.start_receipt, args.start_sha256),
    )
    if boundary is None:
        args.output.mkdir(parents=True)
        (args.output / "batch_inventory.json").write_text(
            json.dumps(
                {
                    "accepted": False,
                    "status": "pending_completed_producer_boundary",
                    "assets_read": 0,
                    "script_sha256": sha(Path(__file__)),
                },
                indent=2,
            )
            + "\n"
        )
        return
    reference = _load_numeric_parity_reference(args.expected_numeric)
    if reference is None:
        raise ValueError(
            f"expected numeric parity reference could not be loaded from {args.expected_numeric}"
        )
    _check_numeric_parity_reference(reference, args.report)
    numeric = {name: sha(args.report / name) for name in reference["artifact_sha256"]}
    if args.group == "pairs":
        records = json.loads(
            (args.report / "REPORT_SUPPLEMENT_COMPARISONS.json").read_text()
        )["comparisons"]
        if args.kind:
            records = [r for r in records if r["comparison_kind"] == args.kind]
        records.sort(key=pair_comparison_identity)
        suffixes = [
            f"density_{p}_social.{e}"
            for p in ("straight", "draw", "fade")
            for e in ("png", "pdf", "svg")
        ]
        suffixes += [
            f"{k}.{e}" for k in ("density", "overhead") for e in ("png", "pdf", "svg")
        ]
        suffixes += ["overhead_social.png"]
    else:
        records = json.loads((args.report / "REPORT_SUPPLEMENT.json").read_text())[
            "unique_cells"
        ]
        records.sort(key=lambda r: (r["source_protocol"], r["cell"]))
        suffixes = [
            f"{k}.{e}"
            for k in ("density", "density_social", "overhead", "overhead_social")
            for e in ("png", "pdf", "svg")
        ]
    args.output.mkdir(parents=True)
    reviewed: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    for index, record in enumerate(
        records[args.offset : args.offset + args.limit], args.offset
    ):
        stem = (
            pair_comparison_slug(record)
            if args.group == "pairs"
            else _slug(record["cell"])
        )
        directory = "comparisons" if args.group == "pairs" else "cells"
        files = [args.report / "figures" / directory / f"{stem}_{s}" for s in suffixes]
        hashes = verify_files(args.report, files, boundary["asset_sha256"])
        extracted = {}
        for p in files:
            if p.suffix == ".svg":
                text = " ".join(
                    html.unescape(t)
                    for t in re.findall(r"<!-- (.*?) -->", p.read_text(), re.DOTALL)
                )
                extracted[p.name] = " ".join(text.split())
                # Social density captions carry exact source and physical-target identity.
                if "density" in p.name and "social" in p.name:
                    verify_social_caption(record, args.group, text)
        pngs = [p for p in files if p.suffix == ".png"]
        contacts = []
        # Only two charts per large contact: preserve useful scientific label size.
        for start in range(0, len(pngs), 2):
            canvas = Image.new("RGB", (3000, 1800), "white")
            draw = ImageDraw.Draw(canvas)
            for j, p in enumerate(pngs[start : start + 2]):
                with Image.open(p) as image:
                    image.load()
                    image.thumbnail((1480, 1690))
                    canvas.paste(
                        image.convert("RGB"), (j * 1500 + (1500 - image.width) // 2, 75)
                    )
                draw.text(
                    (j * 1500 + 10, 10),
                    f"{index}: {p.name[len(stem) + 1 :]}",
                    fill="black",
                    font=font,
                )
            contact = args.output / f"contact_{index:03}_{start // 2 + 1}.jpg"
            canvas.save(contact, quality=96)
            contacts.append({"path": contact.name, "sha256": sha(contact)})
        verify_files(args.report, files, hashes)
        reviewed.append(
            {
                "index": index,
                "record": record,
                "stem": stem,
                "asset_sha256": hashes,
                "svg_text": extracted,
                "contacts": contacts,
                "visual_review": "pending",
            }
        )
    if source_before != _report_source_hashes():
        raise ValueError("reviewed source changed")
    verify_files(args.report, [args.report / name for name in numeric], numeric)
    if boundary != completion_boundary(
        args.report,
        args.completion_receipt,
        source_before,
        (args.start_receipt, args.start_sha256),
    ):
        raise ValueError("completion boundary changed during review")
    _check_numeric_parity_reference(reference, args.report)
    receipt = {
        "scope": "Provisional Complete-Component File Snapshot; Not Visual or Whole-Publication Acceptance",
        "accepted": False,
        "producer_sources": source_before,
        "completion_boundary": {
            k: v for k, v in boundary.items() if k != "asset_sha256"
        },
        "script_sha256": sha(Path(__file__)),
        "numeric_sha256": numeric,
        "expected_reference": reference,
        "group": args.group,
        "kind": args.kind,
        "offset": args.offset,
        "limit": args.limit,
        "selected": reviewed,
        "pending": pending,
        "final_manifest_reconciliation": "pending",
    }
    (args.output / "batch_inventory.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    sys.stdout.write(
        json.dumps(
            {"complete_components": len(reviewed), "pending_components": len(pending)}
        )
    )


if __name__ == "__main__":
    main()
