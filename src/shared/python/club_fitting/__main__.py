"""``python3 -m src.shared.python.club_fitting``: same-hand-motion flex sweep.

Runs one bundle's grip motion through each shaft flex and writes
``flex_sweep.json`` (rows and bend traces) and ``flex_sweep.csv``; with
``--report`` also the shaft report pages and tables (:mod:`shaft_report`) and with
``--comparison`` the flex comparison sheet (:mod:`flex_comparison`). The golfer does not adapt to the
shaft, so the rows are a groundwork answer, not a speed-potential model.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import sys

from src.shared.python.club_fitting.flex_sweep import FlexRow, run_flex_sweep

logger = logging.getLogger(__name__)
OUTPUT_STEM = "flex_sweep"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python3 -m src.shared.python.club_fitting")
    p.add_argument("--bundle", type=Path, required=True, help="same-input bundle .npz")
    p.add_argument("--club", default="driver", help="club of the bundle")
    p.add_argument(
        "--flexes",
        default=None,
        help="comma-separated ratings (default: every known rating)",
    )
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--report", action="store_true", help="also write the report")
    p.add_argument(
        "--report-settings", type=Path, default=None, help="shaft report settings"
    )
    p.add_argument(
        "--comparison", action="store_true", help="also write the flex comparison"
    )
    p.add_argument(
        "--comparison-settings", type=Path, default=None, help="comparison settings"
    )
    return p


def write_outputs(rows: list[FlexRow], out_dir: Path) -> tuple[Path, Path]:
    """Write ``flex_sweep.json`` and ``flex_sweep.csv`` into ``out_dir``."""
    if not rows:
        raise ValueError("no rows to write")
    out_dir.mkdir(parents=True, exist_ok=True)
    records = [r.record() for r in rows]
    json_path = out_dir / f"{OUTPUT_STEM}.json"
    traces = {r.flex: r.bend.to_dict() for r in rows if r.bend is not None}
    flights = {r.flex: r.flight.to_dict() for r in rows if r.flight is not None}
    payload = {"rows": records, "traces": traces, "flights": flights}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = out_dir / f"{OUTPUT_STEM}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return json_path, csv_path


def _write_comparison(json_path: Path, out: Path, settings: Path | None) -> list:
    from src.shared.python.club_fitting import flex_comparison as fc
    from src.shared.python.club_fitting.shaft_report import load_sweep_result

    cfg = fc.load_comparison_settings(settings)
    return fc.write_comparison(load_sweep_result(json_path), out, cfg)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    from src.shared.python.club_fitting.bundle_input import load_bundle_shot

    flexes = (
        None if args.flexes is None else [f.strip() for f in args.flexes.split(",")]
    )
    loaded = load_bundle_shot(args.bundle, args.club, args.out)
    rows = run_flex_sweep(
        loaded.track,
        loaded.club,
        loaded.head,
        loaded.impact_time_s,
        flexes,
        shot=loaded.shot,
    )
    json_path, csv_path = write_outputs(rows, args.out)
    written = [json_path, csv_path]
    if args.report:
        from src.shared.python.club_fitting import shaft_report as sr

        settings = sr.load_report_settings(args.report_settings)
        written += sr.write_report(sr.load_sweep_result(json_path), args.out, settings)
    if args.comparison:
        written += _write_comparison(json_path, args.out, args.comparison_settings)
    for path in written:
        logger.info("wrote %s", path.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
