"""Run a deterministic shot-pattern comparison without opening a GUI."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TypedDict

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


class ClubInputs(TypedDict):
    club_speed_mps: float
    loft_deg: float
    attack_angle_deg: float
    clubhead_mass_kg: float
    lie_deg: float
    shaft_lean_deg: float
    club_id: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare straight, draw, and fade shot dispersion."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_261_008)
    parser.add_argument("--face-sd", type=float, default=1.0)
    parser.add_argument("--curve-scale", type=float, default=1.0)
    parser.add_argument("--club-speed-mps", type=float)
    parser.add_argument("--loft-deg", type=float)
    parser.add_argument("--attack-angle-deg", type=float)
    parser.add_argument("--clubhead-mass-kg", type=float)
    parser.add_argument(
        "--club-preset",
        choices=("custom", "driver", "seven_iron", "pitching_wedge"),
        default=None,
        help="Use an illustrative club parameter set; explicit numeric flags override it.",
    )
    parser.add_argument(
        "--delivery-mode",
        choices=("fixed_loft", "shaft_rotation"),
        default="fixed_loft",
        help="Keep delivered loft fixed or couple face closure to shaft rotation.",
    )
    parser.add_argument("--lie-deg", "--lie", dest="lie_deg", type=float)
    parser.add_argument(
        "--shaft-lean-deg",
        "--shaft-lean",
        dest="shaft_lean_deg",
        type=float,
    )
    return parser


def resolve_club_inputs(args: argparse.Namespace) -> ClubInputs:
    """Resolve a shared illustrative preset plus explicit CLI overrides."""
    from src.tools.shot_pattern_analysis.presets import get_club_preset

    preset = (
        get_club_preset(args.club_preset)
        if args.club_preset and args.club_preset != "custom"
        else None
    )
    defaults = {
        "club_speed_mps": 45.0,
        "loft_deg": 10.9,
        "attack_angle_deg": 0.0,
        "clubhead_mass_kg": 0.2,
        "lie_deg": 58.0,
        "shaft_lean_deg": 0.0,
    }

    def resolve(field: str) -> float:
        override = getattr(args, field)
        return float(
            override
            if override is not None
            else getattr(preset, field)
            if preset
            else defaults[field]
        )

    return {
        "club_speed_mps": resolve("club_speed_mps"),
        "loft_deg": resolve("loft_deg"),
        "attack_angle_deg": resolve("attack_angle_deg"),
        "clubhead_mass_kg": resolve("clubhead_mass_kg"),
        "lie_deg": resolve("lie_deg"),
        "shaft_lean_deg": resolve("shaft_lean_deg"),
        "club_id": preset.id if preset else "custom",
    }


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if not arguments:
        return launch_gui()
    parser = build_parser()
    args = parser.parse_args(arguments)
    if args.output is None:
        parser.error("--output is required for headless CLI mode")
    from src.tools.shot_pattern_analysis.core import AnalysisConfig, run_analysis
    from src.tools.shot_pattern_analysis.reporting import export_analysis
    from src.tools.shot_pattern_analysis.scoring import score_saved_bundle
    from src.tools.shot_pattern_analysis.provenance import (
        assert_source_unchanged,
        source_snapshot,
        stamp_execution_receipt,
        write_run_start,
    )

    config = AnalysisConfig(
        n_shots=args.shots,
        seed=args.seed,
        face_sd_deg=args.face_sd,
        curve_scale=args.curve_scale,
        delivery_mode=args.delivery_mode,
        **resolve_club_inputs(args),
    )

    def progress(completed: int, total: int) -> None:
        if completed == total or completed % max(1, total // 10) == 0:
            sys.stderr.write(f"Simulated {completed}/{total} shots.\n")

    execution = source_snapshot()
    start_path = write_run_start(args.output, asdict(config), execution)
    result = run_analysis(config, progress_callback=progress)
    paths = export_analysis(result, args.output)
    scoring_path = score_saved_bundle(args.output)
    assert_source_unchanged(execution, source_snapshot())
    stamp_execution_receipt(args.output, execution)
    score_summary = json.loads(scoring_path.read_text(encoding="utf-8"))
    scoring_available = score_summary.get("status", "unavailable") == "available"
    for name, summary in result.summary.items():
        line = (
            f"{name}: aimed lateral SD {summary['aimed_lateral_sd_m']:.2f} m; "
            f"aimed target hit {summary['aimed_target_hit_fraction']:.1%}"
        )
        if scoring_available:
            mean_sg = score_summary["patterns"][name]["mean_strokes_gained"]
            scenario_id = score_summary.get("scenario_id", "historical_approach")
            line += f"; model-conditional {scenario_id} SG {mean_sg:+.3f} strokes"
        else:
            line += (
                "; model-conditional strokes gained unavailable for this configuration"
            )
        sys.stdout.write(line + "\n")
    for path in (*paths.values(), scoring_path, start_path):
        sys.stdout.write(f"{path}\n")
    return 0


def launch_gui() -> int:
    """Open the optional standalone desktop window when invoked without flags."""
    try:
        from PyQt6.QtWidgets import QApplication
        from src.tools.shot_pattern_analysis.gui import MainWidget
    except (ImportError, OSError) as exc:
        sys.stderr.write(
            "The desktop window requires the GUI extra. Install with:\n"
            "  pip install -e '.[gui-tools]'\n"
            f"Import error: {exc}\n"
        )
        return 1
    app = QApplication(sys.argv)
    widget = MainWidget()
    widget.resize(1200, 900)
    widget.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
