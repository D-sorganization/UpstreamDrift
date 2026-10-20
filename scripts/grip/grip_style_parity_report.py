"""Cross-engine parity report generator for grip-style pairs (HRG-GS-G10, #13308).

Builds one ``StyleParityReport`` per style (``overlap``, ``interlock``,
``ten_finger``) across Pinocchio (reference), OpenSim (G7), MuJoCo (G8) and
Drake (G9), and writes JSON plus a Markdown table to a chosen ``--out``
directory.
"""

from __future__ import annotations

import argparse
import functools
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from src.shared.python.contracts import ensure, require
from src.shared.python.grip_contact.contact_run import ContactRun
from src.shared.python.grip_contact.contact_topology import GripStyle
from src.shared.python.grip_contact.parity import (
    DETERMINATE_QUANTITIES,
    INDETERMINACY_CITATION,
    INDETERMINATE_LABEL,
    INDETERMINATE_QUANTITIES,
    PAIR_PARITY_QUANTITIES,
    PEAK_TOLERANCE,
    RMS_TOLERANCE,
    STYLE_PARITY_QUANTITIES,
    StyleParityReport,
    build_style_parity_report,
)

DEFAULT_STYLES: tuple[GripStyle, ...] = (
    GripStyle.OVERLAP,
    GripStyle.INTERLOCK,
    GripStyle.TEN_FINGER,
)
DEFAULT_MOTIONS: tuple[str, ...] = ("hold", "swing")
ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "docs/development/full_body_models"
FIXTURES = ROOT / "tests/fixtures/club_face"
SPEC_PATH = MODELS / "full_body_spec_anthro_driver.json"
DEFAULT_SQUEEZE_N = 1100.0


@functools.lru_cache(maxsize=8)
def _load_style_inputs(
    style: GripStyle,
) -> tuple[bytes, object, object, object]:
    """Load spec_bytes, swing, GripInterface, and TopologyContactModel for ``style``."""
    from src.shared.python.grip_contact import (  # noqa: PLC0415
        GripInterface,
        build_topology_model,
        contact_topology,
        load_coordinate_swing,
    )
    from src.shared.python.motion_matching.grip_contract import (  # noqa: PLC0415
        GripSpec,
    )

    spec_bytes = SPEC_PATH.read_bytes()
    spec = json.loads(spec_bytes)
    names = spec["coordinate_order"]
    swing = load_coordinate_swing(
        FIXTURES / "swing_q_driver.npz",
        FIXTURES / "address_poses.json",
        "driver",
        names,
    )
    interface = GripInterface.from_spec(spec)
    topo = contact_topology(GripSpec(style=GripStyle(style), trigger_finger=False))
    topo_model = build_topology_model(
        interface, topo, squeeze_per_hand_n=DEFAULT_SQUEEZE_N
    )
    return spec_bytes, swing, interface, topo_model


def _import_engine_simulator(engine: str) -> object:
    """Import the ``simulate_grip_contact`` callable for ``engine``."""
    if engine == "pinocchio":
        import pinocchio  # noqa: F401, PLC0415
        from src.engines.physics_engines.pinocchio.python.grip_contact_sim import (  # noqa: PLC0415
            simulate_grip_contact as _sim_pin,
        )

        return _sim_pin
    if engine == "opensim":
        import opensim  # noqa: F401, PLC0415
        from src.engines.physics_engines.opensim.python.grip_contact_osim_sim import (  # noqa: PLC0415
            simulate_grip_contact as _sim_osim,
        )

        return _sim_osim
    if engine == "mujoco":
        import mujoco  # noqa: F401, PLC0415
        from src.engines.physics_engines.mujoco.python.grip_contact_sim import (  # noqa: PLC0415
            simulate_grip_contact as _sim_mj,
        )

        return _sim_mj
    if engine == "drake":
        import pydrake  # noqa: F401, PLC0415
        from src.engines.physics_engines.drake.python.grip_contact_sim import (  # noqa: PLC0415
            simulate_grip_contact as _sim_drake,
        )

        return _sim_drake
    raise ValueError(f"Unsupported engine {engine!r}")


def run_candidate_engine(
    engine: str,
    motion: str,
    style: GripStyle,
    duration_s: float,
) -> tuple[ContactRun | None, str | None]:
    """Execute one candidate engine or return ``(None, reason)`` if unavailable."""
    require(
        motion in ("hold", "swing"), f"motion must be hold or swing; got {motion!r}"
    )
    require(duration_s > 0.0, "duration_s must be positive")
    require(
        engine in ("opensim", "mujoco", "drake"),
        f"unsupported candidate engine {engine!r}",
    )
    try:
        sim_fn = _import_engine_simulator(engine)
    except ImportError as exc:
        return None, f"{type(exc).__name__}: {exc}"
    spec_bytes, swing, interface, topo_model = _load_style_inputs(GripStyle(style))
    run = sim_fn(  # type: ignore[operator]
        spec_bytes,
        swing,
        interface=interface,
        t_end_s=duration_s,
        hold=(motion == "hold"),
        topology_model=topo_model,
    )
    return run, None


def build_style_report_for_motion(
    style: GripStyle,
    motion: str = "swing",
    duration_s: float = 0.04,
) -> StyleParityReport:
    """Run Pinocchio reference + OpenSim, MuJoCo, Drake for one style and motion."""
    require(
        motion in ("hold", "swing"), f"motion must be hold or swing; got {motion!r}"
    )
    require(duration_s > 0.0, "duration_s must be positive")
    spec_bytes, swing, interface, topo_model = _load_style_inputs(GripStyle(style))
    pin_sim = _import_engine_simulator("pinocchio")
    ref_run: ContactRun = pin_sim(  # type: ignore[operator]
        spec_bytes,
        swing,
        interface=interface,
        t_end_s=duration_s,
        hold=(motion == "hold"),
        topology_model=topo_model,
    )
    candidate_runs: dict[str, ContactRun] = {}
    unavailable_reasons: dict[str, str] = {}
    for engine in ("opensim", "mujoco", "drake"):
        run, reason = run_candidate_engine(engine, motion, style, duration_s)
        if run is not None:
            candidate_runs[engine] = run
        if reason is not None:
            unavailable_reasons[engine] = reason
    return build_style_parity_report(
        style=GripStyle(style),
        motion=motion,
        duration_s=duration_s,
        reference_run=ref_run,
        candidate_runs=candidate_runs,
        unavailable_engines=unavailable_reasons,
    )


def render_reports_markdown(
    reports_by_motion: Mapping[str, Sequence[StyleParityReport]],
    duration_s: float,
) -> str:
    """Render a multi-motion Markdown document summarizing all style parity reports."""
    require(len(reports_by_motion) > 0, "reports_by_motion must be non-empty")
    lines = [
        "# Cross-Engine Grip-Style Parity Report (HRG-GS-G10, #13308)",
        "",
        f"- **Horizon**: `{duration_s:.3f} s`",
        f"- **Determinate peak tolerance**: `{PEAK_TOLERANCE * 100.0:.1f}%`",
        f"- **Determinate RMS tolerance**: `{RMS_TOLERANCE * 100.0:.1f}%`",
        f"- **Determinate quantities**: `{', '.join(DETERMINATE_QUANTITIES)}`",
        f"- **Pair-level quantities**: `{', '.join(PAIR_PARITY_QUANTITIES)}`",
        f"- **Indeterminate label**: `{INDETERMINATE_LABEL}`",
        f"- **Citation**: `{INDETERMINACY_CITATION}`",
        "",
    ]
    for motion, reports in reports_by_motion.items():
        lines.append(f"## Motion: `{motion}` (`{duration_s:.3f} s`)")
        lines.append("")
        for report in reports:
            lines.append(report.to_markdown())
            lines.append("")
    out = "\n".join(lines).rstrip() + "\n"
    ensure(len(out) > 0, "Markdown output must be non-empty")
    return out


def write_parity_report_bundle(
    out_dir: Path,
    reports_by_motion: Mapping[str, Sequence[StyleParityReport]],
    duration_s: float = 0.04,
) -> tuple[Path, Path]:
    """Write ``grip_style_parity_report.json`` and ``.md`` to ``out_dir``."""
    require(len(reports_by_motion) > 0, "reports_by_motion must be non-empty")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "issue": 13308,
        "duration_s": float(duration_s),
        "peak_tolerance": float(PEAK_TOLERANCE),
        "rms_tolerance": float(RMS_TOLERANCE),
        "determinate_quantities": list(DETERMINATE_QUANTITIES),
        "indeterminate_quantities": list(INDETERMINATE_QUANTITIES),
        "pair_parity_quantities": list(PAIR_PARITY_QUANTITIES),
        "style_parity_quantities": list(STYLE_PARITY_QUANTITIES),
        "indeterminate_label": INDETERMINATE_LABEL,
        "citation": INDETERMINACY_CITATION,
        "motions": {
            motion: {r.style: r.to_dict() for r in reports}
            for motion, reports in reports_by_motion.items()
        },
    }
    json_path = out_dir / "grip_style_parity_report.json"
    md_path = out_dir / "grip_style_parity_report.md"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(
        render_reports_markdown(reports_by_motion, duration_s),
        encoding="utf-8",
    )
    ensure(json_path.is_file() and md_path.is_file(), "output files must exist")
    return json_path, md_path


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for generating cross-engine grip-style parity reports."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for JSON and Markdown parity reports.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=0.04,
        help="Simulation horizon in seconds (default: 0.04).",
    )
    parser.add_argument(
        "--motions",
        nargs="+",
        choices=("hold", "swing"),
        default=list(DEFAULT_MOTIONS),
        help="Motions to evaluate (default: hold swing).",
    )
    parser.add_argument(
        "--styles",
        nargs="+",
        choices=("overlap", "interlock", "ten_finger"),
        default=[s.value for s in DEFAULT_STYLES],
        help="Grip styles to evaluate (default: overlap interlock ten_finger).",
    )
    args = parser.parse_args(argv)
    reports_by_motion: dict[str, list[StyleParityReport]] = {}
    for motion in args.motions:
        reports_by_motion[motion] = [
            build_style_report_for_motion(
                style=GripStyle(style),
                motion=motion,
                duration_s=args.duration_s,
            )
            for style in args.styles
        ]
    write_parity_report_bundle(
        args.out,
        reports_by_motion,
        duration_s=args.duration_s,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
