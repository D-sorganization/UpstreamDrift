"""Closure residual of every engine over the canned swing (OSV-2, #11728).

One report per club: MuJoCo, Drake and Pinocchio through their matching plants,
OpenSim through the native weld frames of the generated model, and MyoSuite
(whose retargeted scene needs the pinned ``myo_sim`` assets).  Each engine that
cannot supply a residual is reported unavailable with the reason.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.shared.python.contracts import require
from src.shared.python.grip_contact.closure_series import (
    GripClosureSeries,
    closure_series_from_residuals,
)
from src.shared.python.grip_contact.swing_input import load_coordinate_swing

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/club_face"
SPECS = ROOT / "docs/development/full_body_models"
OSIM_MODELS = ROOT / "src/engines/physics_engines/opensim/models/generated"
PLANT_ENGINES = ("mujoco", "drake", "pinocchio")
CLUBS = ("driver", "iron7")


def _plant_series(engine: str, spec: dict[str, Any], q: Any) -> GripClosureSeries:
    from src.shared.python.motion_matching.pipeline.plant import (  # noqa: PLC0415
        get_plant,
    )

    try:
        plant = get_plant(engine, spec)
    except Exception as exc:  # noqa: BLE001 - any build failure is "unavailable"
        return GripClosureSeries.unavailable(engine, f"{type(exc).__name__}: {exc}")
    return closure_series_from_residuals(engine, plant.closure_residuals, q)


def _myosuite_series() -> GripClosureSeries:
    from src.engines.physics_engines.myosuite.python import (  # noqa: PLC0415
        golfer_scene,
    )

    scene = golfer_scene.resolve_golfer_scene()
    if scene.is_placeholder:
        return GripClosureSeries.unavailable(
            "myosuite", "pinned myo_sim assets unavailable: placeholder scene"
        )
    return GripClosureSeries.unavailable(
        "myosuite",
        "no canned swing in the MyoSuite retargeted coordinate space is committed",
    )


def swing_closure_report(club: str) -> dict[str, dict[str, object]]:
    """Per-engine closure series documents for the canned swing of ``club``."""
    require(club in CLUBS, f"club must be one of {CLUBS}, got {club!r}")
    spec = json.loads((SPECS / f"full_body_spec_anthro_{club}.json").read_bytes())
    swing = load_coordinate_swing(
        FIXTURES / f"swing_q_{club}.npz",
        FIXTURES / "address_poses.json",
        club,
        spec["coordinate_order"],
    )
    series: list[GripClosureSeries] = [
        _plant_series(engine, spec, swing.q) for engine in PLANT_ENGINES
    ]
    from src.engines.physics_engines.opensim.python.grip_closure import (  # noqa: PLC0415
        weld_closure_series,
    )

    series.append(
        weld_closure_series(
            OSIM_MODELS / f"full_body_anthro_{club}.osim", swing.names, swing.q
        )
    )
    series.append(_myosuite_series())
    return {s.engine: s.as_document() for s in series}


def main(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON report path (OpenSim writes its own log lines to stdout)",
    )
    args = parser.parse_args(argv)
    report = {club: swing_closure_report(club) for club in CLUBS}
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
