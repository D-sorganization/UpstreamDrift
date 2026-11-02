"""Write OpenSim ``net_moment_com`` diagnosis JSON for issue #13560.

When OpenSim is installed, runs Pinocchio reference + OpenSim candidate for each
failing style/motion and records decomposed residual traces.  Otherwise writes the
xfail baseline summary so local OpenSim lanes can refresh the artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.shared.python.grip_contact.contact_topology import GripStyle
from src.shared.python.grip_contact.opensim_net_moment_diagnosis import (
    DIAGNOSIS_DURATION_S,
    diagnosis_case_from_runs,
    diagnosis_payload,
    write_diagnosis_json,
    xfail_baseline_cases,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = (
    ROOT
    / "docs/development/full_body_models/evidence/grip_kinetics"
    / "opensim_net_moment_diagnosis_13560.json"
)

_FAILING: tuple[tuple[GripStyle, str], ...] = (
    (GripStyle.OVERLAP, "hold"),
    (GripStyle.INTERLOCK, "hold"),
    (GripStyle.OVERLAP, "swing"),
    (GripStyle.INTERLOCK, "swing"),
    (GripStyle.TEN_FINGER, "swing"),
)


def _opensim_importable() -> bool:
    try:
        import opensim  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


def _run_live_cases(duration_s: float) -> list[dict]:
    from scripts.grip.grip_style_parity_report import (  # noqa: PLC0415
        _load_style_inputs,
        run_candidate_engine,
    )
    from src.shared.python.grip_contact.club_dynamics import ClubDynamics  # noqa: PLC0415

    spec = json.loads(
        (
            ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
        ).read_bytes()
    )
    club = ClubDynamics.from_spec(spec)
    cases: list[dict] = []
    pin_sim = __import__(
        "src.engines.physics_engines.pinocchio.python.grip_contact_sim",
        fromlist=["simulate_grip_contact"],
    ).simulate_grip_contact
    import pinocchio  # noqa: F401, PLC0415

    for style, motion in _FAILING:
        spec_bytes, swing, interface, topo_model = _load_style_inputs(style)
        ref_run = pin_sim(
            spec_bytes,
            swing,
            interface=interface,
            t_end_s=duration_s,
            hold=(motion == "hold"),
            topology_model=topo_model,
        )
        cand_run, reason = run_candidate_engine("opensim", motion, style, duration_s)
        if cand_run is None:
            raise RuntimeError(
                f"OpenSim unavailable for {style.value} {motion}: {reason}"
            )
        cases.append(
            diagnosis_case_from_runs(
                style=style,
                motion=motion,
                duration_s=duration_s,
                reference=ref_run,
                candidate=cand_run,
                club=club,
                interface=interface,
            )
        )
    return cases


def build_payload(*, duration_s: float, live: bool) -> dict:
    if live:
        cases = _run_live_cases(duration_s)
        generated_by = "opensim_net_moment_diagnosis.py (live OpenSim)"
        opensim_available = True
    else:
        cases = list(xfail_baseline_cases())
        generated_by = (
            "opensim_net_moment_diagnosis.py (xfail baseline; refresh on OpenSim lane)"
        )
        opensim_available = False
    return diagnosis_payload(
        cases,
        duration_s=duration_s,
        generated_by=generated_by,
        opensim_available=opensim_available,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Path for the diagnosis JSON artifact",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=DIAGNOSIS_DURATION_S,
        help="Parity horizon in seconds",
    )
    parser.add_argument(
        "--require-opensim",
        action="store_true",
        help="Fail when OpenSim is not importable instead of writing baselines",
    )
    args = parser.parse_args()
    if args.duration_s <= 0.0:
        raise SystemExit("--duration-s must be positive")
    live = _opensim_importable()
    if args.require_opensim and not live:
        raise SystemExit("OpenSim is not importable in this environment")
    payload = build_payload(duration_s=args.duration_s, live=live)
    write_diagnosis_json(args.out, payload)


if __name__ == "__main__":
    main()
