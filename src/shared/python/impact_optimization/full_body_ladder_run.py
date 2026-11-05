"""Real-run entry point for the full-body speed ladder (SHOW-C3c 3b-2, #14559).

Loads the committed showpiece v2 spec (A2 grip) and the driver_v2 reference
bundle, builds a search window (projection off, 4 substeps; measured within
1e-5 of the default policy) and a full-fidelity window (``DEFAULT_POLICY``),
runs the guarded ladder over a documented joint subset, re-plays every rung at
full fidelity and writes the verdict.  The ladder is not executed by the tests
and not by CI: it is held until A6 (#14555) lands (supervisor D24).  Nothing
here claims 125 mph; an unreached rung is reported as not reached.

The baseline-physical guard (``require_physical_baseline``) refuses a spec
whose recorded efforts do not give a physical impact (an unrefit v2 spec).

Run ``python -m src.shared.python.impact_optimization.full_body_ladder_run
--out <dir outside the repo>``.

Design by Contract:
    Preconditions: every controlled joint is a plant coordinate; outputs go
    outside the repo (arrays never get committed).
    Postconditions: ``summary.json`` and ``report.md`` exist in ``--out``.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.shared.python.contracts import require
from src.shared.python.impact_optimization.speed_continuation import LadderSpec

logger = logging.getLogger(__name__)

__all__ = [
    "CONTROLLED_JOINTS",
    "ladder_spec",
    "main",
    "require_outside_repo",
    "select_controlled",
]

_REPO = Path(__file__).resolve().parents[4]
BUNDLE = (
    _REPO
    / "docs/development/full_body_models/evidence/trunk_visuals/driver_v2_bundle.npz"
)
#: Lumbar and thoracic axial rotation, neck axial, and the lead (left) arm chain.
#: The pelvis (HipInput*) is a root coordinate: fixed, never controlled.
CONTROLLED_JOINTS = frozenset(
    {
        "lumbar_axial_rotation",
        "thoracic_axial_rotation",
        "NeckInputZ",
        "LScapInputX",
        "LScapInputY",
        "LSInputX",
        "LSInputY",
        "LSInputZ",
        "LEInput",
        "LFInput",
        "LWInputX",
        "LWInputY",
    }
)
WINDOW_START, WINDOW_STEPS = 1225, 100  # impact is near step 1325 of the bundle


def select_controlled(order: Sequence[str]) -> tuple[str, ...]:
    """The controlled joints in plant order; every one must exist."""
    missing = sorted(CONTROLLED_JOINTS.difference(order))
    require(not missing, f"controlled joints missing from the plant: {missing}")
    return tuple(j for j in order if j in CONTROLLED_JOINTS)


def ladder_spec() -> LadderSpec:
    """Rungs from 105 mph up to the firm 125 mph target (never relabelled)."""
    return LadderSpec(105.0, 125.0, 5.0)


def require_outside_repo(path: Path) -> None:
    """Batch outputs (arrays, renders) never go inside the repository tree."""
    target = Path(path).resolve()
    require(
        not target.is_relative_to(_REPO),
        "outputs must live outside the repo",
        str(target),
    )


def _report_text(
    run: Any,
    envelope: Any,
    order: tuple[str, ...],
    subject: Any,
    notes: dict[str, str],
) -> str:  # pragma: no cover - part of the held real run (D24)
    """Rung table plus every torque binding with its class (D29, D32, D34)."""
    from src.shared.python.impact_optimization.continuation_report import (
        render_continuation_report,
    )
    from src.shared.python.impact_optimization.full_body_rollout import (
        envelope_subset,
    )
    from src.shared.python.impact_optimization.speed_continuation import (
        binding_report,
        source_band_from_capacity,
    )
    from src.shared.python.loop_nullspace.capacity import capacity_for

    sub = envelope_subset(envelope, run.controlled)
    table = render_continuation_report(
        run.result,
        run.kinematic_by_rung,
        notes,
        run.held,
        fidelity_note="rungs searched with projection off, 4 substeps; verdict "
        "from the default-policy replay",
        assumed_limits=sub.assumed_limits(),
    )
    band = source_band_from_capacity(capacity_for(order, subject), sub)
    lines = binding_report(run.result, sub, notes, band)
    bullets = "".join(f"- {ln}\n" for ln in lines)
    return f"{table}\nTorque bindings (class named on each line):\n{bullets}"


def _run(
    out: Path, baseline_only: bool = False, default_policy: bool = False
) -> int:  # pragma: no cover - the real run is held (D24)
    import mujoco
    import numpy as np

    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )
    from src.shared.python.impact_optimization.full_body_ladder import (
        BaselineBand,
        make_mujoco_impact_fn,
        run_full_body_ladder,
    )
    from src.shared.python.impact_optimization.full_body_rollout import (
        RolloutWindow,
    )
    from src.shared.python.impact_optimization.ladder_verification import (
        summarise_ladder,
        verify_at_full_fidelity,
        write_summary,
    )
    from src.shared.python.impact_optimization.physiological_envelope import (
        DEFAULT_LIMITS_PATH,
        PhysiologicalEnvelope,
        load_limits,
    )
    from src.shared.python.impact_optimization.trajectory_optimiser import (
        OptimiserConfig,
    )
    from src.shared.python.impact_parameters import TargetFrame
    from src.shared.python.impact_parameters.adapters import club_face_spec_from_spec
    from src.shared.python.loop_nullspace.capacity import CapacitySubject
    from src.shared.python.motion_matching.full_body_v2_showpiece import SHOWPIECE_V2
    from src.shared.python.motion_matching.same_input.bundle import InputBundle
    from src.shared.python.motion_matching.same_input.integrator import (
        DEFAULT_POLICY,
        StepPolicy,
    )
    from src.shared.python.motion_matching.same_input.plant import VectorPlant

    spec_bytes = SHOWPIECE_V2.read_bytes()
    spec = json.loads(spec_bytes)
    bundle = InputBundle.load(BUNDLE)
    xml, meta = export_full_body_mjcf(spec_bytes)
    model = mujoco.MjModel.from_xml_string(xml)
    order = tuple(meta["coordinate_order"])
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in order]
    site = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, meta["frame_sites"]["Clubhead"]
    )
    body = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_BODY, int(model.site_bodyid[site])
    )
    impact_fn = make_mujoco_impact_fn(
        model, ids, body, TargetFrame(), club_face_spec_from_spec(spec)
    )
    a = WINDOW_START
    plant = VectorPlant("mujoco", spec_bytes)

    def window(policy: StepPolicy) -> RolloutWindow:
        return RolloutWindow(
            plant,
            np.asarray(bundle.reference_q[a], dtype=float),
            np.asarray(bundle.reference_v[a], dtype=float),
            np.asarray(bundle.efforts[a : a + WINDOW_STEPS], dtype=float),
            bundle.dt_s,
            impact_fn,
            policy,
        )

    joints = select_controlled(order)
    subject = CapacitySubject(mass_kg=78.0, stature_m=1.71)
    envelope = PhysiologicalEnvelope.from_spec(
        spec, subject, load_limits(DEFAULT_LIMITS_PATH)
    )
    if baseline_only:
        policy = (
            DEFAULT_POLICY if default_policy else StepPolicy(substeps=4, project=False)
        )
        return _baseline(window(policy), joints)
    run = run_full_body_ladder(
        window(StepPolicy(substeps=4, project=False)),
        joints,
        envelope,
        band=BaselineBand(),
        trust_radius=0.5,
        config=OptimiserConfig(),
        ladder=ladder_spec(),
    )
    checks = verify_at_full_fidelity(run, window(DEFAULT_POLICY), envelope)
    notes = dict.fromkeys(
        (*envelope.assumed_torque_slots, *envelope.assumed_velocity_slots),
        "assumed limit slot (not a measured value)",
    )
    summary = summarise_ladder(run, checks, notes)
    out.mkdir(parents=True, exist_ok=True)
    write_summary(summary, out / "summary.json")
    report = _report_text(run, envelope, order, subject, notes)
    (out / "report.md").write_text(report, encoding="utf-8")
    logger.info("%s", summary["claim_top_rung"])
    return 0


def _baseline(search: Any, joints: tuple[str, ...]) -> int:  # pragma: no cover
    """Roll out the recorded baseline once and report the physical-baseline guard.

    No ladder runs: this only shows whether the spec gives a physical impact.
    """
    from src.shared.python.impact_optimization.full_body_ladder import (
        BaselineBand,
        require_physical_baseline,
    )
    from src.shared.python.impact_optimization.full_body_rollout import (
        make_full_body_rollout,
    )

    impact = make_full_body_rollout(search, joints)(
        search.baseline_torque(joints)
    ).impact
    logger.info(
        "baseline: %.1f mph, face %s, path %s, aoa %.2f",
        impact.speed_mph,
        impact.face_deg,
        impact.path_deg,
        impact.aoa_deg,
    )
    require_physical_baseline(impact, BaselineBand())
    logger.info("baseline is physical")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``--out`` is a directory outside the repository."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="build the envelope and check the baseline only; no ladder",
    )
    parser.add_argument(
        "--default-policy",
        action="store_true",
        help="with --baseline-only: roll out with DEFAULT_POLICY, not the search policy",
    )
    args = parser.parse_args(argv)
    require_outside_repo(args.out)
    return _run(args.out, args.baseline_only, args.default_policy)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
