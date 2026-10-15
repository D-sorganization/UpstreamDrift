"""Anthro support for the NSX-26 restriction report (#12569).

Takes the ground-supported anthro frames of NSX-60 (welded club, two feet) and
asks what the same trajectory allows when the hand-club joint transmits less.
The declared feasible set (``envelope_only`` preset, ASSUMED capacity prior,
Coulomb foot cones) is identical for every kind; only the closure rows change.

Hand geometry (declared approximation).  The canonical grip origin is the
right-hand weld frame (NSX-13), so the closing hand is the right hand with
``x_R = 0``.  The left grip point is the left-hand solid's centre of mass on
the shaft (grip x) axis; the multiplier lives at the grip origin.  This is the
geometry the model carries, not a measurement of the golfer's hands.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence


import numpy as np

from src.shared.python.loop_nullspace import registry
from src.shared.python.loop_nullspace.admission import (
    AdmissionGates,
    AdmittedTrajectory,
    admit,
)
from src.shared.python.loop_nullspace.contracts import ClosedLoopSnapshot, ClosureKind
from src.shared.python.loop_nullspace.decomposition import Decomposition
from src.shared.python.loop_nullspace.envelopes import joint_torque_quantities
from src.shared.python.loop_nullspace.feasible_set import FeasibleSet
from src.shared.python.loop_nullspace.grip_split import net_club_wrench
from src.shared.python.loop_nullspace.ground_contact import foot_friction_cones
from src.shared.python.loop_nullspace.restriction import (
    QuantityBuilder,
    RestrictionReport,
    restriction_sensitivity,
    snapshot_quantities,
)
from src.shared.python.loop_nullspace.restriction_hand_split import (
    HandSplitFrame,
    hand_split_quantities,
)
from tests.integration.loop_nullspace_lower_body_support import (
    BENCHMARK_FRAMES,
    AnthroBench,
    Frame,
    build_frames,
)

KINDS = (ClosureKind.WELD6, ClosureKind.SHAFT_FREE5, ClosureKind.BALL3)
HAND_MASS_MIN_KG = 0.1  # the standoff solid (0.01 kg) is not a hand


def strided(stride: int) -> tuple[int, ...]:
    """Every ``stride``-th benchmark frame (``1`` is all 200)."""
    return BENCHMARK_FRAMES[::stride]


def admitted(bench: AnthroBench, frames: Sequence[Frame]) -> AdmittedTrajectory:
    """Admission of the decomposed frames; the position level is skipped.

    The anthro record has no closure plant here (``require_position=False``,
    recorded); velocity and acceleration levels, the time base and the stance
    feet are checked.
    """
    times = np.asarray([bench.record["time_s"][f.index] for f in frames])
    report = admit(
        [f.snapshot for f in frames],
        gates=AdmissionGates(require_position=False),
        times=times,
    )
    return AdmittedTrajectory.from_report(report)


def left_grip_x(snapshot: ClosedLoopSnapshot) -> float:
    """Shaft coordinate of the left grip point: the left-hand solid's COM."""
    hands = [h for h in snapshot.welded_hand_inertias if h.mass >= HAND_MASS_MIN_KG]
    far = max(hands, key=lambda h: abs(float(h.com[0])))
    return float(far.com[0])


def hand_frame_of(bench: AnthroBench) -> Callable[[ClosedLoopSnapshot], HandSplitFrame]:
    """Geometry and net bare-club wrench of a snapshot (right hand closes)."""

    def frame_of(snap: ClosedLoopSnapshot) -> HandSplitFrame:
        ne = bench.adapter.club_newton_euler(snap.q, snap.v, snap.a)
        net = net_club_wrench(snap, ne).values
        return HandSplitFrame(left_grip_x(snap), 0.0, net, closing_hand="R")

    return frame_of


def feasible_set_of(
    bench: AnthroBench,
) -> Callable[[Decomposition, ClosedLoopSnapshot], FeasibleSet]:
    """Declared sets shared by every kind: capacity prior, foot friction cones."""

    def build(dec: Decomposition, snap: ClosedLoopSnapshot) -> FeasibleSet:
        cones = foot_friction_cones(snap) if snap.foot_contacts else ()
        return FeasibleSet.from_hypotheses(
            dec,
            registry.preset("envelope_only"),
            capacity=bench.capacity,
            cones=cones,
        )

    return build


def report_quantity_builders(bench: AnthroBench) -> list[QuantityBuilder]:
    """Joint torques (35) and the six hand-split coordinates."""
    return [
        snapshot_quantities(joint_torque_quantities(None)),
        hand_split_quantities(hand_frame_of(bench)),
    ]


def restriction_report(
    bench: AnthroBench, indices: Sequence[int]
) -> tuple[RestrictionReport, list[Frame], dict[int, float]]:
    """The report on ``indices`` plus the frames used and the refused frames."""
    frames, refused = build_frames(bench, indices)
    traj = admitted(bench, frames)
    report = restriction_sensitivity(
        traj,
        KINDS,
        report_quantity_builders(bench),
        feasible_set_of=feasible_set_of(bench),
    )
    return report, frames, refused


def _quantity_group(name: str) -> str:
    return "joint_torque" if name.startswith("tau_") else "hand_split"


def summarise(report: RestrictionReport) -> dict[str, object]:
    """Compact per-kind record: counts, conflicts, width and sign-class changes.

    ``width_ratio`` is over the quantities bounded on both kinds; ``changed``
    counts quantity-frames whose sign class differs from the reference.
    """
    names = report.series(report.reference).names
    out: dict[str, object] = {
        "label": report.label,
        "n_frames": report.n_frames,
        "nesting_violations": len(report.nesting_violations()),
        "kinds": {},
    }
    for kind in report.kinds[1:]:
        groups: dict[str, dict[str, object]] = {}
        for group in ("joint_torque", "hand_split"):
            picked = [n for n in names if _quantity_group(n) == group]
            ratios = [report.width_ratio(kind, n) for n in picked]
            medians = [r.median for r in ratios if r.frames]
            changes: dict[str, int] = {}
            for n in picked:
                for key, count in report.sign_class_changes(kind, n).items():
                    changes[key] = changes.get(key, 0) + count
            groups[group] = {
                "quantities": len(picked),
                "quantities_with_width_data": len(medians),
                "median_width_ratio_min": min(medians) if medians else None,
                "median_width_ratio_median": float(np.median(medians))
                if medians
                else None,
                "narrower_quantity_frames": sum(r.narrower_frames for r in ratios),
                "sign_class_changes": changes,
            }
        out["kinds"][kind.value] = {  # type: ignore[index]
            "lost_frames": list(report.lost_frames(kind)),
            "groups": groups,
        }
    out["feasible"] = {k.value: v for k, v in report.feasible_counts().items()}
    out["infeasible"] = {k.value: v for k, v in report.infeasible_counts().items()}
    out["balance_inconsistent"] = {
        k.value: v for k, v in report.balance_inconsistent_counts().items()
    }
    out["conflict_hypotheses"] = {
        k.value: report.conflict_hypotheses(k) for k in report.kinds
    }
    return out
