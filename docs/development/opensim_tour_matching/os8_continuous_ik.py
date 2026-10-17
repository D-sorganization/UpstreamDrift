"""OS-8: branch-continuous capture-A IK on the calibrated golf_humanoid model.

Stock OpenSim IK on the OS-3b calibrated model flips branch after top: marker
RMS is unchanged across each flip, so it hops between equal-fit solutions of
coordinates the markers barely determine (the right arm and wrist carry only
three markers and the hand none). See #13056.

This driver re-solves the same capture with the shared sequential IK
(``solve_full_body_ik_trajectory``): each frame is warm-started from the
previous one, a weak ``reg_weight * (q - q_prev)`` residual breaks the ties,
and the model's own coordinate ranges bound the solve. It records the
per-frame marker RMS next to stock OpenSim IK so the price of the tie-break is
visible, and the largest frame-to-frame coordinate step so continuity is
checked rather than assumed.

Runs where ``import opensim`` succeeds. Kinematics only; not dynamics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.engines.physics_engines.opensim.python.tour_matching import (  # noqa: E402
    read_trc,
)
from src.engines.physics_engines.opensim.python.tour_matching.marker_continuity import (  # noqa: E402
    MarkerSetPolicy,
    constant_marker_set,
    largest_coordinate_steps,
)
from src.shared.python.motion_matching.full_body_ik import (  # noqa: E402
    solve_full_body_ik_trajectory,
)
from src.shared.python.motion_matching.trajectory_ik_options import (  # noqa: E402
    TrajectoryIkOptions,
)
from src.shared.python.motion_matching.tour_capture_contract import (  # noqa: E402
    TourCapture,
)

EVIDENCE = ROOT / "docs/development/opensim_tour_matching/evidence"
DEFAULT_MODEL = EVIDENCE / "os7_moco_g1/inputs/golf_humanoid_scaled_tour_markers.osim"
DEFAULT_TRC = EVIDENCE / "tour_average_tracked.trc"
STEP_LIMIT_RAD = 0.2
LOGGER = logging.getLogger("os8_continuous_ik")
#: RShoulderTop is missing until frame 526; its late appearance would change
#: the fitted set. The two club trios are rigid on the club.
POLICY = MarkerSetPolicy(
    rigid_groups=(
        ("Marker_2:2:1", "Marker_2:2:2", "Marker_2:2:3"),
        ("Marker_3:3:1", "Marker_3:3:2", "Marker_3:3:3"),
    ),
    dropped=("RShoulderTop",),
    max_spline_gap=10,
)
#: Unbounded coordinates (free translations) get this half-width around the seed.
OPEN_HALF_WIDTH = 10.0


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class OsimKinematics:
    """OpenSim forward kinematics for the shared IK: coordinates, poses, markers."""

    def __init__(self, model_path: Path) -> None:
        import opensim  # type: ignore[import-not-found]

        self.model = opensim.Model(str(model_path))
        self.state = self.model.initSystem()
        coords = self.model.getCoordinateSet()
        self.coords = [coords.get(i) for i in range(coords.getSize())]
        self.names = [str(c.getName()) for c in self.coords]
        self.rotational = [
            c.getMotionType() == opensim.Coordinate.Rotational for c in self.coords
        ]
        markers = self.model.getMarkerSet()
        self.offsets: dict[str, tuple[str, tuple[float, ...]]] = {}
        for i in range(markers.getSize()):
            marker = markers.get(i)
            body = str(marker.getParentFrameName()).split("/")[-1]
            loc = marker.get_location()
            self.offsets[str(marker.getName())] = (body, tuple(loc.to_numpy()))
        self.bodies = sorted({body for body, _ in self.offsets.values()})
        self.couplers = []
        constraints = self.model.getConstraintSet()
        for i in range(constraints.getSize()):
            coupler = opensim.CoordinateCouplerConstraint.safeDownCast(
                constraints.get(i)
            )
            if coupler is None or not coupler.get_isEnforced():
                continue
            independent = coupler.getIndependentCoordinateNames()
            if independent.getSize() != 1:
                raise ValueError(
                    f"{coupler.getName()}: one independent coordinate expected"
                )
            self.couplers.append(
                (
                    self.names.index(str(coupler.getDependentCoordinateName())),
                    self.names.index(str(independent.get(0))),
                    coupler.getFunction(),
                )
            )
        self._arg = opensim.Vector(1, 0.0)

    def enforce(self, q: np.ndarray) -> np.ndarray:
        """``q`` with every enforced coupler's dependent coordinate recomputed."""
        out = np.array(q, dtype=float)
        for dependent, independent, function in self.couplers:
            self._arg.set(0, float(out[independent]))
            out[dependent] = function.calcValue(self._arg)
        return out

    def bounds(self, seed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Clamped ranges where the model clamps; a wide box elsewhere."""
        low, high = seed - OPEN_HALF_WIDTH, seed + OPEN_HALF_WIDTH
        for i, coord in enumerate(self.coords):
            if coord.getDefaultClamped():
                low[i], high[i] = coord.getRangeMin(), coord.getRangeMax()
        return low, high

    def poses(self, q: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        for coord, value in zip(self.coords, self.enforce(q), strict=True):
            coord.setValue(self.state, float(value), False)
        self.model.realizePosition(self.state)
        out = {}
        for body in self.bodies:
            transform = (
                self.model.getBodySet().get(body).getTransformInGround(self.state)
            )
            rot = transform.R().asMat33()
            r = np.array([[rot.get(i, j) for j in range(3)] for i in range(3)])
            out[body] = (r, transform.p().to_numpy())
        return out

    def marker_rms(self, q: np.ndarray, cap: TourCapture) -> np.ndarray:
        rms = np.zeros(cap.frames)
        for f in range(cap.frames):
            poses = self.poses(q[f])
            err = [
                np.sum(
                    (poses[b][0] @ np.asarray(o) + poses[b][1] - cap.points_m[f, i])
                    ** 2
                )
                for i, (b, o) in enumerate(self.offsets[label] for label in cap.labels)
                if cap.valid[f, i]
            ]
            rms[f] = float(np.sqrt(np.mean(err))) if err else 0.0
        return rms


def stock_opensim_ik(
    model_path: Path, trc: Path, cap: TourCapture, kin: OsimKinematics
) -> np.ndarray:
    """Unweighted stock OpenSim IK, the baseline and the frame-0 seed (radians)."""
    import opensim  # type: ignore[import-not-found]

    with tempfile.TemporaryDirectory() as tmp:
        tool = opensim.InverseKinematicsTool()
        model = opensim.Model(str(model_path))
        model.initSystem()
        tool.setModel(model)
        tool.setMarkerDataFileName(str(trc))
        tool.setStartTime(float(cap.time_s[0]))
        tool.setEndTime(float(cap.time_s[-1]))
        tool.setResultsDir(tmp)
        out = Path(tmp) / "ik.mot"
        tool.setOutputMotionFileName(str(out))
        tasks = tool.getIKTaskSet()
        for label in cap.labels:
            task = opensim.IKMarkerTask()
            task.setName(label)
            task.setWeight(1.0)
            task.setApply(True)
            tasks.cloneAndAppend(task)
        tool.run()
        table = opensim.TimeSeriesTable(str(out))
        degrees = table.getTableMetaDataAsString("inDegrees") == "yes"
        labels = list(table.getColumnLabels())
        q = np.column_stack(
            [np.array(table.getDependentColumn(n).to_numpy()) for n in kin.names]
        )
        if not labels:
            raise ValueError("stock OpenSim IK wrote an empty motion")
    if degrees:
        q[:, kin.rotational] = np.radians(q[:, kin.rotational])
    return q


def write_sto(
    path: Path, time_s: np.ndarray, q: np.ndarray, names: list[str], model
) -> None:
    """States file in radians with ``/jointset/<joint>/<coord>/value`` columns."""
    coords = model.getCoordinateSet()
    cols = [
        f"{coords.get(n).getJoint().getAbsolutePathString()}/{n}/value" for n in names
    ]
    header = [
        "ik_states_continuous",
        "version=1",
        f"nRows={q.shape[0]}",
        f"nColumns={q.shape[1] + 1}",
        "inDegrees=no",
        "endheader",
        "\t".join(["time", *cols]),
    ]
    rows = [
        "\t".join(f"{v:.9g}" for v in (t, *row))
        for t, row in zip(time_s, q, strict=True)
    ]
    path.write_text("\n".join(header + rows) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--osim", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--trc", type=Path, default=DEFAULT_TRC)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reg-weight", type=float, default=1e-3)
    parser.add_argument("--max-nfev", type=int, default=100)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--last", type=int, default=None)
    parser.add_argument(
        "--judge-last",
        type=int,
        default=None,
        help="last frame the continuity verdict covers (default: --last)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    args.output.mkdir(parents=True, exist_ok=True)

    raw = read_trc(args.trc)
    cap, preprocessing = constant_marker_set(raw, POLICY)
    kin = OsimKinematics(args.osim)
    missing = [label for label in cap.labels if label not in kin.offsets]
    if missing:
        raise ValueError(f"model lacks markers for capture labels: {missing}")
    stock = stock_opensim_ik(args.osim, args.trc, cap, kin)

    last = cap.frames - 1 if args.last is None else min(args.last, cap.frames - 1)
    rows = np.arange(args.first, last + 1)
    time_s = cap.time_s[rows]
    window = TourCapture(
        time_s - time_s[0], cap.labels, cap.points_m[rows], cap.valid[rows]
    )
    seed = stock[rows[0]].copy()
    low, high = kin.bounds(seed)
    seed = np.clip(seed, low, high)
    options = TrajectoryIkOptions(
        reg_weight=args.reg_weight,
        max_nfev=args.max_nfev,
        coordinate_bounds=(low, high),
    )
    q = solve_full_body_ik_trajectory(
        kin.poses, kin.offsets, window, seed, options=options
    )
    q = np.vstack([kin.enforce(row) for row in q])

    rms_new = kin.marker_rms(q, window)
    rms_stock = kin.marker_rms(stock[rows], window)
    judge = rows.size if args.judge_last is None else args.judge_last - rows[0] + 1
    judged = largest_coordinate_steps(q[:judge], kin.names, kin.rotational)
    steps_new = largest_coordinate_steps(q, kin.names, kin.rotational)
    steps_stock = largest_coordinate_steps(stock[rows], kin.names, kin.rotational)
    continuous = judged[0]["step_rad"] <= STEP_LIMIT_RAD

    sto = args.output / "ik_states_continuous.sto"
    write_sto(sto, time_s, q, kin.names, kin.model)
    receipt = {
        "schema": "os8-continuous-ik/v1",
        "issue": "#13056",
        "qualification": "kinematics only; branch continuity checked, not dynamics",
        "inputs": {
            "model": args.osim.name,
            "model_sha256": sha(args.osim),
            "trc": args.trc.name,
            "trc_sha256": sha(args.trc),
        },
        "solver": {
            "method": "shared solve_full_body_ik_trajectory, bounded TRF, warm-started",
            "reg_weight": args.reg_weight,
            "max_nfev": args.max_nfev,
            "marker_weights": "uniform (the shared solver has no per-marker weights)",
            "bounds": "model range where clamped; seed +/- 10 elsewhere",
            "seed": "stock OpenSim IK at the first frame, clipped to the bounds",
        },
        "preprocessing": preprocessing,
        "frames": {
            "first": int(rows[0]),
            "last": int(rows[-1]),
            "count": int(rows.size),
        },
        "continuity": {
            "step_limit_rad": STEP_LIMIT_RAD,
            "judged_frames": [int(rows[0]), int(rows[0] + judge - 1)],
            "continuous": bool(continuous),
            "judged_largest_steps": judged,
            "largest_steps": steps_new,
            "stock_opensim_largest_steps": steps_stock,
        },
        "marker_rms_m": {
            "mean": float(rms_new.mean()),
            "max": float(rms_new.max()),
            "stock_opensim_mean": float(rms_stock.mean()),
            "stock_opensim_max": float(rms_stock.max()),
            "max_increase_over_stock": float(np.max(rms_new - rms_stock)),
        },
        "outputs": {"states": sto.name, "states_sha256": sha(sto)},
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    np.savez_compressed(
        args.output / "marker_rms.npz",
        time_s=time_s,
        rms_m=rms_new,
        stock_rms_m=rms_stock,
    )
    LOGGER.info(
        "continuity %s; marker RMS %s",
        json.dumps(receipt["continuity"]),
        json.dumps(receipt["marker_rms_m"]),
    )
    return 0 if continuous else 1


if __name__ == "__main__":
    raise SystemExit(main())
