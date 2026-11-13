"""NSX-94 cross-engine co-contraction trade-off evidence (issue #12723).

Builds a JSON summary of :func:`cocontraction_sweep` on the constructed OpenSim
Millard elbow and the MuJoCo-muscle (MyoSuite-equivalent) antagonist elbow used
in ``tests/unit/engines/*/test_muscle_mechanics_*.py``.  The committed artifact
is validated in CI without engine installs; regeneration skips cleanly when an
engine is missing.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.loop_nullspace.muscle_cocontraction import (
    NON_CLAIMS,
    SweepConfig,
    SweepStatus,
    CocontractionSweep,
    cocontraction_sweep,
)
from src.shared.python.loop_nullspace.muscle_fiber import total_activation
from src.shared.python.loop_nullspace.muscle_frame import MuscleFrame
from src.shared.python.loop_nullspace.muscle_noise import muscle_noise_spec
from src.shared.python.loop_nullspace.muscle_stiffness import StiffnessModel

__all__ = [
    "DEFAULT_EVIDENCE_PATH",
    "build_tradeoff_report",
    "load_tradeoff_report",
    "validate_tradeoff_report",
    "write_tradeoff_report",
]

_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_EVIDENCE_PATH = (
    _REPO_ROOT
    / "docs/research/loop_nullspace/evidence/nsx94_cocontraction_tradeoff.json"
)
_SCHEMA = "nsx94_cocontraction_tradeoff/v1"
_REQUIRED_ENGINES = frozenset({"opensim", "myosuite"})
_KAPPA = 0.05
_DT = 1e-3
_N_LEVELS = 7
_LINEARITY_LIMIT = 0.10
_OPENSIM_TORQUE_NM = 15.0
_MYOSUITE_TORQUE_NM = 20.0

# Antagonist MJCF from tests/unit/engines/mujoco/test_muscle_mechanics_mujoco.py
_MJCF_ELBOW = """
<mujoco>
  <option gravity="0 0 0" timestep="0.001"/>
  <worldbody>
    <body name="upper">
      <geom type="capsule" fromto="0 0 0 0 -0.3 0" size="0.02" mass="1"/>
      <site name="o_flex" pos="0.03 -0.05 0"/>
      <site name="o_ext" pos="-0.03 -0.05 0"/>
      <body name="fore" pos="0 -0.3 0">
        <joint name="elbow" type="hinge" axis="0 0 1" range="-0.1 2.5"/>
        <geom type="capsule" fromto="0 0 0 0 -0.25 0" size="0.02" mass="1"/>
        <site name="i_flex" pos="0.03 -0.05 0"/>
        <site name="i_ext" pos="-0.03 -0.05 0"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <spatial name="t_flex"><site site="o_flex"/><site site="i_flex"/></spatial>
    <spatial name="t_ext"><site site="o_ext"/><site site="i_ext"/></spatial>
  </tendon>
  <actuator>
    <muscle name="flexor" tendon="t_flex" force="1000" lengthrange="0.1 0.4"/>
    <muscle name="extensor" tendon="t_ext" force="1200" lengthrange="0.1 0.4"/>
  </actuator>
</mujoco>
"""


@dataclass(frozen=True)
class _EngineSpec:
    engine: str
    torque_nm: float
    fixture: str


def _sweep_config(frame: MuscleFrame) -> SweepConfig:
    return SweepConfig(
        noise=muscle_noise_spec(frame, kappa=_KAPPA),
        dt=_DT,
        stiffness_model=StiffnessModel.ENGINE_CURVE,
        n_levels=_N_LEVELS,
        linearity_limit=_LINEARITY_LIMIT,
    )


def _run_sweep(frame: MuscleFrame, torque_nm: float) -> CocontractionSweep:
    sweep = cocontraction_sweep(
        frame,
        np.array([torque_nm], dtype=np.float64),
        total_activation(frame.n_muscles),
        _sweep_config(frame),
    )
    if sweep.status is not SweepStatus.COMPUTED:
        reasons = "; ".join(sweep.reasons) or "unknown"
        raise ValueError(f"sweep not qualified for report: {reasons}")
    return sweep


def _entry_from_sweep(
    engine: str,
    fixture: str,
    torque_nm: float,
    sweep: CocontractionSweep,
) -> dict[str, Any]:
    noise = sweep.noise_trace.values.ravel()
    stiffness = sweep.stiffness_trace.values.ravel()
    if noise.shape != stiffness.shape or len(noise) < 3:
        raise ValueError("sweep must have at least three co-contraction levels")
    mono_noise = bool(np.all(np.diff(noise) > 0.0))
    mono_stiff = bool(np.all(np.diff(stiffness) > 0.0))
    return {
        "engine": engine,
        "fixture": fixture,
        "net_torque_N_m": torque_nm,
        "sweep_status": sweep.status.value,
        "index_name": sweep.index_name,
        "levels": sweep.levels.tolist(),
        "torque_residual_max": float(np.max(sweep.torque_residual)),
        "joint_torque_noise_variance_N2m2": noise.tolist(),
        "joint_stiffness_trace_Nm_per_rad": stiffness.tolist(),
        "noise_monotone_increasing": mono_noise,
        "stiffness_monotone_increasing": mono_stiff,
        "fiber_nullity": sweep.fiber_nullity,
        "non_claims": NON_CLAIMS,
    }


def _opensim_elbow_frame() -> MuscleFrame:
    from src.engines.physics_engines.opensim.python.muscle_mechanics import (
        opensim_muscle_frame,
    )
    from src.shared.python.engine_core.opensim_runtime import import_opensim

    osim = import_opensim()
    osim.Logger.setLevelString("error")
    model = osim.Model()
    model.setName("elbow-nsx94")
    ground = model.getGround()
    body = osim.Body(
        "forearm", 1.0, osim.Vec3(0, -0.15, 0), osim.Inertia(0.02, 0.02, 0.005)
    )
    model.addBody(body)
    joint = osim.PinJoint(
        "elbow", ground, osim.Vec3(0), osim.Vec3(0), body, osim.Vec3(0), osim.Vec3(0)
    )
    joint.upd_coordinates(0).setName("flex")
    model.addJoint(joint)
    for name, f_max, x in (("flexor", 1000.0, 0.03), ("extensor", 1200.0, -0.03)):
        mus = osim.Millard2012EquilibriumMuscle(name, f_max, 0.12, 0.18, 0.0)
        mus.addNewPathPoint(name + "_o", ground, osim.Vec3(x, 0.25, 0.0))
        mus.addNewPathPoint(name + "_i", body, osim.Vec3(x, -0.04, 0.0))
        model.addForce(mus)
    state = model.initSystem()
    model.getCoordinateSet().get(0).setValue(state, 0.0)
    model.realizePosition(state)
    return opensim_muscle_frame(model, state)


def _myosuite_elbow_frame() -> tuple[MuscleFrame, str]:
    from src.engines.physics_engines.myosuite.python.golfer_scene import (
        MYO_SIM_RELATIVE,
        REPO_ROOT_DEFAULT,
    )
    from src.engines.physics_engines.myosuite.python.muscle_mechanics import (
        mujoco_muscle_frame,
    )

    mujoco = __import__("mujoco")
    myo_path = REPO_ROOT_DEFAULT / MYO_SIM_RELATIVE / "elbow/myoelbow_1dof6muscles.xml"
    if myo_path.is_file():
        model = mujoco.MjModel.from_xml_path(str(myo_path))
        data = mujoco.MjData(model)
        data.qpos[0] = 0.5
        mujoco.mj_forward(model, data)
        return mujoco_muscle_frame(model, data), "pinned-myoelbow-1dof6muscles"
    model = mujoco.MjModel.from_xml_string(_MJCF_ELBOW)
    data = mujoco.MjData(model)
    data.qpos[0] = 0.0
    mujoco.mj_forward(model, data)
    return mujoco_muscle_frame(model, data), "mjcf-antagonist-elbow"


def build_tradeoff_report(
    *,
    engines: Sequence[str] = ("opensim", "myosuite"),
) -> dict[str, Any]:
    """Run sweeps and return the report dict (may import heavy engines)."""
    unknown = set(engines) - _REQUIRED_ENGINES
    if unknown:
        raise ValueError(f"unknown engines: {sorted(unknown)}")
    specs = (
        _EngineSpec("opensim", _OPENSIM_TORQUE_NM, "constructed-millard-elbow"),
        _EngineSpec("myosuite", _MYOSUITE_TORQUE_NM, "mujoco-muscle-elbow"),
    )
    entries: list[dict[str, Any]] = []
    for spec in specs:
        if spec.engine not in engines:
            continue
        if spec.engine == "opensim":
            frame = _opensim_elbow_frame()
            fixture = spec.fixture
        else:
            frame, fixture = _myosuite_elbow_frame()
        sweep = _run_sweep(frame, spec.torque_nm)
        entries.append(_entry_from_sweep(spec.engine, fixture, spec.torque_nm, sweep))
    entries.sort(key=lambda row: row["engine"])
    return {
        "schema": _SCHEMA,
        "issue": 12723,
        "validation_status": "UNVALIDATED",
        "non_claims": NON_CLAIMS,
        "entries": entries,
    }


def write_tradeoff_report(path: Path | None = None) -> Path:
    """Write the full two-engine report to ``path`` (default evidence location)."""
    target = path or DEFAULT_EVIDENCE_PATH
    doc = build_tradeoff_report()
    validate_tradeoff_report(doc)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target


def load_tradeoff_report(path: Path | None = None) -> dict[str, Any]:
    """Load and parse the committed evidence JSON."""
    target = path or DEFAULT_EVIDENCE_PATH
    if not target.is_file():
        raise FileNotFoundError(target)
    return json.loads(target.read_text(encoding="utf-8"))


def validate_tradeoff_report(doc: object) -> None:
    """Fail closed on schema, engine coverage, and trade-off invariants."""
    if not isinstance(doc, dict):
        raise TypeError("report must be a mapping")
    if doc.get("schema") != _SCHEMA:
        raise ValueError(f"unsupported schema: {doc.get('schema')!r}")
    if doc.get("validation_status") != "UNVALIDATED":
        raise ValueError("validation_status must be UNVALIDATED")
    entries = doc.get("entries")
    if not isinstance(entries, list) or len(entries) == 0:
        raise ValueError("entries must be a non-empty list")
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each entry must be a mapping")
        if entry.get("sweep_status") == "NOT_QUALIFIED":
            raise ValueError("NOT_QUALIFIED frames must not appear in the report")
        engine = entry.get("engine")
        if not isinstance(engine, str) or not engine:
            raise ValueError("entry.engine must be a non-empty string")
        seen.add(engine)
        levels = entry.get("levels")
        noise = entry.get("joint_torque_noise_variance_N2m2")
        stiffness = entry.get("joint_stiffness_trace_Nm_per_rad")
        if not isinstance(levels, list) or len(levels) < 3:
            raise ValueError(f"{engine}: need at least three co-contraction levels")
        if not isinstance(noise, list) or not isinstance(stiffness, list):
            raise ValueError(f"{engine}: noise and stiffness traces required")
        if len(noise) != len(levels) or len(stiffness) != len(levels):
            raise ValueError(f"{engine}: trace length must match levels")
        if entry.get("noise_monotone_increasing") is not True:
            raise ValueError(f"{engine}: noise trace must be strictly monotone")
        if entry.get("stiffness_monotone_increasing") is not True:
            raise ValueError(f"{engine}: stiffness trace must be strictly monotone")
        resid = entry.get("torque_residual_max")
        if not isinstance(resid, (int, float)) or float(resid) >= 1e-12:
            raise ValueError(f"{engine}: torque_residual_max must be below 1e-12")
    missing = _REQUIRED_ENGINES - seen
    if missing:
        raise ValueError(f"missing engines in report: {sorted(missing)}")


def _main(argv: Iterable[str] | None = None) -> int:
    _ = argv
    write_tradeoff_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
