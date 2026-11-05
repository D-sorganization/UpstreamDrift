"""Select the MuJoCo model that best matches the WSCG 2024 study (Issue #14437).

WSCG 2024 used a planar, joint-torque-driven Simscape model with two hands on
the grip and a flexible-beam shaft (``WSCG_2024_LEGACY_EVIDENCE_AUDIT.md``). No
engine has reproduced its force series, so the metric is structural: how many
of those four features each compiled ``MjModel`` reproduces. Ties go to planar,
then an existing MuJoCo ZTCF/ZVCF test, then fewer DOF. The winner is pinned in
a #14431 run manifest. A model-scenario selection, not empirical validation.

Usage::

    python3 -m scripts.research.proximal_distal_energy.wscg_model_selection "$(git rev-parse HEAD)"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from src.engines.physics_engines.mujoco import (
    _golf_swing_canonical_xml,
    golf_swing_models_xml,
    pendulum_models_xml,
)
from src.shared.python.simulation_backends import GolfModelParams
from src.shared.python.simulation_backends.mjcf import params_to_mjcf
from src.sim_run.manifest import RUN_MANIFEST_SCHEMA_VERSION, validate_run_manifest

REPO_ROOT = Path(__file__).resolve().parents[3]
RECORD_PATH = Path(
    "docs/research/proximal_distal_energy_transfer/data/wscg_model_selection.json"
)

#: WSCG model features, from the audit's slide 6 and limitations.
FEATURES = ("planar", "joint_torque_driven", "two_hand_closure", "flexible_shaft")

_HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
_JOINT_TRANSMISSION = int(mujoco.mjtTrn.mjTRN_JOINT)
_NO_BIAS = int(mujoco.mjtBias.mjBIAS_NONE)
_BODY_EQUALITIES = {int(mujoco.mjtEq.mjEQ_CONNECT), int(mujoco.mjtEq.mjEQ_WELD)}


@dataclass(frozen=True)
class Candidate:
    """A MuJoCo golf model with the bodies that carry its grip and club."""

    model_id: str
    source: str
    xml: Callable[[], str]
    club_root: str
    hands: tuple[str, ...] = ("left_hand", "right_hand")
    ztcf_test: str | None = None


def _module(model_id: str, module: Any, attr: str, club: str, **kw: Any) -> Candidate:
    return Candidate(
        model_id, f"{module.__name__}.{attr}", lambda: getattr(module, attr), club, **kw
    )


CANDIDATES: tuple[Candidate, ...] = (
    _module(
        "mujoco/advanced_biomechanical_golf_swing",
        golf_swing_models_xml,
        "ADVANCED_BIOMECHANICAL_GOLF_SWING_XML",
        "club_grip",
    ),
    Candidate(
        "mujoco/golf_double_pendulum_params",
        "src.shared.python.simulation_backends.mjcf.params_to_mjcf",
        lambda: params_to_mjcf(GolfModelParams.default()),
        club_root="lower",
        hands=("upper",),
        ztcf_test="tests/unit/simulation_backends/test_ztcf_zvcf.py",
    ),
    _module(
        "mujoco/golf_triple_pendulum",
        pendulum_models_xml,
        "TRIPLE_PENDULUM_XML",
        "club_body",
        hands=("forearm_body",),
    ),
    _module(
        "mujoco/golf_upper_body_swing",
        golf_swing_models_xml,
        "UPPER_BODY_GOLF_SWING_XML",
        "club",
    ),
    _module(
        "mujoco/golf_full_body_swing",
        golf_swing_models_xml,
        "FULL_BODY_GOLF_SWING_XML",
        "club",
    ),
    _module(
        "mujoco/canonical_golf_humanoid",
        _golf_swing_canonical_xml,
        "CANONICAL_GOLF_HUMANOID_XML",
        "club_grip",
    ),
)


@dataclass(frozen=True)
class CandidateScore:
    """Measured WSCG features for one candidate."""

    model_id: str
    features: dict[str, bool]
    nv: int
    has_ztcf_test: bool

    @property
    def score(self) -> int:
        return sum(self.features.values())

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(f for f in FEATURES if not self.features[f])

    def sort_key(self) -> tuple[int, int, int, int]:
        return (-self.score, -self.features["planar"], -self.has_ztcf_test, self.nv)


def _body_id(model: mujoco.MjModel, name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"model has no body named {name!r}")
    return body_id


def _ancestors(model: mujoco.MjModel, body_id: int) -> set[int]:
    chain = set()
    while body_id > 0:
        body_id = int(model.body_parentid[body_id])
        chain.add(body_id)
    return chain


def _subtree(model: mujoco.MjModel, root: int) -> set[int]:
    return {b for b in range(model.nbody) if b == root or root in _ancestors(model, b)}


def _is_planar(model: mujoco.MjModel) -> bool:
    if not np.all(model.jnt_type == _HINGE):
        return False
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    axes = data.xaxis[: model.njnt]
    return bool(np.allclose(np.abs(axes @ axes[0]), 1.0, atol=1e-9))


def _is_joint_torque_driven(model: mujoco.MjModel) -> bool:
    return model.nu > 0 and bool(
        np.all(model.actuator_trntype == _JOINT_TRANSMISSION)
        and np.all(model.actuator_biastype == _NO_BIAS)
    )


def _hands_on_club(model: mujoco.MjModel, candidate: Candidate) -> int:
    """Count hands rigidly carried by the club: as its ancestor or via equality."""
    club = _subtree(model, _body_id(model, candidate.club_root))
    club_ancestors = _ancestors(model, _body_id(model, candidate.club_root))
    linked = set()
    for hand in candidate.hands:
        hand_id = _body_id(model, hand)
        for i in range(model.neq):
            pair = {int(model.eq_obj1id[i]), int(model.eq_obj2id[i])}
            welded = int(model.eq_type[i]) in _BODY_EQUALITIES and hand_id in pair
            if welded and pair - {hand_id} <= club:
                linked.add(hand_id)
        if hand_id in club_ancestors:
            linked.add(hand_id)
    return len(linked)


def _has_flexible_shaft(model: mujoco.MjModel, candidate: Candidate) -> bool:
    root = _body_id(model, candidate.club_root)
    shaft = _subtree(model, root) - {root}
    return any(int(model.jnt_bodyid[j]) in shaft for j in range(model.njnt))


def measure(candidate: Candidate) -> CandidateScore:
    """Compile ``candidate`` and measure each WSCG feature from the MjModel."""
    model = mujoco.MjModel.from_xml_string(candidate.xml())
    features = {
        "planar": _is_planar(model),
        "joint_torque_driven": _is_joint_torque_driven(model),
        "two_hand_closure": _hands_on_club(model, candidate) >= 2,
        "flexible_shaft": _has_flexible_shaft(model, candidate),
    }
    test = candidate.ztcf_test
    has_test = test is not None and (REPO_ROOT / test).is_file()
    return CandidateScore(candidate.model_id, features, int(model.nv), has_test)


def rank_candidates() -> list[CandidateScore]:
    """Return every candidate, best WSCG structural match first."""
    return sorted((measure(c) for c in CANDIDATES), key=CandidateScore.sort_key)


def initial_condition_digest(qpos: Sequence[float], qvel: Sequence[float]) -> str:
    """SHA-256 of an initial state's canonical JSON; needs 0 < len(qvel) <= len(qpos)."""
    if not qpos or len(qvel) > len(qpos):
        raise ValueError(f"qpos must be non-empty and >= qvel ({len(qpos)} given)")
    payload = json.dumps(
        {"qpos": [float(x) for x in qpos], "qvel": [float(x) for x in qvel]},
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _candidate(model_id: str) -> Candidate:
    return next(c for c in CANDIDATES if c.model_id == model_id)


def _why_it_lost(best: CandidateScore, runner_up: CandidateScore) -> str:
    return (
        f"Scored {runner_up.score}/{len(FEATURES)} against "
        f"{best.score}/{len(FEATURES)}: it lacks "
        f"{' and '.join(runner_up.missing)}. With one grip it cannot produce "
        "the separate lead- and trail-hand forces the WSCG study reports, "
        "although it is planar and already has a MuJoCo ZTCF/ZVCF parity test."
    )


def build_record(*, ud_commit: str, created_at: str) -> dict[str, Any]:
    """Build the selection record; its manifest is validated before return."""
    ranking = rank_candidates()
    best, runner_up = ranking[0], ranking[1]
    xml = _candidate(best.model_id).xml()
    model = mujoco.MjModel.from_xml_string(xml)
    qpos = [float(x) for x in model.qpos0]
    qvel = [0.0] * model.nv
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "model": {
            "model_id": best.model_id,
            "revision": "sha256:" + hashlib.sha256(xml.encode("utf-8")).hexdigest(),
        },
        "parameters": {
            "timestep_s": float(model.opt.timestep),
            "gravity_m_s2": [float(g) for g in model.opt.gravity],
            "controls": "zero (ZTCF state); torques are set by #14438",
        },
        "input_data": {
            "sha256": initial_condition_digest(qpos, qvel),
            "label": "initial condition: compiled qpos0 with zero qvel",
        },
        "seed": 0,
        "engine": {"name": "mujoco", "version": mujoco.__version__},
        "ud_commit": ud_commit,
        "created_at": created_at,
    }
    validate_run_manifest(manifest)
    return {
        "issue": 14437,
        "scope": (
            "Model-scenario selection by structural features; not empirical "
            "validation and not a fit to the WSCG force series."
        ),
        "metric": {
            "name": "wscg_structural_feature_score",
            "features": list(FEATURES),
            "definition": (
                "Count of the four documented WSCG model features reproduced "
                "by the compiled MjModel. Ties: planar first, then an existing "
                "MuJoCo ZTCF/ZVCF test, then fewer DOF."
            ),
        },
        "ranking": [
            {
                "model_id": s.model_id,
                "score": s.score,
                "missing": list(s.missing),
                "nv": s.nv,
                "has_ztcf_test": s.has_ztcf_test,
            }
            for s in ranking
        ],
        "selected": best.model_id,
        "selected_source": _candidate(best.model_id).source,
        "runner_up": {
            "model_id": runner_up.model_id,
            "why_it_lost": _why_it_lost(best, runner_up),
        },
        "caveats": [
            "The selected model is 3D; WSCG is planar, so #14438 must project "
            "hand forces onto a declared swing plane.",
            "qpos0 is the compiled reference pose, not a matched address pose.",
        ],
        "initial_condition": {
            "nq": int(model.nq),
            "nv": int(model.nv),
            "qpos": qpos,
        },
        "manifest": manifest,
    }


#: A JSON list or object with no nested containers except one-line lists.
_FLAT = re.compile(r"([\[{])\s+((?:[^\[\]{}]|\[[^\[\]{}\n]*\])*?)\s+([\]}])")


def _inline(match: re.Match[str]) -> str:
    """Put one flat container on a single line."""
    body = re.sub(r",\n\s+", ", ", match.group(2))
    return match.group(1) + body + match.group(3)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ud_commit", help="40-char UpstreamDrift commit SHA")
    args = parser.parse_args(argv)
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    record = build_record(ud_commit=args.ud_commit, created_at=created_at)
    path = REPO_ROOT / RECORD_PATH
    text = _FLAT.sub(_inline, _FLAT.sub(_inline, json.dumps(record, indent=2)))
    path.write_text(text + "\n", encoding="utf-8")
    print(f"selected {record['selected']}; wrote {RECORD_PATH}")


if __name__ == "__main__":
    main()
