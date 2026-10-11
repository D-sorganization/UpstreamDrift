"""Foot width spheres and torsional friction of the contact model (#11671)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"


def _spec() -> dict:
    from src.shared.python.motion_matching.pipeline.lane import add_toe_spheres

    return add_toe_spheres(json.loads(SPEC.read_text(encoding="utf-8")))


def test_foot_width_adds_lateral_pairs_for_heel_and_forefoot() -> None:
    from src.shared.python.motion_matching.pipeline.lane import add_foot_width_spheres

    base = _spec()
    wide = add_foot_width_spheres(base, half_width_m=0.03)
    by_name = {s["name"]: s for s in wide["contact"]["spheres"]}
    for side in ("r", "l"):
        for stem in ("heel", "forefoot"):
            centre = by_name[f"{stem}_{side}"]
            for tag, sign in (("zp", 1.0), ("zn", -1.0)):
                lateral = by_name[f"{stem}_{tag}_{side}"]
                assert lateral["body"] == centre["body"]
                assert lateral["radius_m"] == centre["radius_m"]
                assert lateral["position_m"][2] == pytest.approx(
                    centre["position_m"][2] + sign * 0.03
                )
                assert lateral["position_m"][:2] == centre["position_m"][:2]
        assert f"toe_zp_{side}" not in by_name
    assert len(wide["contact"]["spheres"]) == len(base["contact"]["spheres"]) + 8


def test_foot_width_leaves_the_input_untouched_and_is_idempotent() -> None:
    from src.shared.python.motion_matching.pipeline.lane import add_foot_width_spheres

    base = _spec()
    snapshot = copy.deepcopy(base)
    once = add_foot_width_spheres(base, half_width_m=0.03)
    assert base == snapshot
    assert add_foot_width_spheres(once, half_width_m=0.03) == once


def test_foot_width_records_torsion_and_provenance() -> None:
    from src.shared.python.motion_matching.pipeline.lane import add_foot_width_spheres

    wide = add_foot_width_spheres(_spec(), half_width_m=0.03, torsional_patch_m=0.02)
    assert wide["contact"]["torsion"] == {
        "patch_radius_m": 0.02,
        "transition_rad_s": 0.5,
    }
    assert "foot width" in wide["provenance"]
    plain = add_foot_width_spheres(_spec(), half_width_m=0.03)
    assert "torsion" not in plain["contact"]


@pytest.mark.parametrize("width", [0.0, -0.01, float("nan"), float("inf")])
def test_foot_width_rejects_invalid_width(width: float) -> None:
    from src.shared.python.motion_matching.pipeline.lane import add_foot_width_spheres

    with pytest.raises(ValueError, match="half_width_m"):
        add_foot_width_spheres(_spec(), half_width_m=width)


def test_torsional_patch_rejects_negative() -> None:
    from src.shared.python.motion_matching.pipeline.lane import add_foot_width_spheres

    with pytest.raises(ValueError, match="torsional_patch_m"):
        add_foot_width_spheres(_spec(), half_width_m=0.03, torsional_patch_m=-0.1)


def _sim(document: dict):
    fs = pytest.importorskip(
        "src.shared.python.motion_matching.full_body_forward_dynamics"
    )
    model = pytest.importorskip(
        "src.engines.physics_engines.mujoco.python.full_body_model"
    )
    sim = fs.FullBodySimulator(
        model.NativeMujocoFullBodyModel(json.dumps(document).encode())
    )
    return fs, sim


def test_torsion_resists_pivoting_and_is_inert_when_still() -> None:
    from src.shared.python.motion_matching.pipeline.lane import add_foot_width_spheres

    doc = add_foot_width_spheres(_spec(), half_width_m=0.03)
    torsional = add_foot_width_spheres(
        _spec(), half_width_m=0.03, torsional_patch_m=0.03
    )
    fs, plain = _sim(doc)
    _, grip = _sim(torsional)
    q = fs.preload_feet(plain, np.zeros(plain.nv))
    still = np.zeros(plain.nv)
    spin = still.copy()
    spin[list(plain.names).index("HipInputZ")] = 1.5

    def contact_tau(sim, v):
        return np.asarray(sim.adapter.generalized_forces(sim._map(q), sim._map(v))[1])

    assert contact_tau(plain, still) == pytest.approx(contact_tau(grip, still))
    extra = contact_tau(grip, spin) - contact_tau(plain, spin)
    assert np.linalg.norm(extra) > 0.0
    assert float(extra @ spin) < 0.0  # dissipative: it opposes the pivot


def test_stance_expansion_pins_lateral_siblings_of_pinned_spheres() -> None:
    from src.shared.python.motion_matching.pipeline.lane import (
        add_foot_width_spheres,
        expand_stance_for_width,
    )

    names = [
        s["name"] for s in add_foot_width_spheres(_spec(), 0.03)["contact"]["spheres"]
    ]
    stance = [("heel_r", "forefoot_r", "toe_r"), ("heel_l",), ()]
    out = expand_stance_for_width(stance, names)
    assert set(out[0]) == {
        "heel_r",
        "forefoot_r",
        "toe_r",
        "heel_zp_r",
        "heel_zn_r",
        "forefoot_zp_r",
        "forefoot_zn_r",
    }
    assert set(out[1]) == {"heel_l", "heel_zp_l", "heel_zn_l"}
    assert out[2] == ()
    assert expand_stance_for_width(stance, ["heel_r"]) == [tuple(f) for f in stance]


def test_torsion_alone_changes_no_geometry() -> None:
    from src.shared.python.motion_matching.pipeline.lane import add_torsional_friction

    base = _spec()
    out = add_torsional_friction(base, 0.05)
    assert out["contact"]["spheres"] == base["contact"]["spheres"]
    assert out["contact"]["torsion"]["patch_radius_m"] == 0.05
    assert "torsion" not in base["contact"]
    with pytest.raises(ValueError, match="torsional_patch_m"):
        add_torsional_friction(base, -1.0)
