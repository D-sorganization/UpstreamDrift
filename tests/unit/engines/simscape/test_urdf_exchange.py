"""#11569 task 3: Simscape <-> spec URDF exchange receipt and diff.

Synthetic inventories check the diff contract; the committed receipt is an
R2025b run of ``scripts/matlab/export_simscape_urdf_exchange.m`` (model
inventory, not measured data).
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from src.engines.simscape.urdf_exchange import (
    SEGMENT_GROUPS,
    exchange_report,
    lf_sha256,
    load_exchange_receipt,
    urdf_inventory,
)

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[4]
RECEIPT_PATH = REPO / "tests/fixtures/simscape/simscape_urdf_exchange_receipt.json"
URDF_PATH = REPO / "src/engines/physics_engines/pinocchio/models/generated/golfer.urdf"
SLX_PATH = (
    REPO
    / "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/model"
    / "GolfSwing3D_Kinetic.slx"
)

MINI_URDF = """<robot name="mini">
  <link name="head"><inertial><mass value="5"/></inertial></link>
  <link name="neck_intermediate"><inertial><mass value="0.001"/></inertial></link>
  <link name="right_thigh"><inertial><mass value="7"/></inertial></link>
  <link name="mid_hands"/>
  <joint name="a" type="revolute"><parent link="head"/><child link="neck_intermediate"/></joint>
  <joint name="b" type="revolute"><parent link="neck_intermediate"/><child link="right_thigh"/></joint>
  <joint name="c" type="fixed"><parent link="head"/><child link="mid_hands"/></joint>
</robot>
"""


def _body(path: str, mass: float | None) -> dict:
    return {"path": path, "kind": "Inertia", "shape": "none", "mass_kg": mass}


def _receipt(urdf_sha: str) -> dict:
    canonical = {
        "joints": [{"path": "m/Hip Joint", "type": "Bushing Joint", "dof": 6}],
        "bodies": [_body("m/Hips and Torso Inputs/Head", 4.9), _body("m/Ghost", 0.0)],
        "n_coordinates": 6,
    }
    imported = {
        "joints": [
            {"path": "i/a", "type": "Revolute Joint", "dof": 1},
            {"path": "i/b", "type": "Revolute Joint", "dof": 1},
            {"path": "i/c", "type": "Weld Joint", "dof": 0},
        ],
        "bodies": [_body("i/head/Inertia", 5.0), _body("i/right_thigh/Inertia", 7.0)]
        + [_body("i/neck_intermediate/Inertia", 0.001)],
        "n_coordinates": 2,
    }
    return {
        "schema": "simscape-urdf-exchange/v1",
        "matlab_release": "2025b",
        "smexport_available": False,
        "canonical": {"model": "m", "model_sha256": "0" * 64, "inventory": canonical},
        "spec_urdf": {
            "urdf_sha256_lf": urdf_sha,
            "smimport_ok": True,
            "smimport_error": "",
            "inventory": imported,
        },
    }


@pytest.fixture()
def mini(tmp_path: Path) -> tuple[Path, dict]:
    urdf = tmp_path / "mini.urdf"
    urdf.write_text(MINI_URDF, encoding="utf-8")
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(_receipt(lf_sha256(urdf))), encoding="utf-8")
    return urdf, load_exchange_receipt(path)


def test_urdf_inventory_counts_movable_joints_and_link_masses(mini) -> None:
    urdf, _ = mini
    inv = urdf_inventory(urdf)
    assert inv.n_coordinates == 2
    assert inv.link_masses_kg == {
        "head": 5.0,
        "neck_intermediate": 0.001,
        "right_thigh": 7.0,
    }


def test_round_trip_compares_smimport_with_the_urdf(mini) -> None:
    urdf, receipt = mini
    report = exchange_report(receipt, urdf)
    assert report.receipt_current is True
    assert report.round_trip_mass_error_kg == pytest.approx(0.0, abs=1e-12)
    assert report.round_trip_coordinates == (2, 2)


def test_groups_split_simscape_and_spec_masses(mini) -> None:
    urdf, receipt = mini
    report = exchange_report(receipt, urdf)
    head = report.groups["head_neck"]
    assert head.simscape_kg == pytest.approx(4.9)
    assert head.spec_kg == pytest.approx(5.0)
    assert report.groups["legs"].simscape_kg == 0.0
    assert report.groups["legs"].spec_kg == pytest.approx(7.0)
    assert report.coordinates == (6, 2)
    # Zero-mass Simscape bodies and URDF intermediates are not "unmatched".
    assert report.unmatched_simscape == ()
    assert report.unmatched_spec == ()


def test_unevaluated_mass_is_reported_not_zeroed(mini) -> None:
    urdf, receipt = mini
    broken = copy.deepcopy(receipt)
    broken["canonical"]["inventory"]["bodies"].append(_body("m/Odd Solid", None))
    report = exchange_report(broken, urdf)
    assert report.unavailable_simscape == ("m/Odd Solid",)


def test_receipt_hash_ignores_crlf_checkouts(mini, tmp_path: Path) -> None:
    urdf, receipt = mini
    crlf = tmp_path / "crlf.urdf"
    crlf.write_bytes(MINI_URDF.replace("\n", "\r\n").encode("utf-8"))
    assert exchange_report(receipt, crlf).receipt_current is True


def test_stale_urdf_hash_marks_receipt_not_current(mini) -> None:
    urdf, receipt = mini
    urdf.write_text(MINI_URDF.replace('value="7"', 'value="8"'), encoding="utf-8")
    assert exchange_report(receipt, urdf).receipt_current is False


def test_load_rejects_wrong_schema(tmp_path: Path) -> None:
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"schema": "other/v0"}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_exchange_receipt(path)


def test_every_group_names_disjoint_members() -> None:
    sim = [n for g in SEGMENT_GROUPS.values() for n in g[0]]
    spec = [n for g in SEGMENT_GROUPS.values() for n in g[1]]
    assert len(sim) == len(set(sim))
    assert len(spec) == len(set(spec))


def test_committed_r2025b_receipt() -> None:
    receipt = load_exchange_receipt(RECEIPT_PATH)
    assert receipt["matlab_release"] == "2025b"
    assert receipt["smexport_available"] is False
    slx_sha = hashlib.sha256(SLX_PATH.read_bytes()).hexdigest()
    if receipt["canonical"]["model_sha256"] != slx_sha:
        pytest.skip("GolfSwing3D_Kinetic.slx changed: rerun the R2025b export")
    report = exchange_report(receipt, URDF_PATH)
    if not report.receipt_current:
        pytest.skip("golfer.urdf changed: rerun the R2025b export")
    assert report.coordinates == (27, 43)
    assert report.round_trip_coordinates == (43, 43)
    assert report.round_trip_mass_error_kg < 1e-9
    assert report.unavailable_simscape == ()
    assert report.unmatched_simscape == ()
    assert report.unmatched_spec == ()
    assert report.simscape_total_kg == pytest.approx(77.606, abs=1e-3)
    assert report.spec_total_kg == pytest.approx(77.969, abs=1e-6)
