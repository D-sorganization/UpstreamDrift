"""GCV-3 (#11709): Simscape per-foot ground reaction from sole contacts.

Synthetic mini CSVs check the column contract and the GCV-1 reduction; the
committed fixture is an R2025b run of the exploratory GS3DX_FullBodyContact
model standing from rest (simulation output, not measured data), exported by
``scripts/matlab/export_gs3dx_grf_fixture.m``.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.engines.simscape.force_channels import load_simscape_force_series
from src.engines.simscape.ground_contact import (
    FOOT_CONTACTS,
    contact_columns,
    ground_reaction_series,
)
from src.shared.python.force_overlay import WrenchKind

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).resolve().parents[4] / "tests/fixtures/simscape"
CSV_PATH = FIXTURES / "simscape_gs3dx_grf_fixture.csv"
RECEIPT_PATH = FIXTURES / "simscape_gs3dx_grf_receipt.json"
ZG = -1.0


def _contact_cols(loads: dict[str, tuple[tuple, tuple]]) -> dict[str, list[float]]:
    """Two samples; every contact gets (force, point), default unloaded."""
    cols: dict[str, list[float]] = {"time": [0.0, 0.01]}
    for i, name in enumerate(c for names in FOOT_CONTACTS.values() for c in names):
        force, point = loads.get(name, ((0.0, 0.0, 0.0), (0.1 * i, 0.0, ZG)))
        for q, vec in (("Force", force), ("Point", point)):
            for col, value in zip(contact_columns(name, q), vec, strict=True):
                cols[col] = [float(value)] * 2
    for k, value in zip("123", (0.0, 0.0, 0.0), strict=True):
        cols[f"COMLogs_GlobalPosition_{k}"] = [value] * 2
    cols["GroundContactLogs_GroundHeight"] = [ZG] * 2
    return cols


def _write(tmp_path: Path, cols: dict[str, list[float]]) -> Path:
    path = tmp_path / "contact.csv"
    names = list(cols)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(names)
        for i in range(len(cols["time"])):
            w.writerow([repr(cols[k][i]) for k in names])
    return path


def test_per_foot_grf_at_cop(tmp_path: Path) -> None:
    loads = {
        "LHeel": ((0.0, 0.0, 300.0), (0.0, 0.2, ZG)),
        "LToeIn": ((0.0, 0.0, 100.0), (0.2, 0.2, ZG)),
        "RToeOut": ((10.0, 0.0, 400.0), (0.2, -0.2, ZG)),
    }
    series, missing = load_simscape_force_series(_write(tmp_path, _contact_cols(loads)))
    assert not [m for m in missing if m.startswith("contact:grf_")]
    by = {w.label: w for w in series[1].wrenches}
    left, right = by["contact:grf_left"], by["contact:grf_right"]
    assert left.kind is WrenchKind.CONTACT
    assert left.force_n == pytest.approx((0.0, 0.0, 400.0))
    # CoP: force-weighted mean of the loaded contacts for vertical loads.
    assert left.point_m == pytest.approx((0.05, 0.2, ZG))
    assert right.force_n == pytest.approx((10.0, 0.0, 400.0))
    assert right.point_m[2] == pytest.approx(ZG)
    assert by["contact:grf_net"].force_n == pytest.approx((10.0, 0.0, 800.0))


def test_grf_unavailable_without_contact_columns(tmp_path: Path) -> None:
    """Canonical model: no contact columns -> unavailable, never zero."""
    cols = {"time": [0.0, 0.01]}
    for k in "123":
        cols[f"HipLogs_BaseonHipForceGlobal_{k}"] = [1.0, 1.0]
        cols[f"HipLogs_BaseonHipTorqueGlobal_{k}"] = [0.0, 0.0]
        cols[f"HipLogs_HipGlobalPosition_dim{k}"] = [0.0, 0.0]
    series, missing = load_simscape_force_series(_write(tmp_path, cols))
    assert "contact:grf_left:force" in missing
    assert "contact:grf_right:force" in missing
    assert not any(w.kind is WrenchKind.CONTACT for w in series[0].wrenches)


def test_ground_height_must_be_constant(tmp_path: Path) -> None:
    cols = _contact_cols({})
    cols["GroundContactLogs_GroundHeight"] = [ZG, ZG + 0.1]
    with pytest.raises(ValueError, match="ground height"):
        load_simscape_force_series(_write(tmp_path, cols))


def test_ground_reaction_series_validates_inputs() -> None:
    names = [c for foot in FOOT_CONTACTS.values() for c in foot]
    zeros = {n: np.zeros((2, 3)) for n in names}
    with pytest.raises(ValueError, match="ground_height_m"):
        ground_reaction_series(zeros, zeros, np.zeros((2, 3)), float("nan"))
    with pytest.raises(ValueError, match="shape"):
        ground_reaction_series(zeros, zeros, np.zeros((3, 3)), 0.0)
    with pytest.raises(ValueError, match="missing contact"):
        ground_reaction_series({}, zeros, np.zeros((2, 3)), 0.0)


# --- committed R2025b fixture (software consistency + standing physics) -----


def _receipt() -> dict:
    return json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))


def test_receipt_is_r2025b_and_matches_fixture() -> None:
    receipt = _receipt()
    assert receipt["matlab_release"] == "2025b"
    assert receipt["issue"] == "#11709"
    assert receipt["rest"] is True
    assert receipt["newton_residual_max_ns"] <= receipt["newton_bound_ns"]
    digest = hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()
    assert digest == receipt["fixture_sha256"]


def test_fixture_feet_carry_the_body_with_cop_inside_each_foot() -> None:
    receipt = _receipt()
    series, missing = load_simscape_force_series(CSV_PATH)
    assert not [m for m in missing if m.startswith("contact:")]
    with CSV_PATH.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    peak_share = 0.0
    for frame, row in zip(series, rows, strict=True):
        by = {w.label: w for w in frame.wrenches}
        if "contact:grf_net" not in by:
            continue
        net = by["contact:grf_net"].force_n
        assert net is not None
        peak_share = max(peak_share, net[2] / receipt["weight_n"])
        for foot, names in FOOT_CONTACTS.items():
            grf = by.get(f"contact:grf_{foot}")
            if grf is None:
                continue
            pts = np.array(
                [[float(row[c]) for c in contact_columns(n, "Point")] for n in names]
            )
            lo, hi = pts[:, :2].min(axis=0), pts[:, :2].max(axis=0)
            cop = np.array(grf.point_m[:2])
            assert np.all(cop >= lo - 1e-9) and np.all(cop <= hi + 1e-9), foot
    # The ground carries the body weight (MATLAB's own test bound) and never
    # exceeds the receipt's 1 kHz peak on the decimated rows.
    assert 0.9 < peak_share <= receipt["support_bw_max"] + 1e-9
