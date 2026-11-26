"""Admission contract for the canonical-v2 solution bundle (Refs #15808 WP-0c)."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.optimization.solution_bundle import (
    SolutionBundleError,
    load_solution_bundle,
    save_solution_bundle,
)
from tests.unit.optimization.test_solution_bundle import _bundle

pytestmark = pytest.mark.unit

TOLERANCES = {
    "root_force_bw_fraction": 0.02,
    "root_moment_bw_height_fraction": 0.02,
    "impact_impulse_error_fraction": 0.05,
    "stance_slip_m": 0.005,
    "grip_weld_gap_m": 0.002,
    "replay_joint_rms_rad": math.radians(2.0),
    "feedback_rms_capacity_fraction": 0.05,
}


def _channels(n: int, joints: int) -> dict[str, np.ndarray]:
    """Return torque channels that already satisfy the accounting identity."""
    feed = np.ones((n, joints))
    intervention = np.zeros((n, joints))
    balance = np.full((n, joints), 0.25)
    saturation = np.full((n, joints), -0.1)
    return {
        "tau_feedforward": feed,
        "tau_intervention": intervention,
        "tau_balance_correction": balance,
        "tau_saturation": saturation,
        "tau_realized": feed + intervention + balance + saturation,
    }


def _admitted():
    """Return a legacy bundle plus a valid admission block."""
    base = _bundle()
    time = np.asarray(base.arrays["time"])
    start, end = float(time[1]), float(time[-2])
    arrays = dict(base.arrays)
    arrays["frame_valid"] = (time >= start) & (time <= end)
    arrays.update(_channels(time.shape[0], len(base.joint_order)))
    admission = {
        "controller": {
            "name": "phase-pd",
            "feedback": "phase-wise-pd",
            "capacity_nm": 150.0,
        },
        "interval": {"t_start": start, "t_end": end},
        "tolerances": dict(TOLERANCES),
    }
    return replace(base, arrays=arrays, admission=admission)


def test_preregistered_tolerances_match_the_contract() -> None:
    """The public tolerance table matches the owner-approved contract."""
    from src.shared.python.optimization.admission import PREREGISTERED_TOLERANCES

    assert set(PREREGISTERED_TOLERANCES) == set(TOLERANCES)
    for key, value in TOLERANCES.items():
        assert PREREGISTERED_TOLERANCES[key] == pytest.approx(value)


def test_legacy_bundle_roundtrips_with_null_admission(tmp_path: Path) -> None:
    """A bundle with no admission block saves and loads as null."""
    original = _bundle()
    save_solution_bundle(original, tmp_path / "b")
    loaded = load_solution_bundle(tmp_path / "b")

    assert loaded.admission is None
    manifest = json.loads((tmp_path / "b" / "manifest.json").read_text("utf-8"))
    assert manifest["admission"] is None


def test_admission_roundtrip_keeps_interval_flags_and_channels(tmp_path: Path) -> None:
    """Admission metadata, validity flags, and realized torque survive a reload."""
    original = _admitted()
    save_solution_bundle(original, tmp_path / "b")
    loaded = load_solution_bundle(tmp_path / "b")

    assert loaded.admission == original.admission
    np.testing.assert_array_equal(
        loaded.arrays["frame_valid"], original.arrays["frame_valid"]
    )
    np.testing.assert_array_equal(
        loaded.arrays["tau_realized"], original.arrays["tau_realized"]
    )


def test_legacy_manifest_without_admission_key_loads(tmp_path: Path) -> None:
    """A manifest written before this field still loads."""
    save_solution_bundle(_bundle(), tmp_path / "b")
    path = tmp_path / "b" / "manifest.json"
    manifest = json.loads(path.read_text("utf-8"))
    manifest.pop("admission", None)
    path.write_text(json.dumps(manifest), "utf-8")

    loaded = load_solution_bundle(tmp_path / "b")
    assert loaded.admission is None


def test_save_rejects_loosened_tolerance(tmp_path: Path) -> None:
    """A looser root-force tolerance is not preregistered."""
    bundle = _admitted()
    tolerances = dict(bundle.admission["tolerances"])
    tolerances["root_force_bw_fraction"] = 0.2
    bundle = replace(bundle, admission={**bundle.admission, "tolerances": tolerances})

    with pytest.raises(SolutionBundleError, match="tolerances"):
        save_solution_bundle(bundle, tmp_path / "b")


def test_save_rejects_missing_torque_channel(tmp_path: Path) -> None:
    """Admission requires every torque channel, including intervention."""
    bundle = _admitted()
    del bundle.arrays["tau_intervention"]

    with pytest.raises(SolutionBundleError, match="tau_intervention"):
        save_solution_bundle(bundle, tmp_path / "b")


def test_save_rejects_broken_torque_identity(tmp_path: Path) -> None:
    """Realized torque must equal the sum of the other channels."""
    bundle = _admitted()
    bundle.arrays["tau_realized"] = bundle.arrays["tau_realized"] + 1.0

    with pytest.raises(SolutionBundleError, match="tau_realized"):
        save_solution_bundle(bundle, tmp_path / "b")


def test_save_rejects_valid_flag_outside_interval(tmp_path: Path) -> None:
    """A frame outside the admitted interval cannot be marked valid."""
    bundle = _admitted()
    flags = np.array(bundle.arrays["frame_valid"])
    flags[0] = True
    bundle.arrays["frame_valid"] = flags

    with pytest.raises(SolutionBundleError, match="frame_valid"):
        save_solution_bundle(bundle, tmp_path / "b")


def test_save_rejects_interval_outside_record(tmp_path: Path) -> None:
    """The admitted interval cannot extend past the recorded time base."""
    bundle = _admitted()
    interval = {**bundle.admission["interval"], "t_end": 9.0}
    bundle = replace(bundle, admission={**bundle.admission, "interval": interval})

    with pytest.raises(SolutionBundleError, match="interval"):
        save_solution_bundle(bundle, tmp_path / "b")


def test_save_rejects_unknown_controller_key(tmp_path: Path) -> None:
    """The controller declaration rejects keys outside the contract."""
    bundle = _admitted()
    controller = {**bundle.admission["controller"], "gain": 1.0}
    bundle = replace(bundle, admission={**bundle.admission, "controller": controller})

    with pytest.raises(SolutionBundleError, match="controller"):
        save_solution_bundle(bundle, tmp_path / "b")


def test_save_rejects_admission_with_no_valid_frame(tmp_path: Path) -> None:
    """An admission block must mark at least one sample valid."""
    bundle = _admitted()
    bundle.arrays["frame_valid"] = np.zeros_like(bundle.arrays["frame_valid"])

    with pytest.raises(SolutionBundleError, match="frame_valid"):
        save_solution_bundle(bundle, tmp_path / "b")


def test_save_rejects_channels_without_admission(tmp_path: Path) -> None:
    """Torque channels without an admission block are rejected."""
    bundle = _bundle()
    bundle.arrays.update(_channels(bundle.arrays["time"].shape[0], 3))

    with pytest.raises(SolutionBundleError, match="admission"):
        save_solution_bundle(bundle, tmp_path / "b")
