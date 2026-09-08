"""End to end on a synthetic bundle: synth -> fit -> reconstruction.json with metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.motion_capture.reconstruct import __main__ as cli
from src.motion_capture.reconstruct.fit import RECONSTRUCTION_FILE, fit_bundle

pytestmark = pytest.mark.unit


def test_synth_then_fit_writes_reconstruction_with_metrics(tmp_path: Path) -> None:
    bundle = tmp_path / "synthetic"
    assert (
        cli.main(
            ["synth", "--out", str(bundle), "--frames", "24", "--outliers", "0.02"]
        )
        == 0
    )
    truth = json.loads((bundle / "truth.json").read_text(encoding="utf-8"))
    anchor = ("neck", truth["bone_lengths_m"]["neck"])
    code = cli.main(
        ["fit", "--bundle", str(bundle), "--anchor", f"{anchor[0]}={anchor[1]}"]
    )
    assert code == 0
    record = json.loads((bundle / RECONSTRUCTION_FILE).read_text(encoding="utf-8"))
    assert record["schema_version"].startswith("reconstruction/")
    assert record["frames"] == 24 and len(record["cameras"]) == 3
    assert record["rms_px"] < 2.0
    assert set(record["metrics"]) == {"cameras", "joints", "bone_length_relative_error"}
    assert all(v["rotation_deg"] < 0.5 for v in record["metrics"]["cameras"].values())
    # the shortest segments (nose, hips) are the least constrained by 24 frames
    assert max(record["metrics"]["bone_length_relative_error"].values()) < 0.05
    assert record["metrics"]["joints"]["missing"] == 0
    assert {v["view"] for v in record["views"]} == {"face_on", "down_line", "overhead"}
    assert all(r["residual_px"] > 0 for r in record["rejected"])
    assert (bundle / "joints_3d_m.npy").is_file()
    assert np.load(bundle / "joints_3d_m.npy").shape == (24, 15, 3)


def test_fit_requires_a_start_and_matching_views(tmp_path: Path) -> None:
    bundle = tmp_path / "s"
    cli.main(["synth", "--out", str(bundle), "--frames", "6"])
    (bundle / "truth.json").unlink()
    with pytest.raises(Exception, match="no start cameras"):
        fit_bundle(bundle, scale_anchor=("neck", 0.5))
    with pytest.raises(SystemExit, match="anchor"):
        cli.main(["fit", "--bundle", str(bundle), "--anchor", "neck"])
