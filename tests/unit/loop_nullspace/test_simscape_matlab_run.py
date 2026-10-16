"""NSX-55: run the R2025b exporter and load its output (#12583).

Skips when no MATLAB R2025b is available.  Set ``NSX_MATLAB_R2025B`` to the
``matlab`` executable to point at a specific install; R2026a and any other
release are never used (the exporter itself refuses to run outside R2025b).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.engine_core.capabilities import CapabilityLevel
from src.shared.python.loop_nullspace.contracts import InputProvenance
from src.shared.python.loop_nullspace.fixtures import planar_golfer_snapshot
from src.shared.python.loop_nullspace.simscape_export import (
    load_simscape_export,
    snapshot_from_export,
    verify_model_file,
)
from src.shared.python.loop_nullspace.simscape_rig_spec import planar_rig_spec

pytestmark = [pytest.mark.requires_matlab, pytest.mark.slow]

EXPORTER_DIR = (
    Path(__file__).resolve().parents[3]
    / "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/nsx_closed_loop_export"
)
CANDIDATES = (
    "C:/Program Files/MATLAB/R2025b/bin/matlab.exe",
    "/usr/local/MATLAB/R2025b/bin/matlab",
    "/opt/MATLAB/R2025b/bin/matlab",
)


def _matlab_r2025b() -> str:
    explicit = os.environ.get("NSX_MATLAB_R2025B")
    for path in (explicit, *CANDIDATES):
        if path and Path(path).is_file():
            return path
    found = shutil.which("matlab")
    if found and "R2025b" in found:
        return found
    pytest.skip("MATLAB R2025b not found (set NSX_MATLAB_R2025B)")
    raise AssertionError  # pragma: no cover


def test_exporter_output_loads_and_matches_the_analytic_model(tmp_path: Path) -> None:
    matlab = _matlab_r2025b()
    spec = tmp_path / "rig_spec.json"
    spec.write_text(json.dumps(planar_rig_spec()), encoding="utf-8")
    out = tmp_path / "out"
    command = (
        f"addpath('{EXPORTER_DIR.as_posix()}');"
        f"nsx_simscape_closed_loop_export('{spec.as_posix()}','{out.as_posix()}');"
    )
    run = subprocess.run(  # noqa: S603 - fixed argv, trusted local executable
        [matlab, "-batch", command],
        capture_output=True,
        text=True,
        timeout=1200,
        check=False,
    )
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    export = load_simscape_export(out / "simscape_closed_loop_export.json")
    assert export.provenance.matlab_release == "R2025b"
    assert export.capability() is CapabilityLevel.FULL
    verify_model_file(export, out / "NSX_PlanarRig.slx")
    snap = snapshot_from_export(
        export, 0, input_provenance=InputProvenance("matlab", "pytest")
    )
    ref = planar_golfer_snapshot().snapshot
    for name in ("M", "h", "J_c", "Jdot_v"):
        want = getattr(ref, name)
        err = np.max(np.abs(getattr(snap, name) - want))
        assert err <= 1e-10 * float(np.max(np.abs(want))), name
