"""Localize run18 state parity failures without changing tolerances or fitting."""

from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from src.engines.physics_engines.drake.python.native_model import NativeDrakeModel

base = Path("/mnt/c/Users/diete")
output = base / "drake-run18-diagnostic-10022-01.json"
if output.exists():
    raise FileExistsError(output)
raw = (base / "native_geometry_spec_9967.json").read_bytes()
engine = NativeDrakeModel(
    (base / "native-golf-9967-01.urdf").read_bytes(),
    (base / "native-golf-9967-01.sidecar.json").read_bytes(),
    raw,
)
report = {
    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "scope": "Rate discrepancy location and raw coordinate mass conditioning; no gate relaxation",
    "cases": [],
}
arrays = {}
for label in ("01", "02"):
    pin = np.load(base / f"drake-run18-pin-reference-10022-{label}/trajectory.npz")
    drake = np.load(base / f"drake-run18-parity-10022-{label}/trajectory.npz")
    names = json.loads(
        (base / f"drake-run18-parity-10022-{label}/report.json").read_bytes()
    )["coordinate_order"]
    delta = np.abs(drake["state"][:, 27:] - pin["state"][:, 27:])
    i, j = np.unravel_index(np.argmax(delta), delta.shape)
    row = pin["state"][i]
    engine.frame_poses(dict(zip(names, row[:27], strict=True)))
    mass = engine.plant.CalcMassMatrix(engine.context)
    eigen = np.linalg.eigvalsh(mass)
    peak = {
        "t_s": float(pin["time_s"][i]),
        "coordinate": names[j],
        "pin_qd": float(pin["state"][i, j + 27]),
        "drake_qd": float(drake["state"][i, j + 27]),
        "absolute_rate_difference": float(delta[i, j]),
        "relative_rate_difference": float(
            delta[i, j] / max(abs(pin["state"][i, j + 27]), np.finfo(float).tiny)
        ),
        "mass_eigenvalue_min": float(eigen.min()),
        "mass_eigenvalue_max": float(eigen.max()),
        "mass_condition_number": float(np.linalg.cond(mass)),
        "units_note": "Raw native generalized-coordinate mass condition is unit-dependent, not a physical instability certificate",
    }
    report["cases"].append({"label": label, "peak": peak})
    arrays["pin" + label] = pin
    arrays["drake" + label] = drake
for engine_name in ("pin", "drake"):
    coarse, fine = arrays[engine_name + "01"], arrays[engine_name + "02"]
    report[engine_name + "_coarse_fine"] = {
        "q_max_abs": float(
            np.max(np.abs(coarse["state"][:, :27] - fine["state"][:, :27]))
        ),
        "qd_max_abs": float(
            np.max(np.abs(coarse["state"][:, 27:] - fine["state"][:, 27:]))
        ),
        "marker_max_distance_m": float(
            np.max(np.linalg.norm(coarse["markers_m"] - fine["markers_m"], axis=2))
        ),
    }
output.write_text(json.dumps(report, indent=2))
sys.stdout.write(json.dumps(report) + "\n")
