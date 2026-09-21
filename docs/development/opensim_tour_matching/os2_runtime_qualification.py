"""OS-2 preflight: qualify the isolated OpenSim runtime against packaged inputs.

Runs only where ``import opensim`` succeeds (the ControlTower venv
/home/dieterolson/opensim-10003). Loads golf_humanoid.osim through OpenSim,
initialises the system, inventories bodies/coordinates/markers, reads the
OS-1 TRC through OpenSim's own TRCFileAdapter, and writes a receipt. It does
not calibrate markers, run IK or run Moco.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--osim", type=Path, required=True)
    parser.add_argument("--trc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt: dict = {
        "work_package": "OS-2 preflight: runtime qualification",
        "epic": "#10003",
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "inputs": {"osim_sha256": sha(args.osim), "trc_sha256": sha(args.trc)},
        "status": "running",
    }
    try:
        import opensim  # type: ignore[import-not-found]

        receipt["opensim_version"] = opensim.GetVersionAndDate()
        receipt["capabilities"] = {
            name: hasattr(opensim, name)
            for name in (
                "MocoStudy",
                "MocoTrack",
                "MocoInverse",
                "InverseKinematicsTool",
                "ScaleTool",
                "TRCFileAdapter",
                "CoordinateActuator",
            )
        }
        model = opensim.Model(str(args.osim))
        state = model.initSystem()
        receipt["model"] = {
            "name": model.getName(),
            "bodies": model.getBodySet().getSize(),
            "coordinates": model.getCoordinateSet().getSize(),
            "markers": model.getMarkerSet().getSize(),
            "actuators": model.getActuators().getSize(),
            "constraints": model.getConstraintSet().getSize(),
            "num_q": state.getNQ(),
            "num_u": state.getNU(),
            "gravity": list(model.getGravity().to_numpy()),
            "coordinate_names": [
                model.getCoordinateSet().get(i).getName()
                for i in range(model.getCoordinateSet().getSize())
            ],
        }
        table = opensim.TimeSeriesTableVec3(str(args.trc))
        labels = list(table.getColumnLabels())
        receipt["trc"] = {
            "rows": table.getNumRows(),
            "columns": table.getNumColumns(),
            "labels": labels,
            "first_time": table.getIndependentColumn()[0],
            "last_time": table.getIndependentColumn()[table.getNumRows() - 1],
            "units": table.getTableMetaDataAsString("Units"),
            "data_rate": table.getTableMetaDataAsString("DataRate"),
        }
        receipt["status"] = "qualified" if labels else "failed"
    except (ImportError, RuntimeError, ValueError, OSError) as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, default=str) + "\n")
    return 0 if receipt["status"] == "qualified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
