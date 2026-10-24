"""Measure OpenSim IK and inverse dynamics of the barbell lifts (OSL-3, OSL-12).

Usage (headless, from the repository root, real OpenSim 4.6 and the
``opensim-models`` submodule or ``$UPSTREAMDRIFT_MSK_BASE_MODEL``)::

    MUJOCO_GL=egl python3 scripts/measure_opensim_lift_ik_id.py \
        --out docs/development/lifting/evidence/opensim_ik_id/receipts.json

For each of the five barbell lifts the record holds the static-hold
contact-derived ground reaction of the OSL-2 model and, when a LIFT-5
same-input bundle exists, the IK and inverse-dynamics receipt (marker and
segment-angle errors, residual forces and moments, bundle SHA-256).  A lift
without a bundle is recorded unavailable with the reason, never as zero.

OSL-16 (#13104) loaded-foot tracking landed in ``cb8b736d2e3c0c8f4bd45deafa53896a6f13a5f7``
(PR #13120); acceptance:
``tests/unit/lifting/test_loaded_foot_tracking.py::test_loaded_feet_stay_within_the_lift5_ankle_bounds``
and ``::test_tracking_converges_in_the_rk4_substep``.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shared.python.lifting.opensim_full_body_lifts import (  # noqa: E402
    IK_ID_STRIDE,
    OpenSimFullBodyLiftSpec,
    compose_opensim_full_body_lift,
    opensim_lift_ik_id,
    opensim_lift_static_grf,
)
from src.shared.python.lifting.pack_audit.names import LIFTS  # noqa: E402

logger = logging.getLogger("measure_opensim_lift_ik_id")
DEFAULT_OUT = ROOT / "docs/development/lifting/evidence/opensim_ik_id/receipts.json"


def _git(*args: str) -> str:
    out = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", "-C", str(ROOT), *args],  # noqa: S607
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return out.stdout.strip()


def _static(lift: str) -> dict[str, Any]:
    composition = compose_opensim_full_body_lift(OpenSimFullBodyLiftSpec(lift))
    hold = opensim_lift_static_grf(composition)
    hold["spec"] = {
        "body_mass_kg": composition.spec.body_mass_kg,
        "height_m": composition.spec.height_m,
        "bar_mass_kg": composition.spec.bar_mass_kg,
    }
    return hold


def _portable(value: Any) -> Any:
    """Absolute local paths become repository-relative (or the file name)."""
    if isinstance(value, dict):
        return {k: _portable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_portable(v) for v in value]
    if isinstance(value, str) and value.startswith("/") and Path(value).suffix:
        path = Path(value)
        return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name
    return value


def measure(stride: int) -> dict[str, Any]:
    """Static holds and IK/ID receipts of every barbell lift."""
    lifts = {}
    for lift in LIFTS:
        logger.info("measuring %s", lift)
        lifts[lift] = {
            "static_hold": _static(lift),
            "ik_id": opensim_lift_ik_id(OpenSimFullBodyLiftSpec(lift), stride=stride),
        }
    return {
        "issue": "#13048 (OSL-12, on #13041 OSL-3)",
        "source_commit": _git("rev-parse", "HEAD"),
        "generator": "scripts/measure_opensim_lift_ik_id.py",
        "stride": stride,
        "lifts": _portable(lifts),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stride", type=int, default=IK_ID_STRIDE)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    record = measure(args.stride)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
