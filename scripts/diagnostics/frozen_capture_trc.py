"""C3D preparation in a separate clean process and its source-bound TRC (#12065).

On Windows the native OpenSim SDK and the Python ``ezc3d`` extension ship
ABI-incompatible ``ezc3d.dll`` builds, so one process can host only one of
them.  This module is the preparation side of that boundary: run as
``python -m scripts.diagnostics.frozen_capture_trc CAPTURE OUTPUT`` it reads
the frozen C3D through ``ezc3d`` in a process that never imports ``opensim``
and writes a TRC plus a JSON receipt.  The native side calls
:func:`load_transferred_capture`, which never imports ``ezc3d`` and proves,
from the receipt, that the source hash, labels, validity mask, clock, units
and observed points crossed the boundary bit for bit.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from src.shared.python.data_io.native_dll_guard import EZC3D_DLL, loaded_dll_path
from src.shared.python.motion_matching import tour_capture_contract
from src.shared.python.motion_matching.tour_capture_contract import TourCapture
from src.shared.python.motion_matching.trc_io import read_trc, write_trc

REPO_ROOT = Path(__file__).resolve().parents[2]
TRC_UNITS = "m"
PREPARATION_MODULE = "scripts.diagnostics.frozen_capture_trc"
_RECEIPT_KEYS = (
    "capture_sha256",
    "trc_sha256",
    "units",
    "frames",
    "labels",
    "labels_sha256",
    "valid_mask_sha256",
    "source_clock_sha256",
    "points_sha256",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def capture_fingerprint(capture: TourCapture) -> dict[str, str]:
    """Return the hashes that must be identical on both sides of the boundary.

    Invalid samples are hashed as zero so the NaN payload of a missing marker
    cannot change the identity; the validity mask is hashed separately.
    """
    valid = np.asarray(capture.valid, dtype=bool)
    observed = np.where(valid[..., None], capture.points_m, 0.0).astype(np.float64)
    labels = json.dumps(list(capture.labels)).encode("utf-8")
    return {
        "labels_sha256": hashlib.sha256(labels).hexdigest(),
        "valid_mask_sha256": _sha256_array(valid),
        "source_clock_sha256": _sha256_array(
            np.asarray(capture.time_s, dtype=np.float64)
        ),
        "points_sha256": _sha256_array(observed),
    }


def preparation_runtime() -> dict[str, object]:
    """Identify the interpreter and C3D runtime that produced a receipt."""
    ezc3d = sys.modules.get("ezc3d")
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "ezc3d_version": getattr(ezc3d, "__version__", None),
        "ezc3d_dll": loaded_dll_path(EZC3D_DLL),
        "opensim_loaded": "opensim" in sys.modules,
    }


def export_capture(capture_path: Path, output: Path) -> dict[str, object]:
    """Write the frozen capture as TRC and return its source-bound receipt.

    Postcondition: decoding the written TRC reproduces the capture fingerprint
    exactly (TRC values are written with round-trip precision), otherwise
    ``ValueError`` is raised and no receipt is produced.
    """
    capture = tour_capture_contract.load_tour_capture(capture_path)
    write_trc(capture, output, rate_hz=capture.rate_hz, units=TRC_UNITS)
    restored = read_trc(output)
    if restored.labels != capture.labels or not np.array_equal(
        restored.valid, capture.valid
    ):
        raise ValueError("TRC export changed label order or missingness")
    fingerprint = capture_fingerprint(capture)
    if capture_fingerprint(restored) != fingerprint:
        raise ValueError("TRC export changed the capture identity (clock or points)")
    clock_error = float(np.max(np.abs(restored.time_s - capture.time_s)))
    position_error = float(
        np.max(
            np.abs(restored.points_m[capture.valid] - capture.points_m[capture.valid])
        )
    )
    return {
        "capture_sha256": capture.source_sha256,
        "trc_sha256": _sha256_file(output),
        "loader_sha256": _sha256_file(Path(tour_capture_contract.__file__)),
        "units": TRC_UNITS,
        "frames": capture.frames,
        "labels": list(capture.labels),
        **fingerprint,
        "max_clock_rounding_s": clock_error,
        "max_position_rounding_m": position_error,
        "source_clock_policy": "arange(frozen_frames)/frozen_rate_hz",
        "preparation_runtime": preparation_runtime(),
    }


def load_transferred_capture(
    trc_path: Path, receipt: Mapping[str, object]
) -> TourCapture:
    """Decode a prepared TRC and prove it still carries the receipt identity.

    Never imports ``ezc3d``.  Raises ``ValueError`` when the receipt is
    incomplete, the units are not metres, the TRC bytes differ, or any of the
    labels, mask, clock or observed points drifted.  The returned capture
    carries the original C3D source hash.
    """
    missing = [key for key in _RECEIPT_KEYS if key not in receipt]
    if missing:
        raise ValueError(f"Preparation receipt is missing {missing}")
    if receipt["units"] != TRC_UNITS:
        raise ValueError(f"Transferred TRC units must be {TRC_UNITS!r}")
    if _sha256_file(trc_path) != receipt["trc_sha256"]:
        raise ValueError("TRC file identity differs from the preparation receipt")
    decoded = read_trc(trc_path)
    capture = TourCapture(
        decoded.time_s,
        decoded.labels,
        decoded.points_m,
        decoded.valid,
        str(receipt["capture_sha256"]),
    )
    fingerprint = capture_fingerprint(capture)
    drift = sorted(key for key, value in fingerprint.items() if receipt[key] != value)
    if list(capture.labels) != receipt["labels"]:
        drift.append("labels")
    if capture.frames != receipt["frames"]:
        drift.append("frames")
    if drift:
        raise ValueError(f"Transferred capture identity drifted: {drift}")
    return capture


def clean_process_environment() -> dict[str, str]:
    """Environment for a boundary child: repository paths only, headless."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(REPO_ROOT), str(REPO_ROOT / "src"))
    )
    environment.setdefault("QT_QPA_PLATFORM", "offscreen")
    environment.setdefault("MPLBACKEND", "Agg")
    return environment


def prepare_capture_in_clean_process(
    capture_path: Path, output: Path, *, python: Path, timeout_s: float = 300.0
) -> dict[str, object]:
    """Run C3D preparation in a fresh ``python`` process; return its receipt.

    The child's failure is raised, never replaced: a nonzero exit becomes
    ``RuntimeError`` with the child's stderr, and a receipt whose source hash
    differs from ``capture_path`` or whose process imported ``opensim``
    becomes ``ValueError``.
    """
    if not Path(capture_path).is_file():
        raise ValueError(f"Capture file does not exist: {capture_path}")
    result = subprocess.run(
        [str(python), "-m", PREPARATION_MODULE, str(capture_path), str(output)],
        cwd=REPO_ROOT,
        env=clean_process_environment(),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"C3D preparation process exited {result.returncode}: "
            f"{result.stderr.strip()[-2000:]}"
        )
    receipt = json.loads(result.stdout.strip().splitlines()[-1])
    if receipt["capture_sha256"] != _sha256_file(capture_path):
        raise ValueError("Preparation receipt source identity differs from capture")
    if receipt["preparation_runtime"]["opensim_loaded"]:
        raise ValueError("Preparation process imported opensim; boundary violated")
    return receipt


def main() -> None:
    """Write only summary metadata to stdout; marker arrays stay in the TRC file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    receipt = export_capture(args.capture, args.output)
    if "opensim" in sys.modules:
        raise RuntimeError("C3D preparation imported opensim; process boundary broken")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
