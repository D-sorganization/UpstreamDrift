"""Process-boundary contracts for C3D preparation vs the native OpenSim SDK (#12065).

On Windows the OpenSim SDK and the Python ``ezc3d`` extension ship
ABI-incompatible ``ezc3d.dll`` builds, so one process may host only one of
them.  These tests never load either native library into the pytest process:
every step that needs ``ezc3d`` (C3D preparation) or ``opensim`` (the native
file-driven consumer) runs in its own clean subprocess, and the source
identity, mask, clock, units and registration are asserted to survive the
TRC hand-off.  Unavailable runtimes skip explicitly with their reason; they
never count as a qualified match.
"""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.diagnostics import frozen_capture_trc
from scripts.diagnostics.frozen_capture_trc import (
    capture_fingerprint,
    clean_process_environment,
    export_capture,
    load_transferred_capture,
    prepare_capture_in_clean_process,
)
from src.engines.physics_engines.opensim.python.tour_matching.registration import (
    CaptureRegistration,
    register_points,
)
from src.shared.python.motion_matching.tour_capture_contract import (
    MARKER_SEGMENTS,
    MARKER_VALIDITY_POLICY,
    TOUR_CAPTURE,
    TourCapture,
)
from src.shared.python.motion_matching.trc_io import read_trc, write_trc

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER_C3D = REPO_ROOT / "data" / "C3D_TA_Driver.c3d"

# Installed in a child before anything else: records every attempt to locate
# a native-SDK module, whether or not it is installed on this host.
_IMPORT_RECORDER = """
import json, sys
class _NativeImportRecorder:
    attempts = []
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] in {"opensim", "ezc3d"}:
            self.attempts.append(name)
        return None
_RECORDER = _NativeImportRecorder()
sys.meta_path.insert(0, _RECORDER)
"""


def _source_capture() -> TourCapture:
    labels = MARKER_SEGMENTS["pelvis"]
    rng = np.random.default_rng(12065)
    points = rng.normal(scale=0.4, size=(6, len(labels), 3))
    valid = np.ones((6, len(labels)), dtype=bool)
    valid[2, 0] = False
    valid[4, 3] = False
    points[~valid] = np.nan
    return TourCapture(np.arange(6) / 360.0, labels, points, valid, "a" * 64)


def _run_child(body: str, *args: str) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_RECORDER + body, *args],
        cwd=REPO_ROOT,
        env=clean_process_environment(),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.fixture
def exported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[TourCapture, Path, dict[str, object]]:
    """Export a synthetic capture through the real preparation function."""
    source = _source_capture()
    monkeypatch.setattr(
        frozen_capture_trc.tour_capture_contract,
        "load_tour_capture",
        lambda path: source,
    )
    trc = tmp_path / "prepared.trc"
    receipt = export_capture(tmp_path / "source.c3d", trc)
    return source, trc, receipt


def test_preparation_process_never_attempts_an_opensim_import(tmp_path: Path) -> None:
    """C3D preparation must not touch the native SDK, even to probe for it."""
    observed = _run_child(
        """
from pathlib import Path
import numpy as np
from scripts.diagnostics import frozen_capture_trc
from src.shared.python.motion_matching.tour_capture_contract import TourCapture
capture = TourCapture(np.arange(3) / 360.0, ("WaistLeft",),
                      np.zeros((3, 1, 3)), np.ones((3, 1), bool), "b" * 64)
frozen_capture_trc.tour_capture_contract.load_tour_capture = lambda path: capture
out = Path(sys.argv[1])
frozen_capture_trc.export_capture(out.with_suffix(".c3d"), out)
print(json.dumps({"attempts": _RECORDER.attempts,
                  "opensim_loaded": "opensim" in sys.modules}))
""",
        str(tmp_path / "child.trc"),
    )
    assert observed["opensim_loaded"] is False
    assert not [name for name in observed["attempts"] if name.startswith("opensim")]


def test_native_consumer_never_attempts_an_ezc3d_import(
    exported: tuple[TourCapture, Path, dict[str, object]], tmp_path: Path
) -> None:
    """The native side decodes the TRC without ever importing ``ezc3d``."""
    _, trc, receipt = exported
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    observed = _run_child(
        """
from pathlib import Path
import src.engines.physics_engines.opensim.python.tour_matching.native_moco_preparation
from scripts.diagnostics.frozen_capture_trc import load_transferred_capture
receipt = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
capture = load_transferred_capture(Path(sys.argv[1]), receipt)
print(json.dumps({"attempts": _RECORDER.attempts, "frames": capture.frames,
                  "ezc3d_loaded": "ezc3d" in sys.modules}))
""",
        str(trc),
        str(receipt_path),
    )
    assert observed["ezc3d_loaded"] is False
    assert not [name for name in observed["attempts"] if name.startswith("ezc3d")]
    assert observed["frames"] == 6


def test_receipt_binds_source_hash_mask_clock_and_units(
    exported: tuple[TourCapture, Path, dict[str, object]],
) -> None:
    source, _, receipt = exported
    assert receipt["capture_sha256"] == source.source_sha256
    assert receipt["units"] == "m"
    assert receipt["frames"] == source.frames
    assert receipt["labels"] == list(source.labels)
    for key, value in capture_fingerprint(source).items():
        assert receipt[key] == value, key


def test_transfer_restores_source_identity_bit_for_bit(
    exported: tuple[TourCapture, Path, dict[str, object]],
) -> None:
    source, trc, receipt = exported
    restored = load_transferred_capture(trc, receipt)
    assert restored.source_sha256 == source.source_sha256
    assert restored.labels == source.labels
    np.testing.assert_array_equal(restored.valid, source.valid)
    assert restored.time_s.tobytes() == source.time_s.tobytes()
    np.testing.assert_array_equal(
        restored.points_m[source.valid], source.points_m[source.valid]
    )
    assert capture_fingerprint(restored) == capture_fingerprint(source)


@pytest.mark.parametrize(
    "field",
    [
        "trc_sha256",
        "valid_mask_sha256",
        "source_clock_sha256",
        "points_sha256",
        "labels_sha256",
    ],
)
def test_transfer_rejects_any_identity_drift(
    exported: tuple[TourCapture, Path, dict[str, object]], field: str
) -> None:
    _, trc, receipt = exported
    with pytest.raises(ValueError, match="identity"):
        load_transferred_capture(trc, {**receipt, field: "0" * 64})


def test_transfer_rejects_a_replaced_input_file(
    exported: tuple[TourCapture, Path, dict[str, object]], tmp_path: Path
) -> None:
    source, _, receipt = exported
    valid = source.valid.copy()
    valid[1, 1] = False
    replaced = tmp_path / "replaced.trc"
    write_trc(replace(source, valid=valid), replaced, rate_hz=360.0)
    with pytest.raises(ValueError, match="identity"):
        load_transferred_capture(replaced, receipt)


def test_transfer_rejects_non_metre_units(
    exported: tuple[TourCapture, Path, dict[str, object]],
) -> None:
    _, trc, receipt = exported
    with pytest.raises(ValueError, match="units"):
        load_transferred_capture(trc, {**receipt, "units": "mm"})


def test_capture_registration_survives_the_boundary(
    exported: tuple[TourCapture, Path, dict[str, object]], tmp_path: Path
) -> None:
    """Registering the transferred capture equals registering the source."""
    source, trc, receipt = exported
    restored = load_transferred_capture(trc, receipt)
    angle = np.deg2rad(30.0)
    registration = CaptureRegistration(
        rotation=np.array(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ]
        ),
        translation=np.array([0.25, -0.1, 1.5]),
    )
    expected = register_points(source.points_m, registration)
    observed = register_points(restored.points_m, registration)
    np.testing.assert_array_equal(observed[source.valid], expected[source.valid])
    registered = write_trc(
        TourCapture(restored.time_s, restored.labels, observed, restored.valid),
        tmp_path / "registered.trc",
        rate_hz=360.0,
    )
    reread = read_trc(registered)
    assert reread.time_s.tobytes() == source.time_s.tobytes()
    np.testing.assert_array_equal(reread.valid, source.valid)


def _require_runtime(module: str, purpose: str) -> None:
    """Skip, with the reason, when a native runtime is absent (no DLL is loaded)."""
    if importlib.util.find_spec(module) is None:
        pytest.skip(
            f"{module} is not installed in {sys.executable}; {purpose} cannot run, "
            "so no capture evidence is produced (not a qualified match)"
        )


@pytest.mark.integration
def test_canonical_capture_crosses_a_clean_process_boundary(tmp_path: Path) -> None:
    """Real C3D preparation in a child; masks match the frozen validity policy."""
    _require_runtime("ezc3d", "C3D preparation")
    if not DRIVER_C3D.is_file():
        pytest.skip(f"canonical capture {DRIVER_C3D} is absent; no evidence")
    receipt = prepare_capture_in_clean_process(
        DRIVER_C3D, tmp_path / "driver.trc", python=Path(sys.executable)
    )
    assert receipt["capture_sha256"] == TOUR_CAPTURE.sha256
    assert receipt["preparation_runtime"]["opensim_loaded"] is False
    capture = load_transferred_capture(tmp_path / "driver.trc", receipt)
    assert capture.labels == TOUR_CAPTURE.labels
    assert capture.frames == TOUR_CAPTURE.frames
    expected_clock = np.arange(TOUR_CAPTURE.frames) / TOUR_CAPTURE.rate_hz
    assert capture.time_s.tobytes() == expected_clock.tobytes()
    for column, label in enumerate(capture.labels):
        assert int(np.count_nonzero(capture.valid[:, column])) == (
            MARKER_VALIDITY_POLICY[label].valid_samples
        ), label


@pytest.mark.integration
def test_native_opensim_reads_the_transferred_trc_in_its_own_process(
    tmp_path: Path,
) -> None:
    """The native SDK consumes the source-bound TRC without importing ezc3d."""
    _require_runtime("ezc3d", "C3D preparation")
    _require_runtime("opensim", "the native OpenSim TRC consumer")
    receipt = prepare_capture_in_clean_process(
        DRIVER_C3D, tmp_path / "driver.trc", python=Path(sys.executable)
    )
    observed = _run_child(
        """
from pathlib import Path
import opensim
table = opensim.TimeSeriesTableVec3(sys.argv[1])
times = table.getIndependentColumn()
print(json.dumps({"attempts": _RECORDER.attempts,
                  "ezc3d_loaded": "ezc3d" in sys.modules,
                  "rows": table.getNumRows(),
                  "labels": list(table.getColumnLabels()),
                  "first": times[0], "last": times[len(times) - 1]}))
""",
        str(tmp_path / "driver.trc"),
    )
    assert observed["ezc3d_loaded"] is False
    assert not [name for name in observed["attempts"] if name.startswith("ezc3d")]
    assert observed["rows"] == receipt["frames"]
    assert observed["labels"] == receipt["labels"]
    assert observed["first"] == 0.0
    assert observed["last"] == (TOUR_CAPTURE.frames - 1) / TOUR_CAPTURE.rate_hz


def test_clean_process_environment_never_inherits_a_contaminating_path() -> None:
    environment = clean_process_environment()
    entries = environment["PYTHONPATH"].split(os.pathsep)
    assert str(REPO_ROOT) in entries
    assert str(REPO_ROOT / "src") in entries
