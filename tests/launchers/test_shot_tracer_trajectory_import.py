"""Tests for the ADR-0047 H2 trajectory-record import module (#9351).

Headless and Qt-free: :mod:`src.launchers._shot_tracer_trajectory_import`
never touches Qt, so these tests run without ``qapp``/``qtbot``. Every
test exercises the *real* vendored ``flight_interchange`` reader — no
mocking of ``ball_flight_trajectory_from_json`` — so a refusal reason
asserted here is the reader's own wording, not a stand-in for it.

Two producer fixtures cover both flight-model families named in ADR-0047:

- :func:`_ud_family_record` goes through the real H1 export path
  (:mod:`src.shared.python.physics.flight_trajectory_export`) against a
  real simulated :class:`FlightResult`.
- :func:`_tools_family_record` hand-authors a ``swing_sim.flight`` record
  per the documented wire, deliberately without importing Tools' own
  exporter (which pulls in Tools' flight models) — same posture as the
  H1 module's own docstring recommends for cross-repo producers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.launchers._shot_tracer_trajectory_import import (
    ImportedTrajectoryCurve,
    TrajectoryImportError,
    import_trajectory_record,
)
from src.shared.python.physics.flight_models import (
    FlightModelRegistry,
    FlightModelType,
    UnifiedLaunchConditions,
)
from src.shared.python.physics.flight_trajectory_export import (
    APP_FRAME_ID,
    FLIGHT_FRAME_ID,
    flight_result_to_trajectory_record,
    trajectory_parameter_digest,
    trajectory_record_to_json,
)

pytestmark = pytest.mark.unit


def _ud_family_record() -> dict[str, Any]:
    """Build a real ``ud.flight_models`` record via the H1 export path."""
    launch = UnifiedLaunchConditions.from_imperial(
        ball_speed_mph=163.0, launch_angle_deg=11.0, spin_rate_rpm=2500.0
    )
    model = FlightModelRegistry.get_model(FlightModelType.WATERLOO_PENNER)
    result = model.simulate(launch)
    return flight_result_to_trajectory_record(
        result, model_type=FlightModelType.WATERLOO_PENNER
    )


def _tools_family_record() -> dict[str, Any]:
    """Hand-author a valid ``swing_sim.flight`` record per the wire.

    Deliberately does not import Tools' vendored exporter
    (``flight_interchange.adapters``), which pulls in Tools' own flight
    models; the wire is a documented JSON contract, the same posture
    the H1 exporter's module docstring describes for a cross-repo
    producer.
    """
    parameters = {"cd": 0.21, "cl": 0.15, "spin_decay": 0.08}
    digest = trajectory_parameter_digest(parameters)
    return {
        "format": "swing_sim.ball_flight_trajectory/1",
        "source_id": "swing_sim.flight:MacDonald-Hanzely",
        "frame_id": FLIGHT_FRAME_ID,
        "channels": ["velocity_mps"],
        "provenance": {
            "model_family": "swing_sim.flight",
            "model_name": "MacDonald-Hanzely",
            "parameter_digest": digest,
        },
        "samples": [
            {
                "time_s": 0.0,
                "position_m": [0.0, 0.0, 0.0],
                "velocity_mps": [60.0, 0.0, 15.0],
            },
            {
                "time_s": 0.5,
                "position_m": [29.0, 0.0, 6.5],
                "velocity_mps": [58.0, 0.0, 8.0],
            },
            {
                "time_s": 1.0,
                "position_m": [56.0, 0.0, -2.0],
                "velocity_mps": [55.0, 0.0, -9.0],
            },
        ],
    }


def _write_record(tmp_path: Path, record: dict[str, Any], name: str) -> Path:
    path = tmp_path / name
    path.write_text(trajectory_record_to_json(record), encoding="utf-8")
    return path


def test_import_ud_family_record(tmp_path: Path) -> None:
    """A native UD-family record round-trips through export -> import."""
    record = _ud_family_record()
    path = _write_record(tmp_path, record, "ud_record.json")

    curve = import_trajectory_record(path)

    assert isinstance(curve, ImportedTrajectoryCurve)
    assert curve.model_family == "ud.flight_models"
    assert curve.model_name == "Waterloo/Penner"
    assert curve.label == "ud.flight_models / Waterloo/Penner"
    assert curve.frame_id == FLIGHT_FRAME_ID
    assert curve.positions.shape == (len(record["samples"]), 3)
    np.testing.assert_allclose(curve.positions[0], record["samples"][0]["position_m"])
    np.testing.assert_allclose(curve.positions[-1], record["samples"][-1]["position_m"])


def test_import_tools_family_record(tmp_path: Path) -> None:
    """A Tools-family record (hand-authored per the wire) imports too."""
    record = _tools_family_record()
    path = _write_record(tmp_path, record, "tools_record.json")

    curve = import_trajectory_record(path)

    assert curve.model_family == "swing_sim.flight"
    assert curve.model_name == "MacDonald-Hanzely"
    assert curve.label == "swing_sim.flight / MacDonald-Hanzely"
    assert curve.frame_id == FLIGHT_FRAME_ID
    assert curve.positions.shape == (3, 3)
    np.testing.assert_allclose(curve.positions[1], [29.0, 0.0, 6.5])


def test_import_refuses_unsupported_frame(tmp_path: Path) -> None:
    """A record declaring the wire's other valid frame is refused by name.

    ``app_xtarget_yup_zright`` is a legal wire frame — the vendored
    reader accepts the record — but Shot Tracer has not implemented a
    converter for it, so the refusal is this module's own, naming the
    unsupported frame explicitly.
    """
    record = _ud_family_record()
    record["frame_id"] = APP_FRAME_ID
    path = _write_record(tmp_path, record, "bad_frame.json")

    with pytest.raises(TrajectoryImportError) as excinfo:
        import_trajectory_record(path)

    message = str(excinfo.value)
    assert "unsupported frame" in message
    assert APP_FRAME_ID in message


def test_import_refuses_unknown_field(tmp_path: Path) -> None:
    """An extra top-level field is refused with the reader's own reason."""
    record = _ud_family_record()
    record["extra_field"] = "not part of the wire"
    path = _write_record(tmp_path, record, "unknown_field.json")

    with pytest.raises(TrajectoryImportError) as excinfo:
        import_trajectory_record(path)

    assert "unknown trajectory fields" in str(excinfo.value)
    assert "extra_field" in str(excinfo.value)


def test_import_refuses_malformed_json(tmp_path: Path) -> None:
    """Malformed JSON is refused, not raised as an uncaught exception."""
    path = tmp_path / "malformed.json"
    path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(TrajectoryImportError):
        import_trajectory_record(path)


def test_import_refuses_missing_file(tmp_path: Path) -> None:
    """A missing path is refused with a reason naming the path."""
    missing = tmp_path / "does_not_exist.json"

    with pytest.raises(TrajectoryImportError) as excinfo:
        import_trajectory_record(missing)

    assert "cannot read" in str(excinfo.value)


def test_import_rejects_non_path_argument() -> None:
    with pytest.raises(TypeError, match="pathlib.Path"):
        import_trajectory_record("not-a-path")  # type: ignore[arg-type]


def test_import_refuses_when_vendor_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing/unresolvable Tools checkout is refused, never a crash."""
    import src.launchers._shot_tracer_trajectory_import as import_module

    monkeypatch.setattr(import_module, "resolve_tools_repo", lambda *_a, **_k: None)
    record = _ud_family_record()
    path = _write_record(tmp_path, record, "ud_record.json")

    with pytest.raises(TrajectoryImportError, match="Tools repository not found"):
        import_trajectory_record(path)


def test_import_refuses_malformed_provenance(tmp_path: Path) -> None:
    """A provenance section missing a mandatory key is refused by name."""
    record = _tools_family_record()
    del record["provenance"]["parameter_digest"]
    path = _write_record(tmp_path, record, "bad_provenance.json")

    with pytest.raises(TrajectoryImportError) as excinfo:
        import_trajectory_record(path)

    assert "provenance" in str(excinfo.value)
