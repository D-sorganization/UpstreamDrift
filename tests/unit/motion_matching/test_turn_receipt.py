"""Shared receipt ``turn`` block and writer ledger (issue #12042, slice 2)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching import turn_receipt
from src.shared.python.motion_matching.pipeline.receipt_schema import (
    validate_receipt,
)
from src.shared.python.motion_matching.tour_capture_contract import TourCapture
from src.shared.python.motion_matching.turn_receipt import (
    FOLLOW_UP_WRITERS,
    WIRED_WRITERS,
    attach_turn_block,
    attach_turn_block_from_markers,
    build_receipt_turn_block,
    capture_turn_inputs,
    model_points_from_markers,
)
from src.shared.python.swing_comparison.turn import validate_turn_block

REPO = Path(__file__).resolve().parents[3]
N, DT = 201, 0.01
T = np.arange(N) * DT
ADDRESS, TOP, IMPACT, FINISH = 20, 110, 140, 180


def _theta() -> np.ndarray:
    th = np.zeros(N)
    bs = np.linspace(0.0, 1.0, TOP - ADDRESS, endpoint=False)
    th[ADDRESS:TOP] = -140.0 * 0.5 * (1.0 - np.cos(np.pi * bs))
    ds = np.linspace(0.0, 1.0, IMPACT - TOP, endpoint=False)
    th[TOP:IMPACT] = -140.0 * (1.0 - ds**2)
    ft = np.linspace(0.0, 1.0, FINISH - IMPACT, endpoint=False)
    th[IMPACT:FINISH] = 90.0 * np.sin(0.5 * np.pi * ft)
    th[FINISH:] = 90.0
    return th


def _line(turn_deg: np.ndarray, width: float, z: float):
    yaw = np.radians(-90.0 - turn_deg)
    half = width / 2.0
    left = np.column_stack([half * np.cos(yaw), half * np.sin(yaw), np.full(N, z)])
    right = -left + np.array([0.0, 0.0, 2 * z])
    return left, right


def _native_markers(with_club: bool = True) -> dict[str, np.ndarray]:
    turn = -_theta() * (60.0 / 140.0)  # +60 deg at the top (backswing positive)
    mk: dict[str, np.ndarray] = {}
    for (lname, rname), (scale, width, z) in {
        ("WaistLeft", "WaistRight"): (0.5, 0.30, 0.9),
        ("BackLeft", "BackRight"): (1.5, 0.30, 1.3),
        ("LShoulderBack", "RShoulderBack"): (1.7, 0.40, 1.4),
    }.items():
        mk[lname], mk[rname] = _line(turn * scale, width, z)
    if with_club:
        th = np.radians(_theta())
        head = np.column_stack([np.sin(th), np.zeros(N), -np.cos(th) + 0.1])
        grip = 0.3 * head + np.array([0.0, 0.0, 0.7])
        for i in (1, 2, 3):
            mk[f"Marker_2:2:{i}"] = head
            mk[f"Marker_3:3:{i}"] = grip
    return mk


def _capture() -> TourCapture:
    mk = _native_markers()
    labels = tuple(mk)
    native = np.stack([mk[k] for k in labels], axis=1)  # (N, L, 3), Z up
    y_up = np.stack([native[..., 0], native[..., 2], -native[..., 1]], axis=-1)
    return TourCapture(T, labels, y_up, np.ones((N, len(labels)), dtype=bool))


@pytest.mark.unit
class TestTurnReceiptBlock:
    def test_capture_inputs_map_to_native_world(self) -> None:
        cap = capture_turn_inputs(_capture())
        np.testing.assert_allclose(
            cap.markers["WaistLeft"], _native_markers()["WaistLeft"], atol=1e-12
        )
        assert cap.events.address_idx < cap.events.top_idx < cap.events.impact_idx

    def test_block_markers_and_model_agree_when_model_is_exact(self) -> None:
        mk = _native_markers()
        points = model_points_from_markers(
            list(mk), np.stack([mk[k] for k in mk], axis=1)
        )
        block = build_receipt_turn_block(
            _capture(), model_time_s=T, model_points=points, model_source="unit"
        )
        validate_turn_block(block)
        json.dumps(block, allow_nan=False)
        top_pelvis = block["markers"]["pelvis"]["top_deg"]
        assert top_pelvis == pytest.approx(0.5 * 60.0, abs=2.0)
        assert block["model"]["pelvis"]["top_deg"] == pytest.approx(top_pelvis)
        assert block["markers"]["x_factor"]["top_deg"] == pytest.approx(
            block["markers"]["upper_trunk"]["top_deg"]
            - block["markers"]["pelvis"]["top_deg"]
        )
        assert block["model"]["pelvis"]["points"] == ["WaistLeft", "WaistRight"]

    def test_capture_without_valid_data_reports_unavailable_lines(self) -> None:
        cap = _capture()
        empty = TourCapture(
            cap.time_s,
            cap.labels,
            np.full_like(cap.points_m, np.nan),
            np.zeros_like(cap.valid),
        )
        block = build_receipt_turn_block(
            empty, model_time_s=T, model_points={}, model_source="unit"
        )
        validate_turn_block(block)
        for line in ("shoulder_girdle", "upper_trunk", "pelvis", "x_factor"):
            entry = block["markers"][line]
            assert entry["status"] == "unavailable" and entry["reason"]
            assert entry["top_deg"] is None  # never zero

    def test_event_detection_failure_gives_unavailable_block(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*_a: object, **_k: object) -> None:
            raise ValueError("no club motion")

        monkeypatch.setattr(turn_receipt, "detect_events", boom)
        block = build_receipt_turn_block(
            _capture(), model_time_s=T, model_points={}, model_source="unit"
        )
        validate_turn_block(block)
        assert "no club motion" in block["unavailable_reason"]

    def test_truncated_model_horizon_reports_null_not_last_frame(self) -> None:
        mk = _native_markers()
        short = slice(0, 60)  # ends before the top of the backswing
        pts = {k: v[short] for k, v in mk.items()}
        block = build_receipt_turn_block(
            _capture(), model_time_s=T[short], model_points=pts, model_source="unit"
        )
        assert block["model"]["pelvis"]["top_deg"] is None
        assert block["model"]["pelvis"]["impact_deg"] is None
        assert block["markers"]["pelvis"]["top_deg"] is not None

    def test_attach_sets_key_and_validates_in_receipt_schema(self) -> None:
        mk = _native_markers()
        receipt = attach_turn_block_from_markers(
            {},
            _capture(),
            model_time_s=T,
            labels=list(mk),
            model_markers_m=np.stack([mk[k] for k in mk], axis=1),
            model_source="unit",
        )
        validate_turn_block(receipt["turn"])
        bad = {"ground": {}, "address": {}, "ik": {}, "dynamics": {}, "turn": {}}
        with pytest.raises(ValueError, match="^turn:"):
            validate_receipt(bad)

    def test_shape_mismatch_is_recorded_not_raised(self) -> None:
        receipt = attach_turn_block_from_markers(
            {},
            _capture(),
            model_time_s=T,
            labels=["a", "b"],
            model_markers_m=np.zeros((N, 3, 3)),
            model_source="unit",
        )
        assert "unavailable_reason" in receipt["turn"]

    def test_contracts(self) -> None:
        with pytest.raises(TypeError):
            attach_turn_block(
                [], _capture(), model_time_s=T, model_points={}, model_source="u"
            )  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="markers_m"):
            model_points_from_markers(["a"], np.zeros((N, 2, 3)))


@pytest.mark.unit
class TestWriterLedger:
    """Every matched-swing receipt writer is wired or an explicit follow-up."""

    def test_ledger_paths_exist_and_are_disjoint(self) -> None:
        wired, follow = set(WIRED_WRITERS), set(FOLLOW_UP_WRITERS)
        assert wired.isdisjoint(follow)
        for rel in wired | follow:
            assert (REPO / rel).is_file(), rel
        assert all(FOLLOW_UP_WRITERS.values()), "follow-ups need a reason"

    def test_wired_writers_call_the_shared_helper(self) -> None:
        for rel in WIRED_WRITERS:
            text = (REPO / rel).read_text(encoding="utf-8")
            assert "attach_turn_block" in text or "_attach_turn_block" in text, rel

    def test_helper_module_has_no_threshold_language(self) -> None:
        text = Path(turn_receipt.__file__).read_text(encoding="utf-8").lower()
        assert "tolerance" not in text and "threshold =" not in text
