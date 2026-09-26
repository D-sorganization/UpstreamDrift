"""Tests for OpenSim IK on the shared anthropometric document model (MS-41 #10340).

Verifies marker-body mapping for anthropometric document models, XML and SDK
IKTaskSet authoring with MARKER_VALIDITY_POLICY weights, and MatchingPlant interface.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch
import defusedxml.ElementTree as ET

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.tour_matching import marker_map
from src.engines.physics_engines.opensim.python.tour_matching.document_ik import (
    _build_receipt_dict,
    run_document_ik,
)
from src.shared.python.motion_matching.tour_capture_contract import (
    MARKER_VALIDITY_POLICY,
    TourCapture,
    tracked_labels,
)

HAS_OPENSIM = importlib.util.find_spec("opensim") is not None
ROOT = Path(__file__).resolve().parents[2]
DRIVER_SPEC = (
    ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
)
DRIVER_OSIM = (
    ROOT
    / "src/engines/physics_engines/opensim/models/generated/full_body_anthro_driver.osim"
)


@pytest.mark.unit
def test_anthro_document_marker_bodies_covers_all_tracked_labels() -> None:
    """Document model marker map must cover all 34 tracked capture labels."""
    mapping = marker_map.ANTHRO_DOCUMENT_MARKER_BODIES
    assert set(mapping.keys()) == set(tracked_labels())
    assert len(mapping) == 34
    assert mapping["HeadTop"] == "Head"
    assert mapping["HeadFront"] == "Head"
    assert mapping["HeadSide"] == "Head"
    assert mapping["WaistLeft"] == "Hip"
    assert mapping["BackTop"] == "Spine"
    assert mapping["Marker_2:2:1"] == "Clubhead"
    assert mapping["LKneeOut"] == "femur_l"
    assert mapping["RAnkleOut"] == "tibia_r"


@pytest.mark.unit
def test_write_ik_tasks_xml_generates_valid_taskset() -> None:
    """Pure-XML IKTaskSet serialization matches OpenSim 4.0 schema and policy weights."""
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "ik_tasks.xml"
        marker_map.write_ik_tasks_xml(out_path)
        assert out_path.is_file()

        tree = ET.parse(out_path)
        root = tree.getroot()
        assert root.tag == "OpenSimDocument"
        assert root.attrib.get("Version") in ("40000", "40500", "40001", "30000")

        taskset = root.find("IKTaskSet")
        assert taskset is not None
        objects = taskset.find("objects")
        assert objects is not None

        tasks = objects.findall("IKMarkerTask")
        assert len(tasks) == 34

        for task in tasks:
            name = task.attrib.get("name")
            assert name in tracked_labels()
            apply_val = task.find("apply")
            assert apply_val is not None and apply_val.text == "true"
            weight_val = task.find("weight")
            assert weight_val is not None
            expected_weight = MARKER_VALIDITY_POLICY.weight_for(name, is_valid=True)
            assert float(weight_val.text) == pytest.approx(expected_weight)


@pytest.mark.unit
def test_opensim_plant_registration() -> None:
    """OpensimMatchingPlant must be registered and retrievable from plant factory."""
    from src.shared.python.motion_matching.pipeline.plant import (
        available_engines,
        get_plant,
    )

    assert "opensim" in available_engines()

    if DRIVER_SPEC.exists():
        plant = get_plant("opensim", DRIVER_SPEC.read_bytes())
        assert plant.engine_name == "opensim"
        assert len(plant.coordinate_order) == 44
        assert plant.ground_plane is not None


@pytest.mark.unit
def test_document_ik_native_run_on_stride() -> None:
    """Run document IK natively if OpenSim is installed, testing 5-frame stride."""
    if not HAS_OPENSIM or not DRIVER_OSIM.exists():
        pytest.skip("OpenSim SDK or driver osim not present")

    from src.engines.physics_engines.opensim.python.tour_matching.document_ik import (
        run_document_ik,
    )

    trc_path = (
        ROOT
        / "docs/development/opensim_tour_matching/evidence/tour_average_tracked.trc"
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        res = run_document_ik(
            model_path=DRIVER_OSIM,
            trc_path=trc_path,
            out_dir=Path(tmpdir),
            spec_path=DRIVER_SPEC,
            stride=5,
            max_frames=10,
        )
        assert res["whole_marker_rmse_m"] > 0.0
        assert res["receipt_path"].is_file()
        assert res["candidate_path"].is_file()
        assert res["mot_path"].is_file()


def _make_dummy_files(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    """Helper to create minimal synthetic files for receipt testing."""
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps({"coordinate_order": ["pos_x", "pos_y"]}), encoding="utf-8"
    )
    osim_path = tmp_path / "model.osim"
    osim_path.write_text("<OpenSimDocument />", encoding="utf-8")
    cand_path = tmp_path / "candidate.npz"
    cand_path.write_bytes(b"dummy_candidate_bytes")
    trc_path = tmp_path / "capture.trc"
    trc_path.write_bytes(b"dummy_trc_bytes")
    canonical_path = tmp_path / "canonical_receipt.json"
    canonical_path.write_text(
        json.dumps(
            {
                "ik": {"marker_rms_m": 0.0520, "frames": 654},
                "acceptance": {"status": "QUALIFIED"},
            }
        ),
        encoding="utf-8",
    )
    return spec_path, osim_path, cand_path, trc_path, canonical_path


@pytest.mark.unit
def test_build_receipt_dict_excludes_inherited_sections(tmp_path: Path) -> None:
    """Receipt must be built from scratch and exclude inherited MuJoCo sections."""
    spec, osim, cand, trc, canonical = _make_dummy_files(tmp_path)
    receipt = _build_receipt_dict(
        spec_path=spec,
        osim_path=osim,
        candidate_path=cand,
        trc_path=trc,
        labels=["HeadTop", "WaistLeft"],
        whole_rmse=0.053,
        seg_rms_dict={"head": 0.04, "trunk": 0.05},
        elapsed_sec=1.23,
        frames=42,
        canonical_receipt_path=canonical,
    )

    # Must NOT inherit MuJoCo stages or models
    for key in (
        "dynamics",
        "address",
        "hip_calibration",
        "ground",
        "club",
        "grip_rotation_deg",
        "posture_top_of_backswing",
    ):
        assert key not in receipt, f"Key '{key}' must not be in receipt"

    # ik block must only contain OpenSim-computed fields
    assert "ik" in receipt
    ik_block = receipt["ik"]
    assert ik_block["frames"] == 42
    assert ik_block["marker_rms_m"] == pytest.approx(0.053)
    assert ik_block["segment_rms_m"] == {"head": 0.04, "trunk": 0.05}
    for ik_key in (
        "closure_error_max_m",
        "lowest_sphere_height_min_m",
        "lowest_sphere_height_max_m",
        "attachments_m",
        "reference",
        "calibration",
        "segment_scaling",
    ):
        assert ik_key not in ik_block, f"IK key '{ik_key}' must not be inherited"


@pytest.mark.unit
def test_build_receipt_dict_accepted_is_false_even_with_ik_delta_under_5mm(
    tmp_path: Path,
) -> None:
    """Even when IK delta <= 5 mm, accepted must be False and status IK_PARITY_ONLY."""
    spec, osim, cand, trc, canonical = _make_dummy_files(tmp_path)
    # canonical is 0.0520; delta is 0.001 m <= 0.005 m
    receipt = _build_receipt_dict(
        spec_path=spec,
        osim_path=osim,
        candidate_path=cand,
        trc_path=trc,
        labels=["HeadTop"],
        whole_rmse=0.053,
        seg_rms_dict={"head": 0.053},
        elapsed_sec=1.0,
        frames=10,
        canonical_receipt_path=canonical,
    )
    acceptance = receipt["acceptance"]
    assert acceptance["is_physically_accepted"] is False
    assert acceptance["status"] == "IK_PARITY_ONLY"
    assert acceptance["gates"][0]["status"] == "passed"
    assert acceptance["gates"][1]["name"] == "dynamics_stage"
    assert acceptance["gates"][1]["status"] == "not_run"


@pytest.mark.unit
def test_build_receipt_dict_rejected_when_ik_delta_exceeds_5mm(tmp_path: Path) -> None:
    """When IK delta > 5 mm, accepted is False and status is REJECTED."""
    spec, osim, cand, trc, canonical = _make_dummy_files(tmp_path)
    # canonical is 0.0520; whole_rmse is 0.070 (delta = 0.018 > 0.005)
    receipt = _build_receipt_dict(
        spec_path=spec,
        osim_path=osim,
        candidate_path=cand,
        trc_path=trc,
        labels=["HeadTop"],
        whole_rmse=0.070,
        seg_rms_dict={"head": 0.070},
        elapsed_sec=1.0,
        frames=10,
        canonical_receipt_path=canonical,
    )
    acceptance = receipt["acceptance"]
    assert acceptance["is_physically_accepted"] is False
    assert acceptance["status"] == "REJECTED"
    assert acceptance["gates"][0]["status"] == "failed"


@pytest.mark.unit
def test_build_receipt_dict_missing_canonical_raises_file_not_found(
    tmp_path: Path,
) -> None:
    """Missing canonical receipt must raise FileNotFoundError with a clear message (DbC)."""
    spec, osim, cand, trc, _ = _make_dummy_files(tmp_path)
    nonexistent = tmp_path / "does_not_exist_receipt.json"
    with pytest.raises(FileNotFoundError, match="Canonical MuJoCo receipt not found"):
        _build_receipt_dict(
            spec_path=spec,
            osim_path=osim,
            candidate_path=cand,
            trc_path=trc,
            labels=["HeadTop"],
            whole_rmse=0.053,
            seg_rms_dict={"head": 0.053},
            elapsed_sec=1.0,
            frames=10,
            canonical_receipt_path=nonexistent,
        )


@pytest.mark.unit
def test_build_receipt_dict_canonical_missing_rmse_raises_value_error(
    tmp_path: Path,
) -> None:
    """Canonical receipt missing marker_rms_m must raise ValueError without fallback."""
    spec, osim, cand, trc, canonical = _make_dummy_files(tmp_path)
    canonical.write_text(json.dumps({"ik": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing 'ik.marker_rms_m'"):
        _build_receipt_dict(
            spec_path=spec,
            osim_path=osim,
            candidate_path=cand,
            trc_path=trc,
            labels=["HeadTop"],
            whole_rmse=0.053,
            seg_rms_dict={"head": 0.053},
            elapsed_sec=1.0,
            frames=10,
            canonical_receipt_path=canonical,
        )


@pytest.mark.unit
def test_build_receipt_dict_frames_equals_computed_input_length(
    tmp_path: Path,
) -> None:
    """Frames in receipt must equal actual computed frame count, not hardcoded 654."""
    spec, osim, cand, trc, canonical = _make_dummy_files(tmp_path)
    for frame_count in (1, 17, 100):
        receipt = _build_receipt_dict(
            spec_path=spec,
            osim_path=osim,
            candidate_path=cand,
            trc_path=trc,
            labels=["HeadTop"],
            whole_rmse=0.053,
            seg_rms_dict={"head": 0.053},
            elapsed_sec=1.0,
            frames=frame_count,
            canonical_receipt_path=canonical,
        )
        assert receipt["ik"]["frames"] == frame_count


@pytest.mark.unit
def test_run_document_ik_monkeypatched_synthetic_end_to_end(tmp_path: Path) -> None:
    """Test run_document_ik end-to-end with monkeypatched OpenSim calls."""
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps({"coordinate_order": ["tx", "ty"]}),
        encoding="utf-8",
    )
    model_path = tmp_path / "model.osim"
    model_path.write_text("<OpenSimDocument />", encoding="utf-8")
    canonical_path = tmp_path / "canonical.json"
    canonical_path.write_text(
        json.dumps({"ik": {"marker_rms_m": 0.050}}),
        encoding="utf-8",
    )

    n_frames = 15
    times = np.linspace(0.0, 0.14, n_frames)
    labels = ("HeadTop", "WaistLeft")
    points_m = np.zeros((n_frames, len(labels), 3))
    valid = np.ones((n_frames, len(labels)), dtype=bool)
    mock_capture = TourCapture(
        time_s=times, labels=labels, points_m=points_m, valid=valid
    )

    q_mock = np.zeros((n_frames, 2))
    pred_markers_mock = np.zeros((n_frames, len(labels), 3)) + 0.01

    out_dir = tmp_path / "output"

    trc_dummy = tmp_path / "capture.trc"
    trc_dummy.write_text("dummy trc", encoding="utf-8")

    mock_opensim = MagicMock()
    with (
        patch.dict("sys.modules", {"opensim": mock_opensim}),
        patch(
            "src.engines.physics_engines.opensim.python.tour_matching.document_ik.read_trc",
            return_value=mock_capture,
        ),
        patch(
            "src.engines.physics_engines.opensim.python.tour_matching.document_ik._run_opensim_ik_tool"
        ),
        patch(
            "src.engines.physics_engines.opensim.python.tour_matching.document_ik._load_mot_table",
            return_value=(times, q_mock),
        ),
        patch(
            "src.engines.physics_engines.opensim.python.tour_matching.document_ik._forward_marker_positions",
            return_value=pred_markers_mock,
        ),
        patch(
            "src.engines.physics_engines.opensim.python.tour_matching.document_ik.generate_ik_overlay_gif"
        ),
        patch(
            "src.engines.physics_engines.opensim.python.tour_matching.document_ik._build_and_save_candidate",
            return_value=tmp_path / "cand.npz",
        ),
    ):
        (tmp_path / "cand.npz").write_bytes(b"cand")
        res = run_document_ik(
            model_path=model_path,
            trc_path=trc_dummy,
            out_dir=out_dir,
            spec_path=spec_path,
            canonical_receipt_path=canonical_path,
        )

        assert res["receipt_path"].is_file()
        receipt = json.loads(res["receipt_path"].read_text(encoding="utf-8"))
        assert receipt["ik"]["frames"] == n_frames
        assert "dynamics" not in receipt
        assert "address" not in receipt
        assert "hip_calibration" not in receipt
        assert receipt["acceptance"]["is_physically_accepted"] is False
