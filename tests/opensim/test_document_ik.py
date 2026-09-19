"""Tests for OpenSim IK on the shared anthropometric document model (MS-41 #10340).

Verifies marker-body mapping for anthropometric document models, XML and SDK
IKTaskSet authoring with MARKER_VALIDITY_POLICY weights, and MatchingPlant interface.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import pytest

from src.engines.physics_engines.opensim.python.tour_matching import marker_map
from src.shared.python.motion_matching.tour_capture_contract import (
    MARKER_VALIDITY_POLICY,
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
