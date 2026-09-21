"""Unit tests for Tour Baselines measurement map (TB-01 #10586)."""

import pytest

from src.shared.python.tour_baselines.measurement_map import (
    MEASUREMENT_MAP_DRIVER,
    MEASUREMENT_MAP_IRON,
    MeasurementClass,
    get_measurement_map,
)

pytestmark = pytest.mark.unit


def test_measurement_classes_defined() -> None:
    assert MeasurementClass.OBSERVED_SURFACE == "observed_surface"
    assert MeasurementClass.INFERRED_JOINT_CENTER == "inferred_joint_center"
    assert MeasurementClass.OBSERVED_CLUSTER_CENTROID == "observed_cluster_centroid"
    assert MeasurementClass.CALIBRATED_CLUB_POINT == "calibrated_club_point"
    assert MeasurementClass.UNASSIGNED_OR_SENTINEL == "unassigned_or_sentinel"


def test_driver_measurement_map_contents() -> None:
    driver_map = get_measurement_map("driver")
    assert len(driver_map) == 38
    assert "Uname*38" in driver_map
    assert "pelvis" not in driver_map
    assert (
        driver_map["Uname*38"].measurement_class
        == MeasurementClass.UNASSIGNED_OR_SENTINEL
    )

    # Surface markers
    assert driver_map["HeadTop"].measurement_class == MeasurementClass.OBSERVED_SURFACE
    assert driver_map["HeadTop"].is_observed is True
    assert (
        driver_map["WaistLeft"].measurement_class == MeasurementClass.OBSERVED_SURFACE
    )

    # Clusters
    for lbl in ("Marker_2:2:1", "Marker_2:2:2", "Marker_2:2:3"):
        assert (
            driver_map[lbl].measurement_class
            == MeasurementClass.OBSERVED_CLUSTER_CENTROID
        )
        assert driver_map[lbl].segment == "club_grip"
    for lbl in ("Marker_3:3:1", "Marker_3:3:2", "Marker_3:3:3"):
        assert (
            driver_map[lbl].measurement_class
            == MeasurementClass.OBSERVED_CLUSTER_CENTROID
        )
        assert driver_map[lbl].segment == "club_head"

    # Sentinel
    assert (
        driver_map["Marker_0:0:0"].measurement_class
        == MeasurementClass.UNASSIGNED_OR_SENTINEL
    )


def test_iron_measurement_map_contents() -> None:
    iron_map = get_measurement_map("iron")
    assert len(iron_map) == 38
    assert "pelvis" in iron_map
    assert "Uname*38" not in iron_map
    assert (
        iron_map["pelvis"].measurement_class == MeasurementClass.UNASSIGNED_OR_SENTINEL
    )


def test_calibrated_clubface_and_impact_points_are_explicitly_unavailable() -> None:
    driver_map = get_measurement_map("driver")
    # Neither raw capture has calibrated clubface orientation or impact point
    for semantics in driver_map.values():
        assert semantics.measurement_class != MeasurementClass.CALIBRATED_CLUB_POINT


def test_invalid_capture_kind_raises() -> None:
    with pytest.raises(ValueError, match="Unknown capture kind"):
        get_measurement_map("putter")
