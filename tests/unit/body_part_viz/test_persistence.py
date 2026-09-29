"""Unit tests for ``body_part_viz.persistence`` (JSON v2 viz-set I/O)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.body_part_viz import (
    BindingKind,
    MarkerBinding,
    SegmentVizSet,
    SegmentVizSpec,
    ShapeTheme,
    migrate_v1_to_v2,
)
from src.shared.python.body_part_viz.persistence import (
    SCHEMA_VERSION,
    VALID_FITTER_KINDS,
    VALID_SHAPE_KINDS,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BETWEEN_TWO_BINDING = MarkerBinding(
    kind=BindingKind.BETWEEN_TWO,
    marker_names=("WaistLeft", "WaistRight"),
    rest_dimensions=(0.32,),
)

_THEME = ShapeTheme(color="#1f77b4", opacity=0.8, group="pelvis")


def _spec(
    shape_kind: str, shape_params: dict, fitter_kind: str = "between_two"
) -> SegmentVizSpec:
    return SegmentVizSpec(
        binding=_BETWEEN_TWO_BINDING,
        shape_kind=shape_kind,  # type: ignore[arg-type]
        shape_params=dict(shape_params),
        fitter_kind=fitter_kind,  # type: ignore[arg-type]
        theme=_THEME,
    )


# ---------------------------------------------------------------------------
# Round-trip per shape kind
# ---------------------------------------------------------------------------


def test_line_round_trip(tmp_path: Path) -> None:
    spec = _spec("line", {"length": 0.4})
    vs = SegmentVizSet(segments=(spec,))
    out = tmp_path / "v.json"
    vs.save(out)
    loaded = SegmentVizSet.load(out)
    assert loaded == vs
    assert loaded.segments[0].shape_params == {"length": 0.4}


def test_cylinder_round_trip(tmp_path: Path) -> None:
    spec = _spec("cylinder", {"length": 0.32, "radius": 0.04, "n_facets": 24})
    vs = SegmentVizSet(segments=(spec,))
    out = tmp_path / "v.json"
    vs.save(out)
    loaded = SegmentVizSet.load(out)
    assert loaded.segments[0].shape_params == {
        "length": 0.32,
        "radius": 0.04,
        "n_facets": 24,
    }


def test_cylinder_default_n_facets() -> None:
    spec = _spec("cylinder", {"length": 0.5, "radius": 0.05})
    assert spec.shape_params["n_facets"] == 16


def test_ellipsoid_round_trip(tmp_path: Path) -> None:
    spec = _spec("ellipsoid", {"a": 0.1, "b": 0.2, "c": 0.3})
    vs = SegmentVizSet(segments=(spec,))
    out = tmp_path / "v.json"
    vs.save(out)
    loaded = SegmentVizSet.load(out)
    p = loaded.segments[0].shape_params
    assert p["a"] == 0.1 and p["b"] == 0.2 and p["c"] == 0.3
    assert p["n_lon"] == 16 and p["n_lat"] == 8


def test_capsule_round_trip(tmp_path: Path) -> None:
    spec = _spec(
        "capsule", {"length": 0.25, "radius": 0.04, "n_facets": 12, "n_lat": 6}
    )
    vs = SegmentVizSet(segments=(spec,))
    out = tmp_path / "v.json"
    vs.save(out)
    loaded = SegmentVizSet.load(out)
    assert loaded.segments[0].shape_params["n_facets"] == 12
    assert loaded.segments[0].shape_params["n_lat"] == 6


def test_mesh_file_round_trip(tmp_path: Path) -> None:
    spec = _spec("mesh_file", {"path": "assets/torso.stl", "max_vertices": 1000})
    vs = SegmentVizSet(segments=(spec,))
    out = tmp_path / "v.json"
    vs.save(out)
    loaded = SegmentVizSet.load(out)
    assert loaded.segments[0].shape_params == {
        "path": "assets/torso.stl",
        "max_vertices": 1000,
    }


def test_mesh_file_default_max_vertices() -> None:
    spec = _spec("mesh_file", {"path": "x.obj"})
    assert spec.shape_params["max_vertices"] == 5000


def test_library_shape_round_trip(tmp_path: Path) -> None:
    spec = _spec("library_shape", {"library_name": "default", "shape_id": "upper_arm"})
    vs = SegmentVizSet(segments=(spec,))
    out = tmp_path / "v.json"
    vs.save(out)
    loaded = SegmentVizSet.load(out)
    assert loaded.segments[0].shape_params == {
        "library_name": "default",
        "shape_id": "upper_arm",
    }


def test_composite_round_trip(tmp_path: Path) -> None:
    spec = _spec(
        "composite",
        {
            "children": [
                {
                    "shape_kind": "cylinder",
                    "shape_params": {"length": 0.2, "radius": 0.03},
                },
                {
                    "shape_kind": "ellipsoid",
                    "shape_params": {"a": 0.05, "b": 0.05, "c": 0.07},
                },
            ]
        },
    )
    vs = SegmentVizSet(segments=(spec,))
    out = tmp_path / "v.json"
    vs.save(out)
    loaded = SegmentVizSet.load(out)
    children = loaded.segments[0].shape_params["children"]
    assert len(children) == 2
    assert children[0]["shape_kind"] == "cylinder"
    assert children[0]["shape_params"]["n_facets"] == 16
    assert children[1]["shape_params"]["n_lon"] == 16


# ---------------------------------------------------------------------------
# Round-trip per fitter kind
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fitter_kind", list(VALID_FITTER_KINDS))
def test_all_fitter_kinds_round_trip(tmp_path: Path, fitter_kind: str) -> None:
    binding = (
        _BETWEEN_TWO_BINDING
        if fitter_kind == "between_two"
        else MarkerBinding(
            kind=BindingKind.CLUSTER,
            marker_names=("M1", "M2", "M3"),
        )
    )
    spec = SegmentVizSpec(
        binding=binding,
        shape_kind="cylinder",
        shape_params={"length": 0.3, "radius": 0.04},
        fitter_kind=fitter_kind,  # type: ignore[arg-type]
        theme=_THEME,
    )
    vs = SegmentVizSet(segments=(spec,))
    out = tmp_path / "v.json"
    vs.save(out)
    loaded = SegmentVizSet.load(out)
    assert loaded.segments[0].fitter_kind == fitter_kind


# ---------------------------------------------------------------------------
# DbC validation
# ---------------------------------------------------------------------------


def test_unknown_shape_kind_raises() -> None:
    with pytest.raises(ValueError, match="unknown shape_kind"):
        SegmentVizSpec(
            binding=_BETWEEN_TWO_BINDING,
            shape_kind="sphere",  # type: ignore[arg-type]
            shape_params={},
            fitter_kind="between_two",
            theme=_THEME,
        )


def test_unknown_fitter_kind_raises() -> None:
    with pytest.raises(ValueError, match="unknown fitter_kind"):
        SegmentVizSpec(
            binding=_BETWEEN_TWO_BINDING,
            shape_kind="line",
            shape_params={"length": 0.3},
            fitter_kind="kalman",  # type: ignore[arg-type]
            theme=_THEME,
        )


def test_missing_shape_params_lists_missing_keys() -> None:
    with pytest.raises(ValueError, match=r"missing required keys: \['radius'\]"):
        _spec("cylinder", {"length": 0.3})


def test_missing_shape_params_multiple_keys() -> None:
    with pytest.raises(ValueError, match=r"missing required keys"):
        _spec("ellipsoid", {"a": 0.1})


def test_shape_params_must_be_dict() -> None:
    with pytest.raises(TypeError, match="shape_params must be a dict"):
        SegmentVizSpec(
            binding=_BETWEEN_TWO_BINDING,
            shape_kind="line",
            shape_params=[],  # type: ignore[arg-type]
            fitter_kind="between_two",
            theme=_THEME,
        )


def test_binding_must_be_marker_binding() -> None:
    with pytest.raises(TypeError, match="binding must be MarkerBinding"):
        SegmentVizSpec(
            binding={"kind": "between_two"},  # type: ignore[arg-type]
            shape_kind="line",
            shape_params={"length": 0.3},
            fitter_kind="between_two",
            theme=_THEME,
        )


def test_theme_must_be_shape_theme() -> None:
    with pytest.raises(TypeError, match="theme must be ShapeTheme"):
        SegmentVizSpec(
            binding=_BETWEEN_TWO_BINDING,
            shape_kind="line",
            shape_params={"length": 0.3},
            fitter_kind="between_two",
            theme={"color": "red"},  # type: ignore[arg-type]
        )


def test_visible_must_be_bool() -> None:
    with pytest.raises(TypeError, match="visible must be bool"):
        SegmentVizSpec(
            binding=_BETWEEN_TWO_BINDING,
            shape_kind="line",
            shape_params={"length": 0.3},
            fitter_kind="between_two",
            theme=_THEME,
            visible="yes",  # type: ignore[arg-type]
        )


def test_composite_children_must_be_list() -> None:
    with pytest.raises(ValueError, match="must be a list"):
        _spec("composite", {"children": "not-a-list"})


def test_composite_child_must_be_dict() -> None:
    with pytest.raises(ValueError, match="must be a dict"):
        _spec("composite", {"children": ["x"]})


def test_composite_child_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown shape_kind"):
        _spec(
            "composite",
            {"children": [{"shape_kind": "sphere", "shape_params": {}}]},
        )


def test_from_dict_unknown_shape_kind() -> None:
    with pytest.raises(ValueError, match="unknown shape_kind"):
        SegmentVizSpec.from_dict(
            {
                "binding": {
                    "kind": "between_two",
                    "marker_names": ["A", "B"],
                },
                "shape_kind": "sphere",
                "shape_params": {},
                "fitter_kind": "between_two",
                "theme": {},
            }
        )


def test_from_dict_unknown_fitter_kind() -> None:
    with pytest.raises(ValueError, match="unknown fitter_kind"):
        SegmentVizSpec.from_dict(
            {
                "binding": {
                    "kind": "between_two",
                    "marker_names": ["A", "B"],
                },
                "shape_kind": "line",
                "shape_params": {"length": 0.3},
                "fitter_kind": "ekf",
                "theme": {},
            }
        )


def test_from_dict_unknown_binding_kind() -> None:
    with pytest.raises(ValueError, match="unknown binding kind"):
        SegmentVizSpec.from_dict(
            {
                "binding": {"kind": "telekinesis", "marker_names": ["A", "B"]},
                "shape_kind": "line",
                "shape_params": {"length": 0.3},
                "fitter_kind": "between_two",
                "theme": {},
            }
        )


def test_from_dict_quat_wrong_length() -> None:
    with pytest.raises(ValueError, match="rest_orientation_quat"):
        SegmentVizSpec.from_dict(
            {
                "binding": {
                    "kind": "between_two",
                    "marker_names": ["A", "B"],
                    "rest_orientation_quat": [1.0, 0.0],
                },
                "shape_kind": "line",
                "shape_params": {"length": 0.3},
                "fitter_kind": "between_two",
                "theme": {},
            }
        )


def test_from_dict_segment_must_be_dict() -> None:
    with pytest.raises(TypeError, match="segment entry must be a dict"):
        SegmentVizSpec.from_dict([1, 2, 3])  # type: ignore[arg-type]


def test_from_dict_binding_must_be_dict() -> None:
    with pytest.raises(TypeError, match="binding entry must be a dict"):
        SegmentVizSpec.from_dict(
            {
                "binding": "not-a-dict",
                "shape_kind": "line",
                "shape_params": {"length": 0.3},
                "fitter_kind": "between_two",
                "theme": {},
            }
        )


def test_from_dict_theme_must_be_dict() -> None:
    with pytest.raises(TypeError, match="theme entry must be a dict"):
        SegmentVizSpec.from_dict(
            {
                "binding": {"kind": "between_two", "marker_names": ["A", "B"]},
                "shape_kind": "line",
                "shape_params": {"length": 0.3},
                "fitter_kind": "between_two",
                "theme": "blue",
            }
        )


def test_from_dict_shape_params_must_be_dict() -> None:
    with pytest.raises(TypeError, match="shape_params must be a dict"):
        SegmentVizSpec.from_dict(
            {
                "binding": {"kind": "between_two", "marker_names": ["A", "B"]},
                "shape_kind": "line",
                "shape_params": [0.3],
                "fitter_kind": "between_two",
                "theme": {},
            }
        )


# ---------------------------------------------------------------------------
# SegmentVizSet validation
# ---------------------------------------------------------------------------


def test_segment_viz_set_default_empty() -> None:
    vs = SegmentVizSet()
    assert vs.schema_version == SCHEMA_VERSION
    assert vs.segments == ()


def test_segment_viz_set_rejects_non_int_schema() -> None:
    with pytest.raises(TypeError, match="schema_version must be int"):
        SegmentVizSet(schema_version="2")  # type: ignore[arg-type]


def test_segment_viz_set_rejects_wrong_schema() -> None:
    with pytest.raises(ValueError, match="only supports schema_version"):
        SegmentVizSet(schema_version=99)


def test_segment_viz_set_rejects_non_tuple_segments() -> None:
    with pytest.raises(TypeError, match="segments must be a tuple"):
        SegmentVizSet(segments=[_spec("line", {"length": 0.3})])  # type: ignore[arg-type]


def test_segment_viz_set_rejects_non_spec_entries() -> None:
    with pytest.raises(TypeError, match="must be SegmentVizSpec"):
        SegmentVizSet(segments=("not-a-spec",))  # type: ignore[arg-type]


def test_from_dict_must_be_dict() -> None:
    with pytest.raises(TypeError, match="payload must be dict"):
        SegmentVizSet.from_dict("hello")  # type: ignore[arg-type]


def test_from_dict_unsupported_schema() -> None:
    with pytest.raises(ValueError, match="unsupported schema_version"):
        SegmentVizSet.from_dict({"schema_version": 99, "segments": []})


def test_from_dict_schema_must_be_int() -> None:
    with pytest.raises(ValueError, match="schema_version must be int"):
        SegmentVizSet.from_dict({"schema_version": "2", "segments": []})


def test_from_dict_segments_must_be_list() -> None:
    with pytest.raises(ValueError, match="'segments' must be a list"):
        SegmentVizSet.from_dict({"schema_version": 2, "segments": {}})


def test_load_missing_path() -> None:
    with pytest.raises(ValueError, match="path must be provided"):
        SegmentVizSet.load(None)  # type: ignore[arg-type]


def test_load_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        SegmentVizSet.load(tmp_path / "nope.json")


def test_save_missing_path() -> None:
    with pytest.raises(ValueError, match="path must be provided"):
        SegmentVizSet().save(None)  # type: ignore[arg-type]


def test_save_creates_parent_dirs(tmp_path: Path) -> None:
    out = tmp_path / "deep" / "nested" / "v.json"
    spec = _spec("line", {"length": 0.3})
    SegmentVizSet(segments=(spec,)).save(out)
    assert out.is_file()


# ---------------------------------------------------------------------------
# v1 → v2 migration
# ---------------------------------------------------------------------------


_V1_SAMPLE = {
    "schema_version": 1,
    "segments": [
        {
            "a": "WaistLeft",
            "b": "WaistRight",
            "geometry": "cylinder",
            "group": "pelvis",
            "visible": True,
            "radius": 0.04,
        },
        {
            "a": "LShoulder",
            "b": "LElbow",
            "geometry": "line",
            "group": "left_arm",
            "visible": False,
            "radius": 0.02,
        },
    ],
}


def test_migrate_v1_to_v2_basic() -> None:
    v2 = migrate_v1_to_v2(_V1_SAMPLE)
    assert v2["schema_version"] == SCHEMA_VERSION
    assert len(v2["segments"]) == 2

    s0 = v2["segments"][0]
    assert s0["binding"]["kind"] == "between_two"
    assert s0["binding"]["marker_names"] == ["WaistLeft", "WaistRight"]
    assert s0["shape_kind"] == "cylinder"
    assert s0["shape_params"]["radius"] == 0.04
    assert s0["fitter_kind"] == "between_two"
    assert s0["theme"]["group"] == "pelvis"

    s1 = v2["segments"][1]
    assert s1["shape_kind"] == "line"
    assert s1["visible"] is False
    assert s1["theme"]["group"] == "left_arm"


def test_migrate_v1_load_and_save_writes_v2(tmp_path: Path) -> None:
    v1_path = tmp_path / "v1.json"
    with v1_path.open("w", encoding="utf-8") as f:
        json.dump(_V1_SAMPLE, f)

    loaded = SegmentVizSet.load(v1_path)
    assert loaded.schema_version == SCHEMA_VERSION
    assert len(loaded.segments) == 2

    out = tmp_path / "v2.json"
    loaded.save(out)
    with out.open("r", encoding="utf-8") as f:
        rewritten = json.load(f)
    assert rewritten["schema_version"] == SCHEMA_VERSION


def test_migrate_v1_rejects_non_dict() -> None:
    with pytest.raises(TypeError, match="must be dict"):
        migrate_v1_to_v2(["not", "a", "dict"])  # type: ignore[arg-type]


def test_migrate_v1_rejects_wrong_version() -> None:
    with pytest.raises(ValueError, match="requires schema_version=1"):
        migrate_v1_to_v2({"schema_version": 2, "segments": []})


def test_migrate_v1_segments_must_be_list() -> None:
    with pytest.raises(ValueError, match="must be a list"):
        migrate_v1_to_v2({"schema_version": 1, "segments": {}})


def test_migrate_v1_entry_must_be_dict() -> None:
    with pytest.raises(ValueError, match="must be dict"):
        migrate_v1_to_v2({"schema_version": 1, "segments": ["x"]})


def test_migrate_v1_invalid_geometry() -> None:
    payload = {
        "schema_version": 1,
        "segments": [{"a": "A", "b": "B", "geometry": "blob", "group": "g"}],
    }
    with pytest.raises(ValueError, match="must be 'line' or 'cylinder'"):
        migrate_v1_to_v2(payload)


def test_migrate_v1_default_geometry_line() -> None:
    payload = {
        "schema_version": 1,
        "segments": [{"a": "A", "b": "B", "group": "g"}],
    }
    v2 = migrate_v1_to_v2(payload)
    assert v2["segments"][0]["shape_kind"] == "line"


# ---------------------------------------------------------------------------
# Multi-segment full sample
# ---------------------------------------------------------------------------


def test_full_sample_round_trip(tmp_path: Path) -> None:
    specs = (
        _spec("line", {"length": 0.4}),
        _spec("cylinder", {"length": 0.32, "radius": 0.04}),
        _spec("ellipsoid", {"a": 0.1, "b": 0.2, "c": 0.3}),
        _spec("capsule", {"length": 0.2, "radius": 0.03}),
        _spec("mesh_file", {"path": "x.stl"}),
        _spec("library_shape", {"library_name": "default", "shape_id": "ua"}),
        _spec(
            "composite",
            {
                "children": [
                    {
                        "shape_kind": "cylinder",
                        "shape_params": {"length": 0.1, "radius": 0.01},
                    }
                ]
            },
        ),
    )
    vs = SegmentVizSet(segments=specs)
    out = tmp_path / "all.json"
    vs.save(out)
    loaded = SegmentVizSet.load(out)
    assert loaded == vs
    # Confirms the full kind matrix is exercised:
    assert {s.shape_kind for s in loaded.segments} == set(VALID_SHAPE_KINDS)


def test_to_dict_rounds_floats() -> None:
    spec = _spec("cylinder", {"length": 0.123456789012345, "radius": 0.04})
    d = spec.to_dict()
    assert d["shape_params"]["length"] == round(0.123456789012345, 9)
