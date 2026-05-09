"""Unit tests for ``body_part_viz.asset_library``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.body_part_viz import (
    BindingKind,
    MarkerBinding,
)
from src.shared.python.body_part_viz.asset_library import ShapeLibrary
from src.shared.python.body_part_viz.shapes import MeshShape


def test_default_loads_without_raising() -> None:
    lib = ShapeLibrary.default()
    assert isinstance(lib, ShapeLibrary)
    # Issue requires 8 default shapes.
    assert len(lib.names()) == 8


def test_default_names_match_expected_set() -> None:
    lib = ShapeLibrary.default()
    expected = {
        "head",
        "torso",
        "upper_arm",
        "forearm",
        "hand",
        "thigh",
        "shin",
        "foot",
    }
    assert set(lib.names()) == expected


def test_get_returns_mesh_shape_for_every_name() -> None:
    lib = ShapeLibrary.default()
    for name in lib.names():
        shape = lib.get(name)
        assert isinstance(shape, MeshShape)
        # Triangle-budget acceptance criterion.
        assert len(shape.faces()) <= 5000


def test_get_caches_instances() -> None:
    lib = ShapeLibrary.default()
    a = lib.get("head")
    b = lib.get("head")
    assert a is b


def test_rest_dimensions_match_manifest_within_tolerance() -> None:
    lib = ShapeLibrary.default()
    manifest_path = (
        Path(__file__).resolve().parents[3]
        / "assets"
        / "body_part_shapes"
        / "default"
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, entry in manifest["shapes"].items():
        expected = tuple(entry["rest_dimensions"])
        actual = lib.get(name).rest_dimensions
        for e, a in zip(expected, actual, strict=True):
            assert abs(float(e) - float(a)) < 1e-3, (
                f"{name}: rest_dimensions mismatch (manifest={expected}, "
                f"loaded={actual})"
            )


def test_unknown_name_raises_key_error_listing_available() -> None:
    lib = ShapeLibrary.default()
    with pytest.raises(KeyError) as excinfo:
        lib.get("not_a_real_shape")
    msg = str(excinfo.value)
    assert "not_a_real_shape" in msg
    for name in lib.names():
        assert name in msg


def test_unknown_binding_template_raises_key_error() -> None:
    lib = ShapeLibrary.default()
    with pytest.raises(KeyError):
        lib.binding_template("does_not_exist")


def test_binding_template_head_is_marker_binding() -> None:
    lib = ShapeLibrary.default()
    binding = lib.binding_template("head")
    assert isinstance(binding, MarkerBinding)
    assert binding.kind is BindingKind.BETWEEN_TWO
    assert len(binding.marker_names) == 2


def test_binding_template_every_name_round_trips() -> None:
    lib = ShapeLibrary.default()
    for name in lib.names():
        binding = lib.binding_template(name)
        assert isinstance(binding, MarkerBinding)


def test_binding_template_cluster_kind() -> None:
    lib = ShapeLibrary.default()
    binding = lib.binding_template("torso")
    assert binding.kind is BindingKind.CLUSTER
    assert len(binding.marker_names) >= 3


def test_binding_template_on_marker_kind() -> None:
    lib = ShapeLibrary.default()
    binding = lib.binding_template("hand")
    assert binding.kind is BindingKind.ON_MARKER
    assert len(binding.marker_names) == 1


def test_explicit_asset_root(tmp_path: Path) -> None:
    # Build a tiny manifest pointing at an existing default mesh.
    src_root = (
        Path(__file__).resolve().parents[3] / "assets" / "body_part_shapes" / "default"
    )
    (tmp_path / "head.stl").write_bytes((src_root / "head.stl").read_bytes())
    manifest = {
        "schema_version": 1,
        "shapes": {
            "head": {
                "file": "head.stl",
                "rest_dimensions": [0.1, 0.1, 0.1],
                "binding_template": {
                    "kind": "on_marker",
                    "marker_names": ["X"],
                    "rest_orientation_quat": [1.0, 0.0, 0.0, 0.0],
                },
                "license": "CC0-1.0",
                "source": "test",
            }
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    lib = ShapeLibrary(asset_root=tmp_path)
    assert lib.names() == ("head",)
    assert isinstance(lib.get("head"), MeshShape)


def test_missing_asset_root_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="asset_root"):
        ShapeLibrary(asset_root=tmp_path / "no_such_dir")


def test_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="manifest"):
        ShapeLibrary(asset_root=tmp_path)


def test_unsupported_schema_version_raises(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema_version": 999, "shapes": {}})
    )
    with pytest.raises(ValueError, match="schema_version"):
        ShapeLibrary(asset_root=tmp_path)


def test_empty_shapes_raises(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "shapes": {}})
    )
    with pytest.raises(ValueError, match="shapes"):
        ShapeLibrary(asset_root=tmp_path)


def test_shapes_wrong_type_raises(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "shapes": []})
    )
    with pytest.raises(ValueError, match="shapes"):
        ShapeLibrary(asset_root=tmp_path)


def test_entry_missing_field_raises(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "shapes": {"head": {"file": "head.stl"}},
            }
        )
    )
    with pytest.raises(ValueError, match="rest_dimensions"):
        ShapeLibrary(asset_root=tmp_path)


def test_entry_wrong_type_raises(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "shapes": {"head": "scalar"}})
    )
    with pytest.raises(ValueError, match="object"):
        ShapeLibrary(asset_root=tmp_path)


def _write_minimal_manifest(tmp_path: Path, template: dict[str, object]) -> None:
    src_root = (
        Path(__file__).resolve().parents[3] / "assets" / "body_part_shapes" / "default"
    )
    (tmp_path / "head.stl").write_bytes((src_root / "head.stl").read_bytes())
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "shapes": {
                    "head": {
                        "file": "head.stl",
                        "rest_dimensions": [0.1, 0.1, 0.1],
                        "binding_template": template,
                    }
                },
            }
        )
    )


def test_binding_template_bad_kind_raises(tmp_path: Path) -> None:
    _write_minimal_manifest(
        tmp_path,
        {"kind": "not_a_kind", "marker_names": ["A"]},
    )
    lib = ShapeLibrary(asset_root=tmp_path)
    with pytest.raises(ValueError, match="BindingKind"):
        lib.binding_template("head")


def test_binding_template_kind_not_string_raises(tmp_path: Path) -> None:
    _write_minimal_manifest(
        tmp_path,
        {"kind": 5, "marker_names": ["A"]},
    )
    lib = ShapeLibrary(asset_root=tmp_path)
    with pytest.raises(ValueError, match="kind"):
        lib.binding_template("head")


def test_binding_template_marker_names_wrong_type_raises(tmp_path: Path) -> None:
    _write_minimal_manifest(
        tmp_path,
        {"kind": "on_marker", "marker_names": "X"},
    )
    lib = ShapeLibrary(asset_root=tmp_path)
    with pytest.raises(ValueError, match="marker_names"):
        lib.binding_template("head")


def test_binding_template_quat_wrong_length_raises(tmp_path: Path) -> None:
    _write_minimal_manifest(
        tmp_path,
        {
            "kind": "on_marker",
            "marker_names": ["A"],
            "rest_orientation_quat": [1.0, 0.0, 0.0],
        },
    )
    lib = ShapeLibrary(asset_root=tmp_path)
    with pytest.raises(ValueError, match="rest_orientation_quat"):
        lib.binding_template("head")


def test_binding_template_default_quat_when_missing(tmp_path: Path) -> None:
    _write_minimal_manifest(
        tmp_path,
        {"kind": "on_marker", "marker_names": ["A"]},
    )
    lib = ShapeLibrary(asset_root=tmp_path)
    binding = lib.binding_template("head")
    assert binding.rest_orientation_quat == (1.0, 0.0, 0.0, 0.0)


def test_binding_template_object_required(tmp_path: Path) -> None:
    src_root = (
        Path(__file__).resolve().parents[3] / "assets" / "body_part_shapes" / "default"
    )
    (tmp_path / "head.stl").write_bytes((src_root / "head.stl").read_bytes())
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "shapes": {
                    "head": {
                        "file": "head.stl",
                        "rest_dimensions": [0.1, 0.1, 0.1],
                        "binding_template": "not-an-object",
                    }
                },
            }
        )
    )
    lib = ShapeLibrary(asset_root=tmp_path)
    with pytest.raises(ValueError, match="binding_template"):
        lib.binding_template("head")
