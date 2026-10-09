"""Bounded disk-MJCF resource discovery and manifest admission tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.engines.physics_engines.myosuite.python.native_direct_model_replay import (
    DeclaredModelResource,
    _discover_native_resource_files,
    _verified_discovered_resource_hashes,
    resource_closure_sha256,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _require_supported_native_mujoco() -> None:
    mujoco = pytest.importorskip("mujoco")
    if mujoco.__version__ not in {"3.6.0", "3.8.0"}:
        pytest.skip(
            "bounded resource admission requires reviewed MuJoCo 3.6.0 or 3.8.0"
        )


def _declared_files(root: Path, *paths: str) -> tuple[DeclaredModelResource, ...]:
    return tuple(
        DeclaredModelResource(
            relative_path,
            hashlib.sha256((root / relative_path).read_bytes()).hexdigest(),
        )
        for relative_path in paths
    )


def test_resource_discovery_requires_included_xml_in_declared_inventory(
    tmp_path: Path,
) -> None:
    (tmp_path / "nested").mkdir()
    entry = tmp_path / "main.xml"
    entry.write_text('<mujoco><include file="nested/branch.xml"/></mujoco>')
    branch = tmp_path / "nested/branch.xml"
    branch.write_text('<mujocoinclude><include file="leaf.xml"/></mujocoinclude>')
    leaf = tmp_path / "nested/leaf.xml"
    leaf.write_text(
        '<mujocoinclude><custom><numeric name="leaf" data="1"/></custom></mujocoinclude>'
    )
    with pytest.raises(
        ValueError, match="discovered native source/resource is undeclared"
    ):
        resource_closure_sha256(tmp_path, entry, _declared_files(tmp_path, "main.xml"))


def test_resource_discovery_uses_native_model_root_include_priority(
    tmp_path: Path,
) -> None:
    (tmp_path / "nested").mkdir()
    entry = tmp_path / "main.xml"
    entry.write_text('<mujoco><include file="nested/branch.xml"/></mujoco>')
    branch = tmp_path / "nested/branch.xml"
    branch.write_text('<mujocoinclude><include file="leaf.xml"/></mujocoinclude>')
    (tmp_path / "nested/leaf.xml").write_text(
        '<mujocoinclude><custom><numeric name="nested_leaf" data="1"/></custom></mujocoinclude>'
    )
    (tmp_path / "leaf.xml").write_text(
        '<mujocoinclude><custom><numeric name="root_leaf" data="1"/></custom></mujocoinclude>'
    )
    resources = _declared_files(tmp_path, "main.xml", "nested/branch.xml", "leaf.xml")
    resource_closure_sha256(tmp_path, entry, resources)


def test_resource_discovery_uses_including_directory_when_root_include_missing(
    tmp_path: Path,
) -> None:
    (tmp_path / "nested").mkdir()
    entry = tmp_path / "main.xml"
    entry.write_text('<mujoco><include file="nested/branch.xml"/></mujoco>')
    branch = tmp_path / "nested/branch.xml"
    branch.write_text('<mujocoinclude><include file="leaf.xml"/></mujocoinclude>')
    leaf = tmp_path / "nested/leaf.xml"
    leaf.write_text(
        '<mujocoinclude><custom><numeric name="leaf" data="1"/></custom></mujocoinclude>'
    )
    resources = _declared_files(
        tmp_path, "main.xml", "nested/branch.xml", "nested/leaf.xml"
    )
    resource_closure_sha256(tmp_path, entry, resources)


def test_resource_discovery_rejects_outside_root_include(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.xml"
    outside.write_text(
        '<mujocoinclude><custom><numeric name="outside" data="1"/></custom></mujocoinclude>'
    )
    entry = tmp_path / "main.xml"
    entry.write_text('<mujoco><include file="../' + outside.name + '"/></mujoco>')
    try:
        with pytest.raises(ValueError, match="outside its root"):
            resource_closure_sha256(
                tmp_path, entry, _declared_files(tmp_path, "main.xml")
            )
    finally:
        outside.unlink()


def test_resource_discovery_rejects_include_uri_before_native_parse(
    tmp_path: Path,
) -> None:
    entry = tmp_path / "main.xml"
    entry.write_text(
        '<mujoco><include file="https://invalid.example/leaf.xml"/></mujoco>'
    )
    with pytest.raises(
        ValueError, match="include file must use a contained relative disk path"
    ):
        resource_closure_sha256(tmp_path, entry, _declared_files(tmp_path, "main.xml"))


def test_resource_discovery_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.xml"
    outside.write_text("<mujocoinclude><custom/></mujocoinclude>")
    entry = tmp_path / "main.xml"
    entry.write_text('<mujoco><include file="alias.xml"/></mujoco>')
    alias = tmp_path / "alias.xml"
    try:
        alias.symlink_to(outside)
    except OSError as exc:
        outside.unlink()
        pytest.skip(f"file symlinks are unavailable: {exc}")
    try:
        with pytest.raises(ValueError, match="outside its root"):
            resource_closure_sha256(
                tmp_path, entry, _declared_files(tmp_path, "main.xml")
            )
    finally:
        alias.unlink()
        outside.unlink()


def test_resource_discovery_rejects_unreviewed_file_loaders_before_native_parse(
    tmp_path: Path,
) -> None:
    entry = tmp_path / "main.xml"
    entry.write_text(
        '<mujoco><asset><hfield name="height" file="../height.png" '
        'size="1 1 1 1"/></asset></mujoco>'
    )
    with pytest.raises(ValueError, match="unsupported MJCF source loader: hfield"):
        resource_closure_sha256(tmp_path, entry, _declared_files(tmp_path, "main.xml"))


def test_resource_discovery_requires_meshdir_file_in_declared_inventory(
    tmp_path: Path,
) -> None:
    (tmp_path / "meshes").mkdir()
    (tmp_path / "meshes/tetra.stl").write_text("solid tetra\nendsolid tetra\n")
    entry = tmp_path / "main.xml"
    entry.write_text(
        '<mujoco><compiler meshdir="meshes"/><asset><mesh name="tetra" '
        'file="tetra.stl"/></asset></mujoco>'
    )
    with pytest.raises(
        ValueError, match="discovered native source/resource is undeclared"
    ):
        resource_closure_sha256(tmp_path, entry, _declared_files(tmp_path, "main.xml"))


@pytest.mark.parametrize(
    ("kind", "directory", "filename", "asset_xml"),
    (
        (
            "mesh",
            "meshes",
            "tetra.stl",
            '<mesh name="tetra" file="tetra.stl"/>',
        ),
        (
            "texture",
            "textures",
            "surface.png",
            '<texture name="surface" type="2d" file="surface.png"/>',
        ),
    ),
)
def test_resource_discovery_rejects_changed_native_asset_bytes(
    tmp_path: Path, kind: str, directory: str, filename: str, asset_xml: str
) -> None:
    folder = tmp_path / directory
    folder.mkdir()
    asset_path = folder / filename
    asset_path.write_bytes(b"pinned native asset")
    entry = tmp_path / "main.xml"
    compiler_attribute = "meshdir" if kind == "mesh" else "texturedir"
    entry.write_text(
        f'<mujoco><compiler {compiler_attribute}="{directory}"/>'
        f"<asset>{asset_xml}</asset></mujoco>"
    )
    resources = _declared_files(tmp_path, "main.xml", f"{directory}/{filename}")
    resource_closure_sha256(tmp_path, entry, resources)
    asset_path.write_bytes(b"altered native asset")
    with pytest.raises(ValueError, match="model resource digest differs"):
        resource_closure_sha256(tmp_path, entry, resources)


def test_resource_discovery_rejects_unreviewed_mesh_format_before_native_parse(
    tmp_path: Path,
) -> None:
    (tmp_path / "meshes").mkdir()
    (tmp_path / "meshes/model.obj").write_text("v 0 0 0\n")
    entry = tmp_path / "main.xml"
    entry.write_text(
        '<mujoco><compiler meshdir="meshes"/><asset><mesh name="m" '
        'file="model.obj"/></asset></mujoco>'
    )
    resources = _declared_files(tmp_path, "main.xml", "meshes/model.obj")
    with pytest.raises(ValueError, match="unsupported mesh resource format"):
        resource_closure_sha256(tmp_path, entry, resources)


def test_resource_discovery_requires_texture_cube_faces_in_manifest(
    tmp_path: Path,
) -> None:
    (tmp_path / "textures").mkdir()
    names = (
        "back.png",
        "down.png",
        "front.png",
        "left.png",
        "right.png",
        "up.png",
    )
    for name in names:
        (tmp_path / "textures" / name).write_bytes(b"native texture fixture")
    entry = tmp_path / "main.xml"
    entry.write_text(
        '<mujoco><compiler texturedir="textures"/><asset><texture '
        'name="sky" type="skybox" '
        + " ".join(
            f'file{face}="{name}"'
            for face, name in zip(
                ("back", "down", "front", "left", "right", "up"),
                names,
                strict=True,
            )
        )
        + "/></asset></mujoco>"
    )
    discovered, _spec = _discover_native_resource_files(entry, tmp_path)
    declared = _declared_files(
        tmp_path,
        "main.xml",
        *(f"textures/{name}" for name in names[:-1]),
    )
    declared_hashes = {item.relative_path: item.sha256 for item in declared}
    with pytest.raises(ValueError, match="undeclared: textures/up.png"):
        _verified_discovered_resource_hashes(discovered, tmp_path, declared_hashes)
    assert tmp_path / "textures" / names[-1] in discovered


@pytest.mark.parametrize(
    ("xml", "message"),
    (
        ("<robot/>", "supported MuJoCo root element"),
        ('<mujoco><compiler strippath="true"/></mujoco>', "strippath is unsupported"),
        ("<mujoco><extension/></mujoco>", "unsupported MJCF source loader: extension"),
        (
            '<mujoco><attach model="other"/></mujoco>',
            "unsupported MJCF source loader: attach",
        ),
        (
            '<mujoco><asset><mesh name="m" file="m.stl" content_type="model/stl"/>'
            "</asset></mujoco>",
            "content_type overrides are unsupported",
        ),
        (
            '<mujoco><asset><texture name="sky" type="skybox" '
            'fileup="https://invalid.example/sky.png"/></asset></mujoco>',
            "texture fileup must use a contained relative disk path",
        ),
        (
            '<mujoco><custom file="extra.bin"/></mujoco>',
            "unsupported file-bearing MJCF element",
        ),
    ),
)
def test_resource_discovery_refuses_unreviewed_source_loader_policy(
    tmp_path: Path, xml: str, message: str
) -> None:
    entry = tmp_path / "main.xml"
    entry.write_text(xml)
    with pytest.raises(ValueError, match=message):
        resource_closure_sha256(tmp_path, entry, _declared_files(tmp_path, "main.xml"))


def test_resource_discovery_rejects_compiler_directories_outside_root(
    tmp_path: Path,
) -> None:
    entry = tmp_path / "main.xml"
    entry.write_text(
        '<mujoco><compiler meshdir="../outside"/><asset><mesh name="m" '
        'file="mesh.stl"/></asset></mujoco>'
    )
    with pytest.raises(ValueError, match="model source/resource is outside its root"):
        resource_closure_sha256(tmp_path, entry, _declared_files(tmp_path, "main.xml"))


def test_public_resource_closure_rejects_callbacks_before_native_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mujoco = pytest.importorskip("mujoco")

    import src.engines.physics_engines.myosuite.python.native_direct_model_replay as direct_replay

    entry = tmp_path / "main.xml"
    entry.write_text("<mujoco/>")

    def forbidden_discovery(*_args, **_kwargs):
        raise AssertionError("native discovery must not run with callbacks")

    monkeypatch.setattr(
        direct_replay, "_discover_native_resource_files", forbidden_discovery
    )
    previous_callback = mujoco.get_mjcb_control()
    mujoco.set_mjcb_control(lambda _model, _data: None)
    try:
        with pytest.raises(
            ValueError, match="process-global MuJoCo callbacks are forbidden"
        ):
            resource_closure_sha256(
                tmp_path, entry, _declared_files(tmp_path, "main.xml")
            )
    finally:
        mujoco.set_mjcb_control(previous_callback)
