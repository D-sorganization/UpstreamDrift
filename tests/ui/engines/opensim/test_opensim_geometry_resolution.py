"""OpenSim explorer resolves body meshes and never shows a blank view (#12919)."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from src.engines.physics_engines.opensim.python import club_visuals, geometry_resolver

MODEL_TEMPLATE = (
    '<?xml version="1.0"?><OpenSimDocument Version="40500"><Model name="m">'
    '<Body name="b"><attached_geometry>{meshes}</attached_geometry></Body>'
    "</Model></OpenSimDocument>"
)


def _write_model(path: Path, names: list[str]) -> Path:
    meshes = "".join(
        f'<Mesh name="g{i}"><mesh_file>{n}</mesh_file></Mesh>'
        for i, n in enumerate(names)
    )
    path.write_text(MODEL_TEMPLATE.format(meshes=meshes), encoding="utf-8")
    return path


@pytest.mark.unit
def test_env_var_directory_is_a_candidate(tmp_path: Path) -> None:
    dirs = geometry_resolver.candidate_geometry_dirs(
        {geometry_resolver.GEOMETRY_ENV_VAR: str(tmp_path)}
    )
    assert tmp_path.resolve() in dirs


@pytest.mark.unit
def test_repo_geometry_dir_is_a_candidate() -> None:
    assert club_visuals.GEOMETRY_DIR.resolve() in (
        geometry_resolver.candidate_geometry_dirs({})
    )


@pytest.mark.unit
def test_installed_package_geometry_dir_is_a_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pkg = tmp_path / "opensim"
    (tmp_path / "Geometry").mkdir()
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    monkeypatch.setattr(
        geometry_resolver,
        "_package_geometry_dirs",
        lambda: [pkg / "Geometry", tmp_path / "Geometry"],
    )
    assert (tmp_path / "Geometry").resolve() in (
        geometry_resolver.candidate_geometry_dirs({})
    )


@pytest.mark.unit
def test_registration_goes_through_club_visuals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Path] = []
    monkeypatch.setattr(
        club_visuals,
        "register_geometry_path",
        lambda d: seen.append(Path(d)) or True,
    )
    registered = geometry_resolver.register_geometry_dirs([tmp_path])
    assert seen == [tmp_path]
    assert registered == (tmp_path,)


@pytest.mark.unit
def test_mesh_found_in_each_source_resolves(tmp_path: Path) -> None:
    src_a, src_b = tmp_path / "a", tmp_path / "b"
    src_a.mkdir()
    src_b.mkdir()
    (src_a / "pelvis.vtp").write_text("x")
    (src_b / "femur.vtp").write_text("x")
    model = _write_model(tmp_path / "m.osim", ["pelvis.vtp", "femur.vtp"])
    audit = geometry_resolver.audit_model_geometry(model, [src_a, src_b])
    assert audit.ok
    assert audit.banner is None


@pytest.mark.unit
def test_missing_mesh_sets_banner(tmp_path: Path) -> None:
    (tmp_path / "have.vtp").write_text("x")
    model = _write_model(tmp_path / "m.osim", ["have.vtp", "gone1.vtp", "gone2.vtp"])
    audit = geometry_resolver.audit_model_geometry(model, [])
    assert audit.missing == ("gone1.vtp", "gone2.vtp")
    assert audit.banner == "2 meshes missing; showing simplified geometry"


@pytest.mark.unit
def test_audit_rejects_missing_model(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        geometry_resolver.audit_model_geometry(tmp_path / "nope.osim", [])


@pytest.mark.unit
def test_default_model_is_the_anthro_golfer() -> None:
    assert geometry_resolver.DEFAULT_MODEL.name == "full_body_anthro_driver.osim"
    assert geometry_resolver.DEFAULT_MODEL.is_file()


@pytest.mark.unit
def test_gui_banner_and_default_model(qapp, monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("PyQt6")
    pytest.importorskip("opensim")
    gui_mod = importlib.import_module(
        "src.engines.physics_engines.opensim.python.opensim_gui"
    )
    window = gui_mod.OpenSimGolfGUI(model_path=None)
    try:
        widget = window._main_widget
        assert widget.model_path == str(geometry_resolver.DEFAULT_MODEL)
        assert widget.geometry_audit is not None and widget.geometry_audit.ok
        assert widget.fig.axes, "view must not be blank"
        # A model with unresolvable meshes raises the banner and still draws.
        widget.model_path = str(geometry_resolver.MODELS_DIR / "golf_humanoid.osim")
        widget._refresh_geometry_view()
        text = widget.lbl_geometry_banner.text()
        assert "meshes missing; showing simplified geometry" in text
        assert not widget.lbl_geometry_banner.isHidden()
        assert widget.fig.axes, "view must not be blank"
    finally:
        window.close()
