"""Tests for sibling model-repository discovery (model explorer)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.tools.model_explorer.sibling_repositories import (
    DEFAULT_SIBLING_REPO_NAMES,
    SIBLING_REPOS_ENV_VAR,
    candidate_sibling_roots,
    discover_sibling_models,
)

pytestmark = [pytest.mark.unit]

_URDF = '<robot name="r"><link name="base"/></robot>'
_MJCF = '<mujoco model="m"><worldbody/></mujoco>'
_OSIM = '<OpenSimDocument Version="40500"><Model name="m"/></OpenSimDocument>'
_SDF = '<sdf version="1.6"><model name="s"><link name="base"/></model></sdf>'


def _make_repo(parent: Path, name: str) -> Path:
    repo = parent / name
    repo.mkdir(parents=True)
    (repo / "arm.urdf").write_text(_URDF, encoding="utf-8")
    (repo / "plant.sdf").write_text(_SDF, encoding="utf-8")
    nested = repo / "models" / "hand"
    nested.mkdir(parents=True)
    (nested / "hand.xml").write_text(_MJCF, encoding="utf-8")
    (nested / "arm.osim").write_text(_OSIM, encoding="utf-8")
    # Junk that must be ignored:
    (repo / "notes.txt").write_text("not a model", encoding="utf-8")
    (repo / "config.xml").write_text("<settings/>", encoding="utf-8")
    git_dir = repo / ".git"
    git_dir.mkdir()
    (git_dir / "blob.urdf").write_text(_URDF, encoding="utf-8")
    return repo


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    root = tmp_path / "UpstreamDrift"
    root.mkdir()
    return root


class TestCandidateRoots:
    def test_defaults_resolve_next_to_project_root(
        self, project_root: Path, monkeypatch
    ) -> None:
        monkeypatch.delenv(SIBLING_REPOS_ENV_VAR, raising=False)
        existing = _make_repo(project_root.parent, DEFAULT_SIBLING_REPO_NAMES[0])
        roots = candidate_sibling_roots(project_root)
        assert roots == [existing]

    def test_env_var_overrides_defaults(
        self, project_root: Path, tmp_path: Path, monkeypatch
    ) -> None:
        custom = _make_repo(tmp_path, "My_Custom_Models")
        monkeypatch.setenv(SIBLING_REPOS_ENV_VAR, str(custom))
        roots = candidate_sibling_roots(project_root)
        assert roots == [custom]

    def test_env_var_multiple_roots(
        self, project_root: Path, tmp_path: Path, monkeypatch
    ) -> None:
        a = _make_repo(tmp_path, "A_Models")
        b = _make_repo(tmp_path, "B_Models")
        missing = tmp_path / "missing"
        monkeypatch.setenv(
            SIBLING_REPOS_ENV_VAR, os.pathsep.join([str(a), str(missing), str(b)])
        )
        assert candidate_sibling_roots(project_root) == [a, b]

    def test_invalid_project_root_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not a directory"):
            candidate_sibling_roots(tmp_path / "nope")


class TestDiscovery:
    def test_discovers_urdf_mjcf_sdf_and_osim_only(
        self, project_root: Path, tmp_path: Path, monkeypatch
    ) -> None:
        repo = _make_repo(tmp_path, "MuJoCo_Models")
        monkeypatch.setenv(SIBLING_REPOS_ENV_VAR, str(repo))
        models = discover_sibling_models(project_root)
        names = [m["name"] for m in models]
        assert names == ["arm.osim", "arm.urdf", "hand.xml", "plant.sdf"]
        types = {m["name"]: m["type"] for m in models}
        assert types == {
            "arm.osim": "osim",
            "arm.urdf": "urdf",
            "hand.xml": "mjcf",
            "plant.sdf": "sdf",
        }

    def test_git_dir_and_plain_xml_are_skipped(
        self, project_root: Path, tmp_path: Path, monkeypatch
    ) -> None:
        repo = _make_repo(tmp_path, "Drake_Models")
        monkeypatch.setenv(SIBLING_REPOS_ENV_VAR, str(repo))
        models = discover_sibling_models(project_root)
        paths = [m["path"] for m in models]
        assert not any(".git" in p for p in paths)
        assert not any(p.endswith("config.xml") for p in paths)

    def test_config_keys_are_stable_and_repo_tagged(
        self, project_root: Path, tmp_path: Path, monkeypatch
    ) -> None:
        repo = _make_repo(tmp_path, "Pinocchio_Models")
        monkeypatch.setenv(SIBLING_REPOS_ENV_VAR, str(repo))
        first = discover_sibling_models(project_root)
        second = discover_sibling_models(project_root)
        assert [m["config_key"] for m in first] == [m["config_key"] for m in second]
        assert all(m["repo"] == "Pinocchio_Models" for m in first)
        assert "sibling_Pinocchio_Models_arm.urdf" in {m["config_key"] for m in first}

    def test_no_siblings_yields_empty_list(
        self, project_root: Path, monkeypatch
    ) -> None:
        monkeypatch.setenv(SIBLING_REPOS_ENV_VAR, str(project_root / "absent"))
        assert discover_sibling_models(project_root) == []


class TestModelLibraryIntegration:
    def test_download_human_model_uses_shared_bounded_downloader(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from src.tools.model_explorer.model_library import ModelLibrary

        calls: list[tuple[str, Path, float]] = []

        def fake_download_to_file(url: str, dest: Path, timeout: float) -> Path:
            calls.append((url, dest, timeout))
            dest.write_text(_URDF, encoding="utf-8")
            return dest

        def fail_urlopen(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("model_library must not call urllib.urlopen directly")

        monkeypatch.setattr(
            "src.tools.model_explorer.model_library.download_to_file",
            fake_download_to_file,
        )
        monkeypatch.setattr("urllib.request.urlopen", fail_urlopen)
        library = ModelLibrary(base_path=tmp_path / "models")
        result = library.download_human_model("human_with_meshes", force=True)

        assert (
            result
            == tmp_path / "models" / "human_models" / "human_with_meshes" / "model.urdf"
        )
        assert calls == [
            (
                ModelLibrary.HUMAN_MODELS["human_with_meshes"]["urdf_url"],
                result,
                30,
            )
        ]

    def test_list_available_models_has_sibling_category(
        self, project_root: Path, tmp_path: Path, monkeypatch
    ) -> None:
        from src.tools.model_explorer.model_library import ModelLibrary

        repo = _make_repo(tmp_path, "OpenSim_Models")
        monkeypatch.setenv(SIBLING_REPOS_ENV_VAR, str(repo))
        library = ModelLibrary()
        listing = library.list_available_models()
        assert "sibling" in listing
        names = [m["name"] for m in listing["sibling"]]
        assert "arm.urdf" in names
        assert "arm.osim" in names

    def test_get_model_info_resolves_sibling_key(
        self, project_root: Path, tmp_path: Path, monkeypatch
    ) -> None:
        from src.tools.model_explorer.model_library import ModelLibrary

        repo = _make_repo(tmp_path, "OpenSim_Models")
        monkeypatch.setenv(SIBLING_REPOS_ENV_VAR, str(repo))
        library = ModelLibrary()
        info = library.get_model_info("sibling", "sibling_OpenSim_Models_arm.urdf")
        assert info is not None
        assert info["type"] == "urdf"
        assert info["path"].endswith("arm.urdf")

        osim_info = library.get_model_info(
            "sibling", "sibling_OpenSim_Models_models/hand/arm.osim"
        )
        assert osim_info is not None
        assert osim_info["type"] == "osim"
