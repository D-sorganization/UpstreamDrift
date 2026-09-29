"""Tests for src.shared.python.config.standard_models (Issues #1949, #1744)."""

from __future__ import annotations

from pathlib import Path

import pytest
from src.shared.python.config.standard_models import StandardModelManager

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestStandardModelManagerInit:
    def test_construct_with_tmp_root(self, tmp_path: Path) -> None:
        mgr = StandardModelManager(suite_root=tmp_path)
        assert mgr.suite_root == tmp_path

    def test_construct_default_suite_root(self) -> None:
        mgr = StandardModelManager()
        assert isinstance(mgr.suite_root, Path)

    def test_config_is_dict(self, tmp_path: Path) -> None:
        mgr = StandardModelManager(suite_root=tmp_path)
        assert isinstance(mgr.config, dict)

    def test_config_has_standard_humanoid(self, tmp_path: Path) -> None:
        mgr = StandardModelManager(suite_root=tmp_path)
        assert "standard_humanoid" in mgr.config

    def test_config_has_golf_clubs(self, tmp_path: Path) -> None:
        mgr = StandardModelManager(suite_root=tmp_path)
        assert "golf_clubs" in mgr.config


# ---------------------------------------------------------------------------
# list_available_models
# ---------------------------------------------------------------------------


class TestListAvailableModels:
    def test_standard_models_returns_dict(self, tmp_path: Path) -> None:
        mgr = StandardModelManager(suite_root=tmp_path)
        result = mgr.list_available_models()
        assert isinstance(result, dict)

    def test_humanoid_in_list(self, tmp_path: Path) -> None:
        mgr = StandardModelManager(suite_root=tmp_path)
        result = mgr.list_available_models()
        assert "humanoid" in result

    def test_golf_clubs_in_list(self, tmp_path: Path) -> None:
        mgr = StandardModelManager(suite_root=tmp_path)
        result = mgr.list_available_models()
        assert "golf_clubs" in result


# ---------------------------------------------------------------------------
# get_golf_club_path
# ---------------------------------------------------------------------------


class TestGetGolfClubPath:
    def test_driver_returns_path(self, tmp_path: Path) -> None:
        mgr = StandardModelManager(suite_root=tmp_path)
        path = mgr.get_golf_club_path("driver")
        assert isinstance(path, Path)

    def test_driver_path_ends_with_urdf(self, tmp_path: Path) -> None:
        mgr = StandardModelManager(suite_root=tmp_path)
        path = mgr.get_golf_club_path("driver")
        assert path.suffix == ".urdf"

    def test_iron7_returns_path(self, tmp_path: Path) -> None:
        mgr = StandardModelManager(suite_root=tmp_path)
        path = mgr.get_golf_club_path("iron_7")
        assert isinstance(path, Path)


# ---------------------------------------------------------------------------
# get_standard_humanoid_path
# ---------------------------------------------------------------------------


class TestGetStandardHumanoidPath:
    def test_raises_when_model_absent(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        mgr = StandardModelManager(suite_root=tmp_path)
        # Prevent download attempt, just test the error path
        with patch.object(mgr, "download_standard_humanoid", return_value=False):
            from src.shared.python.data_io.common_utils import GolfModelingError

            with pytest.raises(GolfModelingError):
                mgr.get_standard_humanoid_path()


# ---------------------------------------------------------------------------
# download_standard_humanoid — no silent empty-stub success (issue #7186)
# ---------------------------------------------------------------------------


class TestDownloadStandardHumanoidMeshes:
    """The humanoid download must produce real geometry or fail loudly."""

    _URDF_WITH_MESH = (
        '<?xml version="1.0"?>\n'
        '<robot name="h">\n'
        '  <link name="torso">\n'
        "    <visual><geometry>\n"
        '      <mesh filename="meshes/torso.stl"/>\n'
        "    </geometry></visual>\n"
        "  </link>\n"
        "</robot>\n"
    )

    def _fake_download(self, real_mesh: bool):
        """Return a download_to_file replacement writing URDF/yaml and mesh."""

        def _impl(url: str, dest, timeout: float = 30):
            from pathlib import Path as _P

            dest = _P(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if url.endswith(".urdf"):
                dest.write_text(self._URDF_WITH_MESH, encoding="utf-8")
            elif url.endswith(".yaml"):
                dest.write_text("name: human\n", encoding="utf-8")
            elif url.endswith(".stl"):
                if real_mesh:
                    # Non-empty STL with a real facet.
                    dest.write_text(
                        "solid s\nfacet normal 0 0 1\nendfacet\nendsolid s\n",
                        encoding="utf-8",
                    )
                else:
                    raise OSError("mesh fetch failed")
            else:
                dest.write_text("x", encoding="utf-8")
            return dest

        return _impl

    def test_success_produces_nonempty_meshes(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        mgr = StandardModelManager(suite_root=tmp_path)
        with patch(
            "src.shared.python.config.standard_models.download_to_file",
            side_effect=self._fake_download(real_mesh=True),
        ):
            assert mgr.download_standard_humanoid() is True

        stl_files = list((mgr.meshes_dir / "human").glob("*.stl"))
        assert stl_files, "expected at least one mesh file"
        for stl in stl_files:
            assert stl.stat().st_size > 0
            assert "facet" in stl.read_text()

    def test_mesh_failure_returns_false_no_stubs(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        mgr = StandardModelManager(suite_root=tmp_path)
        with patch(
            "src.shared.python.config.standard_models.download_to_file",
            side_effect=self._fake_download(real_mesh=False),
        ):
            assert mgr.download_standard_humanoid() is False

        # No empty placeholder stub meshes must have been written.
        human_dir = mgr.meshes_dir / "human"
        if human_dir.exists():
            for stl in human_dir.glob("*.stl"):
                body = stl.read_text()
                assert "temporary" not in body

    def test_allow_stub_meshes_opt_in_still_succeeds(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        mgr = StandardModelManager(suite_root=tmp_path)
        with patch(
            "src.shared.python.config.standard_models.download_to_file",
            side_effect=self._fake_download(real_mesh=False),
        ):
            # Explicit dev opt-in writes stubs and reports success.
            assert mgr.download_standard_humanoid(allow_stub_meshes=True) is True
