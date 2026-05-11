"""Unit tests for the biomech sibling-repo resolution layer.

Covers the four-tier resolution order (editable / installed / vendored /
env) for each of the five sibling providers registered in
``src/shared/python/config/model_source_providers.py``. See
``docs/adr/0014-shared-biomech-models.md`` (UpstreamDrift#5184).
"""

from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from src.shared.python.config import model_source_providers as msp

pytestmark = pytest.mark.unit


SIBLINGS = [
    ("mujoco_models", "MuJoCo_Models", "mujoco_models", "MUJOCO_MODELS_HOME"),
    ("drake_models", "Drake_Models", "drake_models", "DRAKE_MODELS_HOME"),
    (
        "pinocchio_models",
        "Pinocchio_Models",
        "pinocchio_models",
        "PINOCCHIO_MODELS_HOME",
    ),
    ("opensim_models", "OpenSim_Models", "opensim_models", "OPENSIM_MODELS_HOME"),
    (
        "movement_optimizer",
        "Movement-Optimizer",
        "movement_optimizer",
        "MOVEMENT_OPTIMIZER_HOME",
    ),
]


@pytest.fixture
def isolated_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """Provide an isolated workspace root with synthetic UpstreamDrift layout.

    Layout::

        tmp_path/
          UpstreamDrift/              (acts as the repo root)
            vendor/biomech-models/
          <sibling repos created by tests as needed>
    """
    workspace = tmp_path
    repo_root = workspace / "UpstreamDrift"
    repo_root.mkdir()
    (repo_root / "vendor" / "biomech-models").mkdir(parents=True)
    monkeypatch.setattr(msp, "_upstreamdrift_repo_root", lambda: repo_root)
    # Wipe every env var so previous tier-4 fallbacks don't pollute tests.
    for _, _, _, env_var in SIBLINGS:
        monkeypatch.delenv(env_var, raising=False)
    yield workspace


def _write_editable_sibling(
    workspace: Path,
    repo_name: str,
    *,
    models_subdir: str = "models",
    with_manifest: bool = True,
    manifest_models_root: str | None = None,
) -> Path:
    """Create a synthetic editable sibling checkout next to UpstreamDrift."""
    sibling = workspace / repo_name
    sibling.mkdir()
    (sibling / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    models_dir = sibling / models_subdir
    models_dir.mkdir(parents=True)
    if with_manifest:
        declared = manifest_models_root or models_subdir
        (sibling / "model_pack.yaml").write_text(
            f"schema: model_pack/v1\n"
            f"repo: {repo_name}\n"
            f"package: pkg\n"
            f"models_root: {declared}\n",
            encoding="utf-8",
        )
    return sibling


def _write_vendor_snapshot(
    workspace: Path,
    repo_name: str,
    *,
    with_manifest: bool = False,
) -> Path:
    """Create a synthetic vendored snapshot inside the repo's vendor/ tree."""
    vendor = workspace / "UpstreamDrift" / "vendor" / "biomech-models" / repo_name
    vendor.mkdir(parents=True)
    (vendor / "models").mkdir()
    if with_manifest:
        (vendor / "model_pack.yaml").write_text(
            f"schema: model_pack/v1\n"
            f"repo: {repo_name}\n"
            f"package: pkg\n"
            f"models_root: models\n",
            encoding="utf-8",
        )
    return vendor


@pytest.mark.parametrize("sibling_name,repo_name,pkg,env_var", SIBLINGS)
def test_tier_missing_when_nothing_resolves(
    sibling_name: str,
    repo_name: str,
    pkg: str,
    env_var: str,
    isolated_workspace: Path,
) -> None:
    """All four tiers miss → ``MISSING`` tier with no models_root."""
    # Make sure no installed package is found.
    with patch.object(msp.importlib.util, "find_spec", return_value=None):
        resolution = msp.resolve_sibling(sibling_name)
    assert resolution.tier == msp.SiblingResolutionTier.MISSING
    assert resolution.models_root is None
    assert resolution.repo_name == repo_name
    assert resolution.package == pkg
    assert resolution.env_var == env_var


@pytest.mark.parametrize("sibling_name,repo_name,pkg,env_var", SIBLINGS)
def test_editable_tier_wins_over_all_others(
    sibling_name: str,
    repo_name: str,
    pkg: str,
    env_var: str,
    isolated_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tier 1 fires when ``../<RepoName>/pyproject.toml`` exists."""
    sibling = _write_editable_sibling(isolated_workspace, repo_name)
    _write_vendor_snapshot(isolated_workspace, repo_name, with_manifest=True)
    monkeypatch.setenv(env_var, str(isolated_workspace / "alt"))
    with patch.object(
        msp.importlib.util,
        "find_spec",
        return_value=object(),
    ):
        resolution = msp.resolve_sibling(sibling_name)
    assert resolution.tier == msp.SiblingResolutionTier.EDITABLE
    assert resolution.models_root == (sibling / "models").resolve()
    assert resolution.manifest_path is not None


@pytest.mark.parametrize("sibling_name,repo_name,pkg,env_var", SIBLINGS)
def test_editable_falls_back_to_conventional_models_dir(
    sibling_name: str,
    repo_name: str,
    pkg: str,
    env_var: str,
    isolated_workspace: Path,
) -> None:
    """Tier 1 without a manifest still resolves via the conventional ``models/`` dir."""
    sibling = _write_editable_sibling(
        isolated_workspace, repo_name, with_manifest=False
    )
    with patch.object(msp.importlib.util, "find_spec", return_value=None):
        resolution = msp.resolve_sibling(sibling_name)
    assert resolution.tier == msp.SiblingResolutionTier.EDITABLE
    assert resolution.models_root == (sibling / "models").resolve()
    assert resolution.manifest_path is None


@pytest.mark.parametrize("sibling_name,repo_name,pkg,env_var", SIBLINGS)
def test_installed_tier_wins_when_no_editable(
    sibling_name: str,
    repo_name: str,
    pkg: str,
    env_var: str,
    isolated_workspace: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tier 2 fires when the sibling package has a ``model_pack`` submodule."""
    _write_vendor_snapshot(isolated_workspace, repo_name, with_manifest=True)
    monkeypatch.setenv(env_var, str(tmp_path / "env"))
    fake_root = tmp_path / "installed_models"
    fake_root.mkdir()

    module = types.ModuleType(f"{pkg}.model_pack")
    module.resolve = lambda: fake_root  # type: ignore[attr-defined]
    sys.modules[f"{pkg}.model_pack"] = module

    # find_spec must return non-None for both the package and its model_pack
    # submodule. We do not register the parent package itself in sys.modules
    # because importlib.util.find_spec only needs a truthy spec.
    def _fake_find_spec(name: str) -> object | None:
        if name == pkg:
            return object()
        if name == f"{pkg}.model_pack":
            return object()
        return None

    try:
        with (
            patch.object(msp.importlib.util, "find_spec", side_effect=_fake_find_spec),
            patch.object(msp.importlib, "import_module", return_value=module),
        ):
            resolution = msp.resolve_sibling(sibling_name)
    finally:
        sys.modules.pop(f"{pkg}.model_pack", None)

    assert resolution.tier == msp.SiblingResolutionTier.INSTALLED
    assert resolution.models_root == fake_root.resolve()


@pytest.mark.parametrize("sibling_name,repo_name,pkg,env_var", SIBLINGS)
def test_vendored_tier_wins_when_no_editable_or_installed(
    sibling_name: str,
    repo_name: str,
    pkg: str,
    env_var: str,
    isolated_workspace: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tier 3 fires when only the vendor snapshot exists."""
    vendor = _write_vendor_snapshot(isolated_workspace, repo_name, with_manifest=True)
    monkeypatch.setenv(env_var, str(tmp_path / "ignored"))
    with patch.object(msp.importlib.util, "find_spec", return_value=None):
        resolution = msp.resolve_sibling(sibling_name)
    assert resolution.tier == msp.SiblingResolutionTier.VENDORED
    assert resolution.models_root == (vendor / "models").resolve()
    assert resolution.manifest_path is not None


@pytest.mark.parametrize("sibling_name,repo_name,pkg,env_var", SIBLINGS)
def test_env_tier_wins_when_others_miss(
    sibling_name: str,
    repo_name: str,
    pkg: str,
    env_var: str,
    isolated_workspace: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tier 4 fires when the env var points at an existing directory."""
    env_dir = tmp_path / "env_root"
    env_dir.mkdir()
    monkeypatch.setenv(env_var, str(env_dir))
    with patch.object(msp.importlib.util, "find_spec", return_value=None):
        resolution = msp.resolve_sibling(sibling_name)
    assert resolution.tier == msp.SiblingResolutionTier.ENV
    assert resolution.models_root == env_dir.resolve()


def test_resolve_all_siblings_returns_every_registered_sibling(
    isolated_workspace: Path,
) -> None:
    """Each registered sibling appears in :func:`resolve_all_siblings`."""
    with patch.object(msp.importlib.util, "find_spec", return_value=None):
        resolutions = msp.resolve_all_siblings()
    assert set(resolutions) == {name for name, *_ in SIBLINGS}


def test_unknown_sibling_raises_key_error(isolated_workspace: Path) -> None:
    """``resolve_sibling`` rejects names that aren't registered."""
    with pytest.raises(KeyError):
        msp.resolve_sibling("nonexistent_sibling")


def test_register_source_decorator_round_trip() -> None:
    """`@register_source` registers a callable that can be retrieved by name."""

    @msp.register_source("__pytest_dummy__")
    def _dummy() -> Path:
        return Path("/tmp")

    try:
        assert msp.get_registered_source("__pytest_dummy__") is _dummy
        assert "__pytest_dummy__" in msp.iter_registered_sources()
    finally:
        msp._MODEL_SOURCES.pop("__pytest_dummy__", None)


@pytest.mark.parametrize(
    "decorated_callable,expected_sibling",
    [
        (msp.mujoco_models_source, "mujoco_models"),
        (msp.drake_models_source, "drake_models"),
        (msp.pinocchio_models_source, "pinocchio_models"),
        (msp.opensim_models_source, "opensim_models"),
        (msp.movement_optimizer_source, "movement_optimizer"),
    ],
)
def test_decorated_provider_returns_resolved_path(
    decorated_callable: object,
    expected_sibling: str,
    isolated_workspace: Path,
) -> None:
    """Each decorated provider returns the resolved models root as an absolute Path."""
    # The synthetic editable checkout uses the human repo name, so we
    # look it up from the SIBLINGS table.
    repo_name = next(repo for name, repo, *_ in SIBLINGS if name == expected_sibling)
    sibling = _write_editable_sibling(isolated_workspace, repo_name)
    with patch.object(msp.importlib.util, "find_spec", return_value=None):
        result = decorated_callable()  # type: ignore[operator]
    assert isinstance(result, Path)
    assert result == (sibling / "models").resolve()


def test_missing_sibling_provider_raises_file_not_found(
    isolated_workspace: Path,
) -> None:
    """A decorated provider raises ``FileNotFoundError`` when every tier misses."""
    with (
        patch.object(msp.importlib.util, "find_spec", return_value=None),
        pytest.raises(FileNotFoundError, match="MuJoCo_Models"),
    ):
        msp.mujoco_models_source()


def test_resolve_sibling_handles_manifest_without_models_root(
    isolated_workspace: Path,
) -> None:
    """If the manifest omits ``models_root``, fall back to conventional paths."""
    sibling = isolated_workspace / "MuJoCo_Models"
    sibling.mkdir()
    (sibling / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (sibling / "model_pack.yaml").write_text(
        "schema: model_pack/v1\nrepo: MuJoCo_Models\npackage: mujoco_models\n",
        encoding="utf-8",
    )
    (sibling / "models").mkdir()
    with patch.object(msp.importlib.util, "find_spec", return_value=None):
        resolution = msp.resolve_sibling("mujoco_models")
    assert resolution.tier == msp.SiblingResolutionTier.EDITABLE
    assert resolution.models_root == (sibling / "models").resolve()


def test_env_var_pointing_at_nonexistent_path_falls_through(
    isolated_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An env var pointing at a non-directory yields ``MISSING``."""
    monkeypatch.setenv("MUJOCO_MODELS_HOME", str(tmp_path / "does-not-exist"))
    with patch.object(msp.importlib.util, "find_spec", return_value=None):
        resolution = msp.resolve_sibling("mujoco_models")
    assert resolution.tier == msp.SiblingResolutionTier.MISSING
    assert resolution.models_root is None


def test_module_can_be_reimported(isolated_workspace: Path) -> None:
    """Sanity check: re-importing the module does not blow up the registry."""
    importlib.reload(msp)
    assert "mujoco_models" in msp.iter_registered_sources()
