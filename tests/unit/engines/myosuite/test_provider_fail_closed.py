"""Tests for MyoSuite provider fail-closed behavior and tile status honesty (MS-50, #10343)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from src.config.launcher_manifest_loader import LauncherManifest
from src.engines.physics_engines.myosuite.python.motion_matching.provider import (
    MyoSuiteFitSwingProvider,
)
from src.shared.python.motion_matching.club_target import ClubTarget, SourceProvenance
from src.shared.python.motion_matching.provider import FitOptions, get_provider

REPO_ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def dummy_club_target() -> ClubTarget:
    time = np.linspace(0, 0.3, 301)
    butt = np.zeros((301, 3))
    clubhead = np.zeros((301, 3))
    club_quat = np.zeros((301, 4))
    club_quat[:, 0] = 1.0  # w=1
    return ClubTarget(
        time=time,
        butt=butt,
        clubhead=clubhead,
        club_quat=club_quat,
        impact_idx=150,
        source=SourceProvenance(
            filename="dummy.c3d",
            format="c3d",
            subject_id="test",
            trial_id="dummy",
            sha256="dummyhash",
        ),
    )


@pytest.mark.unit
def test_no_torch_imported_at_provider_module_level() -> None:
    """Importing provider must not import torch (LoD / import time constraint)."""
    import src.engines.physics_engines.myosuite.python.motion_matching.provider as prov_mod

    assert "torch" not in prov_mod.__dict__


@pytest.mark.unit
def test_fit_swing_never_returns_activations_without_model(
    dummy_club_target: ClubTarget,
) -> None:
    """MyoSuite fit_swing fails closed with status=unsupported and no fake activations."""
    provider = MyoSuiteFitSwingProvider()

    # supports_body_target must be False until MS-53
    assert provider.supports_body_target() is False
    assert provider.supports_ball_target() is False

    result = provider.fit_swing(dummy_club_target, FitOptions())

    assert result.solver_status == "unsupported"
    assert "unsupported" in result.message.lower()
    assert "muscle_activations" not in result.meta
    assert not result.meta.get("inverse_surrogate_applied", False)


@pytest.mark.unit
def test_tile_status_is_experimental() -> None:
    """Tile status for MyoSuite is declared experimental in all registries."""
    models_yaml_path = REPO_ROOT / "src" / "config" / "models.yaml"
    data = yaml.safe_load(models_yaml_path.read_text(encoding="utf-8"))
    myo_models = [m for m in data.get("models", []) if m.get("id") == "myosim_suite"]
    assert len(myo_models) == 1
    assert myo_models[0]["launcher"]["status"] == "experimental"

    manifest = LauncherManifest.load()
    tile = manifest.get_tile("myosim_suite")
    assert tile is not None
    assert tile.status == "experimental"

    manifest_json_path = REPO_ROOT / "src" / "config" / "launcher_manifest.json"
    manifest_data = json.loads(manifest_json_path.read_text(encoding="utf-8"))
    json_tiles = [
        t for t in manifest_data.get("tiles", []) if t.get("id") == "myosim_suite"
    ]
    assert len(json_tiles) == 1
    assert json_tiles[0]["status"] == "experimental"


@pytest.mark.unit
def test_provider_registered_in_canonical_registry() -> None:
    """MyoSuite provider is properly registered under 'myosuite'."""
    prov = get_provider("myosuite")
    assert isinstance(prov, MyoSuiteFitSwingProvider)


@pytest.mark.unit
def test_gui_probes_on_open_and_displays_missing_dependencies(qapp: Any) -> None:
    """MyoSuite MainWidget probes on open and names missing wheel and submodule."""
    from src.engines.physics_engines.myosuite.python.gui import MainWidget

    widget = MainWidget()
    status_text = widget._status_label.text()
    assert status_text != "Engine status: not probed"
    assert "Engine status: unavailable" in status_text
    assert "pip install myosuite" in status_text
    assert "shared/models/myosuite/myo_sim" in status_text
    widget.cleanup()
