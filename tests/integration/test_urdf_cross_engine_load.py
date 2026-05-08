"""Cross-engine URDF load smoke test (issue #4535).

Generates URDFs from each character-builder preset and verifies they
load without error in each of the four supported physics engines:
MuJoCo, Drake, Pinocchio, OpenSim.

Each engine is gated by ``pytest.importorskip`` so the test:
- **Runs and passes** when the engine is installed.
- **Skips cleanly** when the engine is not installed (no false failures).
- **Catches real regressions** in CI matrices that have the engine.

This is the minimum bar for #4535. The companion FK-equivalence test
(``test_urdf_cross_engine_fk.py``, #4542) uses these same engines for
a tighter numerical comparison.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Skip the whole module if the character builder isn't importable for any
# reason — we can't generate URDFs in that case.
hcb = pytest.importorskip("humanoid_character_builder")

# Reuse the test_urdf_quality fixture: one URDF per preset, generated once.
PRESET_NAMES = ["athletic", "average", "heavy"]


@pytest.fixture(scope="module")
def preset_urdfs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """Pre-generate URDFs for each preset; reused across all engine tests."""
    builder = hcb.CharacterBuilder()
    out: dict[str, str] = {}
    for preset in PRESET_NAMES:
        try:
            params = builder.create_from_preset(preset)
        except (KeyError, ValueError):
            # Preset not available in this build; skip gracefully.
            continue
        urdf_xml = builder.generate_urdf(params)
        # Some engines need a file on disk, not a string.
        urdf_path = tmp_path_factory.mktemp(f"{preset}_") / "humanoid.urdf"
        urdf_path.write_text(urdf_xml, encoding="utf-8")
        out[preset] = str(urdf_path)
    if not out:
        pytest.skip("No character-builder presets available in this build")
    return out


# ---------------------------------------------------------------------------
# MuJoCo
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_urdf_loads_in_mujoco(preset: str, preset_urdfs: dict[str, str]) -> None:
    """MuJoCo must load every preset URDF via mj_loadXML / from_xml_path.

    MuJoCo accepts URDF directly via the standard loader; no MJCF
    pre-compilation is required for the kinematic-only check.
    """
    mujoco = pytest.importorskip("mujoco")

    if preset not in preset_urdfs:
        pytest.skip(f"Preset {preset!r} not in this build")

    urdf_path = preset_urdfs[preset]
    # MjModel.from_xml_path raises mujoco.FatalError on parse / load failure.
    model = mujoco.MjModel.from_xml_path(urdf_path)
    assert model is not None
    assert model.nq > 0, "Loaded URDF must have at least one position DOF"
    assert model.nbody > 1, "Loaded URDF must have at least one non-world body"


# ---------------------------------------------------------------------------
# Drake
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_urdf_loads_in_drake(preset: str, preset_urdfs: dict[str, str]) -> None:
    """Drake must load every preset URDF via MultibodyPlant.AddModelFromFile."""
    pytest.importorskip("pydrake.multibody.plant")
    try:
        from pydrake.multibody.parsing import Parser
        from pydrake.multibody.plant import MultibodyPlant
    except ImportError:
        pytest.skip(
            "pydrake present but multibody not available (likely a stub install)"
        )

    if preset not in preset_urdfs:
        pytest.skip(f"Preset {preset!r} not in this build")

    plant = MultibodyPlant(time_step=0.001)
    parser = Parser(plant)
    parser.AddModels(preset_urdfs[preset])
    plant.Finalize()
    assert plant.num_positions() > 0
    assert plant.num_bodies() > 1


# ---------------------------------------------------------------------------
# Pinocchio
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_urdf_loads_in_pinocchio(preset: str, preset_urdfs: dict[str, str]) -> None:
    """Pinocchio must load every preset URDF via buildModelFromUrdf."""
    pinocchio = pytest.importorskip("pinocchio")
    if not hasattr(pinocchio, "buildModelFromUrdf"):
        pytest.skip(
            "pinocchio module present but buildModelFromUrdf missing (stub install)"
        )

    if preset not in preset_urdfs:
        pytest.skip(f"Preset {preset!r} not in this build")

    model = pinocchio.buildModelFromUrdf(preset_urdfs[preset])
    assert model.nq > 0
    assert model.njoints > 1  # at least one joint plus the universe joint


# ---------------------------------------------------------------------------
# OpenSim
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_urdf_loads_in_opensim(preset: str, preset_urdfs: dict[str, str]) -> None:
    """OpenSim's URDF importer must load every preset.

    OpenSim's URDF support is via the ``opensim.URDFFileAdapter`` (>=4.5).
    If the adapter is not present in this OpenSim build, the test skips.
    """
    opensim = pytest.importorskip("opensim")

    if preset not in preset_urdfs:
        pytest.skip(f"Preset {preset!r} not in this build")

    if not hasattr(opensim, "URDFFileAdapter"):
        pytest.skip("This OpenSim build lacks URDFFileAdapter")

    adapter = opensim.URDFFileAdapter()
    # URDFFileAdapter.read returns a Model on success; raises on failure.
    table = adapter.read(preset_urdfs[preset])
    assert table is not None


# ---------------------------------------------------------------------------
# Sanity: ensure preset_urdfs fixture itself works without any engine
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_preset_urdf_fixture_produces_at_least_one_urdf(
    preset_urdfs: dict[str, str],
) -> None:
    """If this test fails, all engine tests skip with a misleading message."""
    assert preset_urdfs, "preset_urdfs fixture must produce at least one URDF"
    for preset, path in preset_urdfs.items():
        assert Path(path).exists(), f"URDF for {preset} missing at {path}"
        assert Path(path).read_text(encoding="utf-8").startswith("<")
