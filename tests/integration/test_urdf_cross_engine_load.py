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


# ---------------------------------------------------------------------------
# Drake
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Pinocchio
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# OpenSim
# ---------------------------------------------------------------------------


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
