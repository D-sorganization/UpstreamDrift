"""Cache + identifier tests for ``mujoco/python/motion_matching/_model_builder.py``.

Covers issue #4109. Three behaviours are exercised:

1. Cold load (no cache present) compiles from XML and stays under the spec
   ceiling of 200 ms. Warm load (cache hit) drops below 20 ms.
2. The cache filename embeds the XML's sha256, so editing the source XML
   invalidates the cached ``.mjb`` automatically.
3. The :class:`CompiledModel` exposes a deterministic joint-name ordering
   and resolves the grip + clubhead body ids for every variant.

Tests run only when the ``mujoco`` package imports successfully; the marker
``requires_mujoco`` lets CI deselect the suite on environments that do not
ship the binary wheel (e.g. some sandboxed Linux runners).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")

# Imports happen after ``importorskip`` so the suite is skip-safe on hosts
# without the MuJoCo wheel installed.
from src.engines.physics_engines.mujoco.python.motion_matching._model_builder import (  # noqa: E402
    CompiledModel,
    _cache_path,
    _hash_xml,
    _xml_for,
    build_model,
    clear_cache,
    load_model,
)

pytestmark = pytest.mark.requires_mujoco

VARIANTS = ("upper_body", "full_body", "advanced")

# Performance budgets from the issue acceptance criteria. Generous head-room
# above the measured ~4 ms cold / ~0.5 ms warm to absorb CI noise (Windows
# self-hosted runners + antivirus scans).
COLD_LOAD_BUDGET_S = 0.200
WARM_LOAD_BUDGET_S = 0.020


@pytest.fixture
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``_default_cache_dir`` at a tmp directory and clear LRU state."""
    monkeypatch.setenv("UPSTREAMDRIFT_MUJOCO_CACHE_DIR", str(tmp_path))
    clear_cache()
    yield tmp_path
    clear_cache()


def test_load_model_returns_mjmodel(isolated_cache: Path) -> None:
    model = load_model("full_body")
    assert isinstance(model, mujoco.MjModel)
    assert model.nu > 0, "full_body variant must declare actuators"


@pytest.mark.parametrize("variant", VARIANTS)
def test_build_model_returns_compiled_model(variant: str, isolated_cache: Path) -> None:
    cm = build_model(variant)
    assert isinstance(cm, CompiledModel)
    assert cm.variant == variant
    assert isinstance(cm.model, mujoco.MjModel)
    assert cm.club_grip_body_id >= 0
    assert cm.club_head_body_id >= 0
    assert cm.club_grip_body_id != cm.club_head_body_id
    assert len(cm.joint_names) == cm.model.njnt
    # joint_ids are simply range(njnt) but exposing them keeps callers from
    # re-deriving the convention.
    assert cm.joint_ids == list(range(cm.model.njnt))


@pytest.mark.parametrize("variant", VARIANTS)
def test_make_data_yields_independent_mjdata(
    variant: str, isolated_cache: Path
) -> None:
    cm = build_model(variant)
    d1 = cm.make_data()
    d2 = cm.make_data()
    assert isinstance(d1, mujoco.MjData)
    assert d1 is not d2  # fresh allocation each call


def test_build_model_caches_within_process(isolated_cache: Path) -> None:
    """Repeated calls within one process return the same object via lru_cache."""
    a = build_model("upper_body")
    b = build_model("upper_body")
    assert a is b


def test_unknown_variant_raises(isolated_cache: Path) -> None:
    with pytest.raises(ValueError, match="unknown MuJoCo model variant"):
        build_model("torso_only")  # type: ignore[arg-type]


@pytest.mark.parametrize("variant", VARIANTS)
def test_cold_load_under_200ms(variant: str, isolated_cache: Path) -> None:
    """First load (cache miss) must beat the 200 ms acceptance budget."""
    # Make absolutely sure no .mjb is on disk yet.
    assert not any(isolated_cache.glob("*.mjb"))
    t0 = time.perf_counter()
    cm = build_model(variant)
    elapsed = time.perf_counter() - t0
    assert isinstance(cm.model, mujoco.MjModel)
    assert elapsed < COLD_LOAD_BUDGET_S, (
        f"{variant}: cold load took {elapsed * 1000:.1f} ms, "
        f"budget {COLD_LOAD_BUDGET_S * 1000:.0f} ms"
    )
    # Cold load must populate the on-disk cache for the next process.
    cached = list(isolated_cache.glob(f"{variant}-*.mjb"))
    assert len(cached) == 1, f"expected exactly one cached .mjb, got {cached}"


@pytest.mark.parametrize("variant", VARIANTS)
def test_warm_load_under_20ms(variant: str, isolated_cache: Path) -> None:
    """Second load (cache hit) must beat the 20 ms acceptance budget."""
    # Prime the cache, then drop the in-process LRU so the next call has
    # to consult the disk.
    build_model(variant)
    build_model.cache_clear()

    # Run a few iterations and take the min — first call after cache_clear
    # may include filesystem stat overhead from the OS page cache.
    samples: list[float] = []
    for _ in range(3):
        build_model.cache_clear()
        t0 = time.perf_counter()
        build_model(variant)
        samples.append(time.perf_counter() - t0)
    best = min(samples)
    assert best < WARM_LOAD_BUDGET_S, (
        f"{variant}: warm load best-of-3 was {best * 1000:.1f} ms, "
        f"budget {WARM_LOAD_BUDGET_S * 1000:.0f} ms"
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_cache_filename_embeds_xml_sha256(variant: str, isolated_cache: Path) -> None:
    build_model(variant)
    expected_hash = _hash_xml(_xml_for(variant))
    cached = list(isolated_cache.glob("*.mjb"))
    assert len(cached) == 1
    assert expected_hash in cached[0].name
    assert cached[0].name.startswith(f"{variant}-")


def test_xml_change_invalidates_cache(
    isolated_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutating the XML string for a variant produces a fresh cache entry."""
    from src.engines.physics_engines.mujoco.python.motion_matching import _model_builder

    original_xml = _xml_for("upper_body")
    build_model("upper_body")
    first_hash = _hash_xml(original_xml)
    assert (isolated_cache / f"upper_body-{first_hash}.mjb").is_file()

    # Simulate an edit to the source XML by patching the lookup table. Any
    # whitespace edit changes sha256, which is the property under test.
    mutated = original_xml.replace(
        '<option timestep="0.002"', '<option timestep="0.0021"', 1
    )
    assert mutated != original_xml, "test setup: substitution had no effect"
    monkeypatch.setitem(_model_builder._VARIANT_XML, "upper_body", mutated)
    build_model.cache_clear()

    cm = build_model("upper_body")
    second_hash = _hash_xml(mutated)
    assert first_hash != second_hash
    assert cm.xml_hash == second_hash
    # Both .mjb files must coexist - the hash IS the cache key.
    assert (isolated_cache / f"upper_body-{first_hash}.mjb").is_file()
    assert (isolated_cache / f"upper_body-{second_hash}.mjb").is_file()


def test_corrupt_cache_recovers_via_recompile(isolated_cache: Path) -> None:
    """A truncated .mjb on disk must not be fatal; we recompile from XML."""
    build_model("upper_body")
    cached = list(isolated_cache.glob("upper_body-*.mjb"))
    assert len(cached) == 1
    cached[0].write_bytes(b"not a real mjb")  # poison the cache
    build_model.cache_clear()

    cm = build_model("upper_body")  # must not raise
    assert isinstance(cm.model, mujoco.MjModel)


def test_clear_cache_removes_disk_artifacts(isolated_cache: Path) -> None:
    build_model("upper_body")
    assert any(isolated_cache.glob("*.mjb"))
    clear_cache()
    assert not any(isolated_cache.glob("*.mjb"))


@pytest.mark.parametrize("variant", VARIANTS)
def test_joint_names_deterministic_across_calls(
    variant: str, isolated_cache: Path
) -> None:
    cm_a = build_model(variant)
    build_model.cache_clear()
    cm_b = build_model(variant)
    assert cm_a.joint_names == cm_b.joint_names


def test_cache_path_helper_uses_variant_prefix(tmp_path: Path) -> None:
    p = _cache_path("full_body", "deadbeef" * 8, tmp_path)
    assert p.parent == tmp_path
    assert p.name.startswith("full_body-")
    assert p.name.endswith(".mjb")
