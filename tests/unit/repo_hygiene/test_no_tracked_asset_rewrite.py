"""Guard: loading models must not rewrite tracked source assets.

See UpstreamDrift issue #9182.

``ModelLibrary.get_human_model("mujoco_humanoid")`` used to materialise the
embedded MJCF string over the *tracked* asset
``src/shared/urdf/human_models/mujoco_humanoid/model.xml``. Any test run that
touched that code path left ``git status`` dirty, which pollutes developer and
agent workflows (spurious stashes, blocked rebases, false PR diffs, and
``git status --short`` checks that agents use to detect in-progress work).

Derived artifacts now go to :func:`get_derived_model_cache_dir` instead. These
tests pin that behaviour by hashing the tracked asset before and after
exercising the loader.

Follow-up (issue #9220): with the rewrite gone, that asset had no reader left
anywhere in the repository -- it was derived output kept alive only by the
rewrite -- so it was deleted. The loader path must therefore never re-create
it either, which :func:`test_source_tree_model_xml_stays_absent` pins.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.tools.model_explorer.model_library import (
    MODEL_CACHE_DIR_ENV_VAR,
    ModelLibrary,
    get_derived_model_cache_dir,
)

_PROJECT_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()
)

TRACKED_MODEL_XML = (
    _PROJECT_ROOT / "src" / "shared" / "urdf" / "human_models" / "mujoco_humanoid"
) / "model.xml"


def _digest(path: Path) -> str | None:
    """Return the SHA-256 of ``path``, or ``None`` when it does not exist."""
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.unit
def test_embedded_model_cache_dir_is_outside_the_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default derived-model cache must not live inside the checkout."""
    monkeypatch.delenv(MODEL_CACHE_DIR_ENV_VAR, raising=False)
    cache_dir = get_derived_model_cache_dir()
    assert not cache_dir.is_relative_to(_PROJECT_ROOT), (
        f"Derived model cache {cache_dir} is inside the repository; "
        "generated files would dirty the working tree."
    )

    monkeypatch.setenv(MODEL_CACHE_DIR_ENV_VAR, str(tmp_path / "cache"))
    assert get_derived_model_cache_dir() == (tmp_path / "cache").resolve()


@pytest.mark.unit
def test_get_human_model_does_not_rewrite_tracked_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loading the embedded humanoid must leave the tracked asset untouched."""
    monkeypatch.setenv(MODEL_CACHE_DIR_ENV_VAR, str(tmp_path / "cache"))

    before_digest = _digest(TRACKED_MODEL_XML)
    before_mtime = (
        TRACKED_MODEL_XML.stat().st_mtime_ns if TRACKED_MODEL_XML.exists() else None
    )

    library = ModelLibrary()
    resolved = library.get_human_model("mujoco_humanoid")

    assert resolved is not None
    assert resolved.exists()
    assert not resolved.is_relative_to(_PROJECT_ROOT), (
        f"Embedded model was materialised at {resolved}, inside the repository."
    )

    assert _digest(TRACKED_MODEL_XML) == before_digest, (
        f"{TRACKED_MODEL_XML} was rewritten by ModelLibrary.get_human_model()."
    )
    after_mtime = (
        TRACKED_MODEL_XML.stat().st_mtime_ns if TRACKED_MODEL_XML.exists() else None
    )
    assert after_mtime == before_mtime


@pytest.mark.unit
def test_explicit_base_path_keeps_derived_output_under_that_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller-supplied ``base_path`` still contains all generated output."""
    monkeypatch.delenv(MODEL_CACHE_DIR_ENV_VAR, raising=False)
    base = tmp_path / "models"
    library = ModelLibrary(base_path=base)

    resolved = library.get_human_model("mujoco_humanoid")

    assert resolved is not None
    assert resolved.is_relative_to(base)
    assert not resolved.is_relative_to(library.human_models_path)


@pytest.mark.unit
def test_source_tree_model_xml_stays_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deleted orphan must not reappear in the source tree (issue #9220).

    ``src/shared/urdf/human_models/mujoco_humanoid/model.xml`` was derived
    output with no reader; it was removed. Loading the embedded humanoid must
    not resurrect it.
    """
    monkeypatch.setenv(MODEL_CACHE_DIR_ENV_VAR, str(tmp_path / "cache"))

    assert not TRACKED_MODEL_XML.exists(), (
        f"{TRACKED_MODEL_XML} is back in the source tree. It is derived output "
        "with no reader; regenerate it into the model cache instead."
    )

    ModelLibrary().get_human_model("mujoco_humanoid")

    assert not TRACKED_MODEL_XML.exists()


@pytest.mark.unit
def test_repeated_loads_do_not_churn_the_cache_file(tmp_path: Path) -> None:
    """Re-loading unchanged content must not rewrite the cached file."""
    library = ModelLibrary(base_path=tmp_path / "models", cache_path=tmp_path / "cache")

    first = library.get_human_model("mujoco_humanoid")
    assert first is not None
    first_mtime = first.stat().st_mtime_ns

    second = library.get_human_model("mujoco_humanoid")

    assert second == first
    assert second.stat().st_mtime_ns == first_mtime


@pytest.mark.unit
def test_derived_humanoid_mjcf_is_not_committed_to_the_source_tree() -> None:
    """The generated MJCF must not be re-committed as a tracked asset (#9220).

    The canonical humanoid MJCF is the string embedded in
    ``get_embedded_mujoco_models()``. A committed copy under
    ``human_models/`` is derived output that nothing reads: since #9219 the
    loader resolves ``mujoco_humanoid`` straight to the derived cache, and no
    other code path opens this file. The committed copy had already drifted
    from its generator (render-side blocks only), which is exactly the trap
    this guard exists to prevent.
    """
    assert not TRACKED_MODEL_XML.exists(), (
        f"{TRACKED_MODEL_XML} is derived output committed as a tracked asset. "
        "Nothing reads it and it silently drifts from the embedded MJCF that "
        "generates it. Delete it, or give it an explicit regeneration target "
        "and a freshness check."
    )
