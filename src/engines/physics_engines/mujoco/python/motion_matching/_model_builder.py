"""Compiled-MJCF model builder with disk-backed cache (issue #4109).

Single entry point that the rest of the parity package uses to obtain a
compiled :class:`mujoco.MjModel`. Subsequent calls for the same variant skip
XML parsing entirely by loading a previously-written ``.mjb`` (MuJoCo Binary)
artifact from a per-XML-hash cache directory.

Why a disk cache?
-----------------
``mujoco.MjModel.from_xml_string`` parses the MJCF, resolves assets, and runs
the compiler — measured at 2--4 ms per variant on the parity models. A binary
``.mjb`` reload via :func:`mujoco.MjModel.from_binary_path` is a bare deserialise
(~0.5 ms). Across the fit driver, target-synthesis oracle, viz, and the test
suite the same model is rebuilt dozens of times per process and hundreds of
times per dataset sweep, so the saving compounds.

Cache invalidation
------------------
The cache filename embeds ``sha256(xml_string)``. Editing any of the three
``_golf_swing_*_xml.py`` generators changes the hash and forces a recompile —
no manual eviction step is required. The in-process ``functools.lru_cache``
also keys on the variant name so repeated calls within a process skip even
the disk read.

Public surface
--------------
* :func:`load_model(variant)` -> :class:`mujoco.MjModel` -- thin convenience
  wrapper matching the deliverable signature requested in the parity prompt.
* :func:`build_model(variant)` -> :class:`CompiledModel` -- richer entry point
  matching the issue spec; returns the model alongside a ``data`` factory and
  resolved joint/body identifiers.
* :class:`CompiledModel` -- frozen dataclass holding everything callers need
  without re-running ``mj_id2name`` lookups.
* :func:`clear_cache()` -- testing utility; nukes both the in-process LRU and
  the disk cache directory.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

import mujoco

from src.engines.physics_engines.mujoco._golf_swing_advanced_xml import (
    ADVANCED_BIOMECHANICAL_GOLF_SWING_XML,
)
from src.engines.physics_engines.mujoco._golf_swing_full_body_xml import (
    FULL_BODY_GOLF_SWING_XML,
)
from src.engines.physics_engines.mujoco._golf_swing_upper_body_xml import (
    UPPER_BODY_GOLF_SWING_XML,
)

_LOG = logging.getLogger(__name__)

Variant = Literal["advanced", "full_body", "upper_body"]

_VARIANT_XML: dict[Variant, str] = {
    "advanced": ADVANCED_BIOMECHANICAL_GOLF_SWING_XML,
    "full_body": FULL_BODY_GOLF_SWING_XML,
    "upper_body": UPPER_BODY_GOLF_SWING_XML,
}

# Body-name aliases per variant. ``upper_body`` and ``full_body`` use ``club``
# as the grip body; ``advanced`` was authored with a more anatomical
# ``club_grip`` name. Fallback covers either spelling so callers always see
# the same ``CompiledModel.club_grip_body_id`` attribute.
_GRIP_BODY_CANDIDATES: tuple[str, ...] = ("club_grip", "club")
_HEAD_BODY_CANDIDATES: tuple[str, ...] = ("clubhead", "club_head")


def _default_cache_dir() -> Path:
    """Return the on-disk cache directory.

    Honours ``UPSTREAMDRIFT_MUJOCO_CACHE_DIR`` so CI and tests can isolate
    state. Defaults to a repo-local ``.cache/mujoco_mjb`` so the artifacts are
    co-located with the source they derive from and are trivially gitignored.
    """
    override = os.environ.get("UPSTREAMDRIFT_MUJOCO_CACHE_DIR")
    if override:
        return Path(override)
    # Walk parents up: motion_matching/ -> python/ -> mujoco/ -> physics_engines/ ...
    # Anchor on the package itself so editable installs and worktrees behave.
    return Path(__file__).resolve().parents[6] / ".cache" / "mujoco_mjb"


def _xml_for(variant: Variant) -> str:
    if variant not in _VARIANT_XML:
        raise ValueError(
            f"unknown MuJoCo model variant {variant!r}; "
            f"expected one of {tuple(_VARIANT_XML)}"
        )
    return _VARIANT_XML[variant]


def _hash_xml(xml: str) -> str:
    return hashlib.sha256(xml.encode("utf-8")).hexdigest()


def _cache_path(variant: Variant, xml_hash: str, cache_dir: Path) -> Path:
    # Variant prefix makes hand-inspection of the cache directory readable;
    # the sha256 suffix is what actually drives invalidation.
    return cache_dir / f"{variant}-{xml_hash}.mjb"


def _resolve_body_id(model: mujoco.MjModel, candidates: tuple[str, ...]) -> int:
    """Return the body id for the first matching name. -1 means missing."""
    for name in candidates:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id != -1:
            return body_id
    return -1


def _joint_names(model: mujoco.MjModel) -> list[str]:
    """Joint names in MuJoCo joint-id order (deterministic across calls)."""
    names: list[str] = []
    for jid in range(model.njnt):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        # Anonymous joints (e.g. freejoints without a name=) come back as None;
        # encode positionally so the list length always equals ``njnt``.
        names.append(nm if nm is not None else f"<unnamed_joint_{jid}>")
    return names


def _compile_xml(xml: str) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(xml)


def _load_or_compile(variant: Variant, cache_dir: Path) -> mujoco.MjModel:
    """Return a compiled model, populating the on-disk cache as a side effect.

    Cold path: parse XML, write ``.mjb``, return the in-memory model so we
    never pay the deserialise round-trip on the very first call.
    Warm path: read ``.mjb`` directly and skip XML parsing entirely.
    """
    xml = _xml_for(variant)
    xml_hash = _hash_xml(xml)
    path = _cache_path(variant, xml_hash, cache_dir)

    if path.is_file():
        try:
            return mujoco.MjModel.from_binary_path(str(path))
        except Exception as exc:  # noqa: BLE001 - cache poisoning is recoverable
            # A corrupt or partially-written .mjb (e.g. interrupted CI job)
            # must not be fatal. Drop it and recompile from XML.
            _LOG.warning(
                "MuJoCo cache hit at %s but reload failed (%s); recompiling.",
                path,
                exc,
            )
            try:
                path.unlink()
            except OSError:
                pass

    model = _compile_xml(xml)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Write to a sibling temp path then rename so a crash mid-write cannot
        # leave a half-formed file in the cache. ``os.replace`` is atomic on
        # POSIX and Windows for paths on the same filesystem.
        tmp = path.with_suffix(path.suffix + ".tmp")
        mujoco.mj_saveModel(model, str(tmp))
        os.replace(tmp, path)
    except OSError as exc:
        # Read-only filesystems (some CI sandboxes) should not break the build;
        # the model is already compiled in memory.
        _LOG.warning("MuJoCo cache write to %s failed (%s); continuing.", path, exc)
    return model


@dataclass(frozen=True)
class CompiledModel:
    """Compiled MJCF + the lookups every motion-matching caller needs.

    Frozen so callers cannot accidentally mutate joint-id ordering between
    fit iterations. ``make_data`` is exposed as an attribute (not a method)
    to keep call sites at LOD <= 2 (``cm.make_data()``).
    """

    variant: Variant
    model: mujoco.MjModel
    joint_names: list[str]
    joint_ids: list[int]
    club_grip_body_id: int
    club_head_body_id: int
    xml_hash: str
    make_data: Callable[[], mujoco.MjData] = field(compare=False, repr=False)


@lru_cache(maxsize=4)
def build_model(variant: Variant = "full_body") -> CompiledModel:
    """Return a cached :class:`CompiledModel` for ``variant``.

    The ``maxsize=4`` LRU cache mirrors the issue spec; with three known
    variants there is no realistic eviction pressure. Process-wide reuse
    means the disk cache is only consulted on the very first call per
    variant per process.

    Joint-id ordering convention
    ----------------------------
    MuJoCo assigns joint ids in DFS body-tree order (root-first), which is
    deterministic given a fixed XML. ``CompiledModel.joint_names`` is the
    list of names in that order; ``joint_ids`` is the matching ``range``.
    Callers indexing ``data.qpos`` / ``data.qvel`` / ``data.ctrl`` should
    use ``model.jnt_qposadr`` / ``jnt_dofadr`` / actuator ids respectively
    rather than re-deriving offsets from this list.
    """
    cache_dir = _default_cache_dir()
    model = _load_or_compile(variant, cache_dir)
    names = _joint_names(model)
    grip_id = _resolve_body_id(model, _GRIP_BODY_CANDIDATES)
    head_id = _resolve_body_id(model, _HEAD_BODY_CANDIDATES)
    if grip_id == -1:
        raise RuntimeError(
            f"variant {variant!r}: no grip body found "
            f"(looked for {_GRIP_BODY_CANDIDATES})"
        )
    if head_id == -1:
        raise RuntimeError(
            f"variant {variant!r}: no club-head body found "
            f"(looked for {_HEAD_BODY_CANDIDATES})"
        )
    return CompiledModel(
        variant=variant,
        model=model,
        joint_names=names,
        joint_ids=list(range(model.njnt)),
        club_grip_body_id=grip_id,
        club_head_body_id=head_id,
        xml_hash=_hash_xml(_xml_for(variant)),
        make_data=lambda m=model: mujoco.MjData(m),
    )


def load_model(variant: Variant = "full_body") -> mujoco.MjModel:
    """Convenience wrapper: return just the :class:`mujoco.MjModel`.

    Matches the deliverable signature in the issue prompt. Callers needing
    the joint-name list, body ids, or a fresh ``MjData`` should prefer
    :func:`build_model` to avoid re-running ``mj_id2name`` and
    :class:`mujoco.MjData` allocation themselves.
    """
    return build_model(variant).model


def clear_cache() -> None:
    """Drop the in-process LRU and remove all on-disk ``.mjb`` artifacts.

    Used by tests that need to measure cold-load latency or confirm that
    XML hash invalidation works. Safe to call when the cache directory
    does not yet exist.
    """
    build_model.cache_clear()
    cache_dir = _default_cache_dir()
    if not cache_dir.is_dir():
        return
    for entry in cache_dir.iterdir():
        if entry.suffix in {".mjb", ".tmp"}:
            try:
                entry.unlink()
            except OSError:  # noqa: PERF203 - best-effort cleanup
                _LOG.debug("could not delete cache entry %s", entry)
