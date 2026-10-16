"""Joint-space results runs for the Model Explorer (#12878, #12915).

The engines save a swing as joint coordinates ``q`` over time on the shared
full-body specification. Three layouts are written today:

* ``dynamics_record.npz`` (``time_s``, ``q``) next to the fit's
  ``full_body_spec_hipcal_scaled.json`` and ``receipt.json`` (``backend``);
* a same-input closed-loop rollout ``<stem>.npz`` (``q``, optional
  ``time_s``) with its ``<stem>.json`` receipt and the ``bundle.npz`` it was
  replayed from (Drake, OpenSim, Pinocchio, MyoSuite);
* the same-input ``bundle.npz`` itself (its reference trajectory).

:func:`load_pose_frame_source` normalises all three into
:class:`PoseFrameSource`, which carries the specification so the explorer can
show the run on the current shared-appearance golfer. Marker-only archives keep
using :mod:`src.shared.python.motion_matching.video_frame_source`; both
sources raise a :class:`VideoSourceError` subclass, so callers catch one type.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.video_frame_source import VideoSourceError

__all__ = [
    "PoseFrameSource",
    "ResultRun",
    "ResultsSchemaError",
    "default_datastore_root",
    "discover_result_runs",
    "engine_title",
    "is_pose_results",
    "load_pose_frame_source",
]

#: Specification files a fit writes next to its record, in preference order.
SPEC_FILENAMES = (
    "full_body_spec_hipcal_scaled.json",
    "full_body_spec_hipcal.json",
    "full_body_spec.json",
)
BUNDLE_FILENAME = "bundle.npz"
RECORD_FILENAME = "dynamics_record.npz"
MARKER_CANDIDATE_FILENAME = "candidate.npz"
DATASTORE_ENV = "UD_SWING_DATASTORE"
FLEET_DATASTORE = Path("D:/FleetData/swing-datastore")
CLOSED_LOOP_SCHEMA_PREFIX = "same-input-closed-loop"
RENDER_ENGINE = "mujoco"
ENGINE_TITLES = {
    "mujoco": "MuJoCo",
    "drake": "Drake",
    "pinocchio": "Pinocchio",
    "opensim": "OpenSim",
    "myosuite": "MyoSuite",
    "simscape": "Simscape",
}
_MARKER_KEYS = frozenset({"markers_m", "model_markers_m"})
_MIN_FRAMES = 2


class ResultsSchemaError(VideoSourceError):
    """A results file lacks a required field; ``field`` names it."""

    def __init__(self, path: Path, field: str, detail: str = "") -> None:
        self.path = Path(path)
        self.field = field
        suffix = f": {detail}" if detail else ""
        super().__init__(
            f"{self.path.name}: missing or invalid required field '{field}'{suffix}"
        )


def engine_title(engine: str) -> str:
    """Display name of ``engine`` (``"drake"`` -> ``"Drake"``)."""
    return ENGINE_TITLES.get(engine.lower(), engine)


@dataclass(frozen=True)
class PoseFrameSource:
    """Joint trajectories of one run on the shared full-body specification.

    Invariants (checked on construction): ``time_s`` is ``(n,)`` with
    ``n >= 2``, finite and non-decreasing; ``q`` is ``(n, len(coordinate_order))``
    and finite; ``spec_bytes`` is a specification whose ``coordinate_order``
    equals ``coordinate_order``.
    """

    path: Path
    engine: str
    time_s: NDArray[np.float64]
    q: NDArray[np.float64]
    coordinate_order: tuple[str, ...]
    spec_bytes: bytes

    def __post_init__(self) -> None:
        n = len(self.time_s)
        if self.time_s.ndim != 1 or n < _MIN_FRAMES:
            raise ResultsSchemaError(
                self.path, "time_s", f"need at least {_MIN_FRAMES} samples, got {n}"
            )
        if not np.isfinite(self.time_s).all() or np.any(np.diff(self.time_s) < 0):
            raise ResultsSchemaError(
                self.path, "time_s", "must be finite and non-decreasing"
            )
        width = len(self.coordinate_order)
        if self.q.shape != (n, width):
            raise ResultsSchemaError(
                self.path,
                "q",
                f"expected shape ({n}, {width}) for {width} coordinates, "
                f"got {tuple(self.q.shape)}",
            )
        if not np.isfinite(self.q).all():
            raise ResultsSchemaError(self.path, "q", "contains NaN or inf")

    @property
    def frame_count(self) -> int:
        return len(self.time_s)

    @property
    def is_native_render(self) -> bool:
        """True when the run comes from the engine that draws it (MuJoCo)."""
        return self.engine == RENDER_ENGINE

    @property
    def playback_label(self) -> str:
        """Honest title: native MuJoCo result, or playback of another engine."""
        title = engine_title(self.engine)
        if self.is_native_render:
            return f"{title} result"
        return f"Playback of {title} result (rendered on the MuJoCo golfer model)"

    @property
    def spec(self) -> dict[str, Any]:
        return dict(json.loads(self.spec_bytes))


def _npz_keys(path: Path) -> set[str]:
    try:
        with np.load(path, allow_pickle=False) as data:
            return set(data.files)
    except (OSError, ValueError) as exc:
        raise VideoSourceError(
            f"{path.name} is not a readable NPZ archive: {exc}"
        ) from exc


def is_pose_results(path: str | Path) -> bool:
    """True when ``path`` is a joint-space run rather than a marker archive."""
    target = Path(path)
    if target.suffix.lower() != ".npz" or not target.is_file():
        return False
    try:
        keys = _npz_keys(target)
    except VideoSourceError:
        return False
    if keys & _MARKER_KEYS:
        return False
    if "manifest" in keys or ("q" in keys and _has_spec_near(target)):
        return True
    # A fit record or rollout is a joint-space run even when ``q`` or its
    # specification is missing: load it as one so validation names the field.
    return target.name == RECORD_FILENAME or _closed_loop_receipt(target) is not None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return {}


def _spec_bytes_near(path: Path) -> bytes | None:
    """The specification saved next to ``path`` (spec JSON, else the bundle)."""
    for name in SPEC_FILENAMES:
        candidate = path.parent / name
        if candidate.is_file():
            return candidate.read_bytes()
    bundle = path.parent / BUNDLE_FILENAME
    if bundle.is_file() and bundle != path:
        return _load_bundle(bundle).spec_bytes
    return None


def _load_bundle(path: Path) -> Any:
    from src.shared.python.motion_matching.same_input.bundle import InputBundle

    try:
        return InputBundle.load(path)
    except (KeyError, ValueError) as exc:
        raise ResultsSchemaError(path, "manifest", str(exc)) from exc


def _bundle_for_rollout(path: Path) -> Path:
    """The bundle a closed-loop rollout was replayed from.

    ``bundle.npz`` next to it, else a sibling whose stem prefixes the
    rollout's (``iron.npz`` for ``iron_closed_opensim.npz``).
    """
    sibling = path.parent / BUNDLE_FILENAME
    if sibling.is_file():
        return sibling
    for candidate in sorted(path.parent.glob("*.npz")):
        if candidate != path and path.stem.startswith(f"{candidate.stem}_"):
            return candidate
    raise ResultsSchemaError(
        path, "bundle", f"no {BUNDLE_FILENAME} next to the rollout to give the model"
    )


def _from_bundle(path: Path) -> PoseFrameSource:
    bundle = _load_bundle(path)
    q = np.asarray(bundle.reference_q, dtype=float)
    return PoseFrameSource(
        path=path,
        engine=str(bundle.reference_engine),
        time_s=np.arange(len(q), dtype=float) * float(bundle.dt_s),
        q=q,
        coordinate_order=tuple(bundle.coordinate_order),
        spec_bytes=bundle.spec_bytes,
    )


def _from_rollout(path: Path, receipt: Path) -> PoseFrameSource:
    from src.tools.native_viewer_export.core import load_receipt_rollout

    bundle = _load_bundle(_bundle_for_rollout(path))
    try:
        q, engine = load_receipt_rollout(receipt, bundle)
    except (OSError, ValueError) as exc:
        raise ResultsSchemaError(receipt, "bundle.spec_sha256", str(exc)) from exc
    with np.load(path, allow_pickle=False) as data:
        times = (
            np.asarray(data["time_s"], dtype=float)
            if "time_s" in data.files
            else np.arange(len(q), dtype=float) * float(bundle.dt_s)
        )
    return PoseFrameSource(
        path=path,
        engine=engine,
        time_s=times,
        q=q,
        coordinate_order=tuple(bundle.coordinate_order),
        spec_bytes=bundle.spec_bytes,
    )


def _record_engine(path: Path) -> str:
    receipt = _read_json(path.parent / "receipt.json")
    return str(receipt.get("engine") or receipt.get("backend") or "unknown").lower()


def _from_record(path: Path) -> PoseFrameSource:
    with np.load(path, allow_pickle=False) as data:
        for key in ("q", "time_s"):
            if key not in data.files:
                raise ResultsSchemaError(
                    path, key, f"archive has {', '.join(sorted(data.files))}"
                )
        q = np.asarray(data["q"], dtype=float)
        times = np.asarray(data["time_s"], dtype=float)
    spec_bytes = _spec_bytes_near(path)
    if spec_bytes is None:
        raise ResultsSchemaError(
            path,
            "spec",
            f"no {SPEC_FILENAMES[0]} or {BUNDLE_FILENAME} next to the record",
        )
    order = json.loads(spec_bytes).get("coordinate_order")
    if not order:
        raise ResultsSchemaError(path, "spec.coordinate_order")
    return PoseFrameSource(
        path=path,
        engine=_record_engine(path),
        time_s=times,
        q=q,
        coordinate_order=tuple(order),
        spec_bytes=spec_bytes,
    )


def _closed_loop_receipt(path: Path) -> Path | None:
    receipt = path.with_suffix(".json")
    schema = str(_read_json(receipt).get("schema", "")) if receipt.is_file() else ""
    return receipt if schema.startswith(CLOSED_LOOP_SCHEMA_PREFIX) else None


def load_pose_frame_source(path: str | Path) -> PoseFrameSource:
    """Load a joint-space run; validate it against its specification.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ResultsSchemaError: a required field (``q``, ``time_s``, ``spec``,
            ``bundle``, ...) is missing or malformed; ``.field`` names it.
    """
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Results file not found: {target}")
    keys = _npz_keys(target)
    if "manifest" in keys:
        return _from_bundle(target)
    receipt = _closed_loop_receipt(target)
    if receipt is not None:
        return _from_rollout(target, receipt)
    return _from_record(target)


@dataclass(frozen=True)
class ResultRun:
    """One playable run found in a results datastore.

    ``kind`` is ``"pose"`` (joint-space, shown on the golfer) or ``"markers"``
    (a marker archive, played through the marker frame source).
    """

    path: Path
    engine: str
    swing: str
    kind: str = "pose"

    @property
    def label(self) -> str:
        what = "golfer" if self.kind == "pose" else "markers"
        return (
            f"{engine_title(self.engine)} [{what}]  |  {self.swing}  |  "
            f"{self.path.parent.name}/{self.path.name}"
        )


def _rollout_engine(path: Path) -> str | None:
    receipt = _closed_loop_receipt(path)
    if receipt is None:
        return None
    return str(_read_json(receipt).get("engine", "unknown")).lower()


def _swing_name(path: Path, root: Path) -> str:
    parts = path.relative_to(root).parts
    return parts[0] if len(parts) > 1 else path.parent.name


def _classify(path: Path) -> tuple[str, str] | None:
    """``(engine, kind)`` of a playable archive, else ``None``."""
    if path.name == RECORD_FILENAME:
        return (_record_engine(path), "pose") if _has_spec_near(path) else None
    engine = _rollout_engine(path)
    if engine is not None:
        return engine, "pose"
    if path.name == MARKER_CANDIDATE_FILENAME:
        from src.shared.python.motion_matching.video_frame_source import (
            video_source_blocker,
        )

        if video_source_blocker(path) is None:
            return _record_engine(path), "markers"
    return None


def discover_result_runs(root: str | Path) -> list[ResultRun]:
    """Playable runs under ``root`` (searched recursively).

    Finds fit records (``dynamics_record.npz`` with a specification beside
    it), same-input rollouts with a receipt, and marker candidates
    (``candidate.npz``). Each run's ``swing`` is its first folder under
    ``root``. Raises ``FileNotFoundError`` when ``root`` is not a directory.
    """
    base = Path(root)
    if not base.is_dir():
        raise FileNotFoundError(f"Results folder not found: {base}")
    runs: list[ResultRun] = []
    for path in sorted(base.rglob("*.npz")):
        found = _classify(path)
        if found is not None:
            engine, kind = found
            runs.append(ResultRun(path, engine, _swing_name(path, base), kind))
    return runs


def _has_spec_near(path: Path) -> bool:
    names = (*SPEC_FILENAMES, BUNDLE_FILENAME)
    return any((path.parent / name).is_file() for name in names)


def default_datastore_root() -> Path | None:
    """The swing datastore's ``swings`` folder, if one is reachable.

    ``UD_SWING_DATASTORE`` (the datastore root) wins; otherwise the fleet
    store at ``D:/FleetData/swing-datastore`` is used when present.
    """
    configured = os.environ.get(DATASTORE_ENV)
    root = Path(configured) if configured else FLEET_DATASTORE
    swings = root / "swings"
    if swings.is_dir():
        return swings
    return root if configured and root.is_dir() else None
